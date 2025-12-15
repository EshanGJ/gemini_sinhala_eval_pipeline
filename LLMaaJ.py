import streamlit as st
from langfuse import Langfuse, observe, propagate_attributes
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_bytes

# Load environment variables
load_dotenv(".env")

# Page configuration
st.set_page_config(
    page_title="Document Analyzer with Gemini",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Langfuse
@st.cache_resource
def init_langfuse():
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL"),
        timeout=300,
        sample_rate=1.0 ,
    )

langfuse = init_langfuse()

# Initialize Gemini client
@st.cache_resource
def init_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"), timeot=600)

# Prompt templates
PROMPTS = {
    "Simple Markdown": """## Task
Analyze the document pdf and provide:
1. **Transcribe** all text exactly as written
2. **Describe** any diagrams, tables, or visual elements

## Output
Respond in **Markdown format**:
- Use headings, lists, and tables to match the document structure
- For diagrams, describe the flow (e.g., A → B → C)
- Mark unclear text as `[unclear]`""",
    "Detailed Markdown": """# Document Analysis

## Task
Analyze the document and provide:
1. A markdown description of the page content (diagrams, tables, images, etc.)
2. Complete transcription of all text

## Output Format

### 1. Page Description
Provide a brief markdown summary describing:
- Document type and layout
- Visual elements present (diagrams, tables, charts, images)
- Overall structure

### 2. Transcription
Output using the `document_elements` array:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., B1, N1) |
| `text` | string | The transcribed content |
| `type` | enum | TITLE, PARAGRAPH, LIST, TABLE_CELL, DIAGRAM_NODE, KEY_VALUE_PAIR |
| `relations` | array | Connections: `target_id` and `relation_type` (FLOWS_TO, VALUE_FOR) |

## Instructions
- **Text:** Transcribe all visible text
- **Diagrams:** Describe flow and connections using FLOWS_TO
- **Tables/Forms:** Link labels to values using VALUE_FOR""",
    "JSON Structure": """Task: Identify all meaningful blocks of content and extract the structural relationships between them.

JSON Schema: Output using the 'document_elements' array, where each object contains:
- id (string): Unique identifier (e.g., B1, N1)
- text (string): The transcribed content
- type (enum): TITLE, PARAGRAPH, LIST, TABLE_CELL, DIAGRAM_NODE, DIAGRAM_ARROW, KEY_VALUE_PAIR
- relations (array): Semantic connections with:
- target_id (string): Connected element's id
- relation_type (enum): FLOWS_TO, IS_LABEL_FOR, VALUE_FOR

Specific Instructions:
1. For diagrams: Use DIAGRAM_NODE for shapes, DIAGRAM_ARROW for lines. Link with FLOWS_TO.
2. For forms/tables: Use KEY_VALUE_PAIR. Link labels to values using VALUE_FOR.""",
    "Custom": ""
}


import mimetypes
from google import genai
from google.genai import types

def init_gemini_client_v1alpha():
    """Initializes the client specifically for v1alpha to support media_resolution."""
    # Ensure your API key is set in environment variables or passed here
    return genai.Client(http_options={'api_version': 'v1alpha'}, api_key=os.getenv("GOOGLE_API_KEY"))

@observe(name="streamlit-gemini-call", as_type="generation", capture_input=False, capture_output=True)
def call_gemini(
    input_prompt, 
    ground_truth, 
    model_id="gemini-2.0-flash", 
    file_paths=None, 
    generation_config=None, 
    session_id="streamlit_session",
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH # Default to High
):
    """Process files with Gemini (Inline) to support media_resolution."""
    
    with propagate_attributes(
        user_id="streamlit_user",
        session_id=session_id,
        tags=["gemini", "streamlit", "document-analysis"],
        metadata={"source": "streamlit_app"},
        version="1.0.0",
    ):
        # 1. We must use v1alpha for media_resolution support
        client = init_gemini_client_v1alpha()
        
        if file_paths is None:
            file_paths = []
        elif isinstance(file_paths, str):
            file_paths = [file_paths]
        
        media_parts = []
        
        try:
            # 2. Prepare Inline Data Parts (Instead of uploading)
            for file_path in file_paths:
                # Guess mime type (e.g., 'image/jpeg') based on file extension
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = "application/octet-stream" # Fallback

                with open(file_path, 'rb') as f:
                    file_bytes = f.read()

                # Create the part with the specific resolution setting
                part = types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type,
                    media_resolution=media_resolution
                )
                media_parts.append(part)
            
            # Combine text prompt with image parts
            contents = [input_prompt] + media_parts

            # 3. Generate Content
            if generation_config:
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=generation_config,
                )
            else:
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                )
            
            # 4. Extract Usage Metadata
            usage_meta = response.usage_metadata
            prompt_tokens = usage_meta.prompt_token_count or 0
            candidate_tokens = usage_meta.candidates_token_count or 0
            thought_tokens = getattr(usage_meta, 'thoughts_token_count', 0) or 0
            cached_tokens = usage_meta.cached_content_token_count or 0
            total_tokens = usage_meta.total_token_count or 0

            effective_output_tokens = candidate_tokens + thought_tokens

            # 5. Log to Langfuse
            langfuse.update_current_trace(
                input={
                    "prompt": input_prompt,
                    # We log file names, but remember data was sent inline
                    "files": [fp for fp in file_paths], 
                    "resolution_setting": str(media_resolution)
                },
                output=response.text,
                metadata={
                    "ground_truth": ground_truth,
                    "model_id": model_id,
                    "method": "inline_data"
                }
            )

            # 6. Calculate Costs 
            # Note: Prices here are examples; verify current Gemini 2.0 Flash pricing
            INPUT_PRICE_PER_TOKEN = 0.3 / 1000000 # Verify for v1alpha/2.0
            OUTPUT_PRICE_PER_TOKEN = 2.5 / 1000000 # Verify for v1alpha/2.0
            CACHING_PRICE_PER_TOKEN = 0.03 / 1000000

            input_cost = prompt_tokens * INPUT_PRICE_PER_TOKEN
            output_cost = effective_output_tokens * OUTPUT_PRICE_PER_TOKEN
            cache_read_input_cost = cached_tokens * CACHING_PRICE_PER_TOKEN
            total_cost = input_cost + output_cost + cache_read_input_cost
            
            langfuse.update_current_generation(
                cost_details={
                    "input": input_cost,
                    "cache_read_input_tokens": cache_read_input_cost,
                    "output": output_cost,
                    "total": total_cost,
                },
                usage_details={
                    "input": prompt_tokens,
                    "output": effective_output_tokens,
                    "cache_read_input_tokens": cached_tokens 
                },
            )

            usage_info = {
                "prompt_tokens": prompt_tokens,
                "output_tokens": effective_output_tokens,
                "cached_tokens": cached_tokens,
                "total_tokens": total_tokens,
                "total_cost": total_cost
            }

            return response.text, ground_truth, usage_info
            
        except Exception as e:
            raise e

# import io
# import time

# @observe(name="streamlit-gemini-call", as_type="generation", capture_input=False, capture_output=True)
# def call_gemini(
#     input_prompt, 
#     ground_truth, 
#     model_id="gemini-2.0-flash", 
#     file_paths=None, 
#     generation_config=None, 
#     session_id="streamlit_session",
#     media_resolution=None # We handle resolution via DPI manually now
# ):
#     """
#     Process files with local optimization to prevent 15-minute hangs.
#     """
    
#     with propagate_attributes(
#         user_id="streamlit_user",
#         session_id=session_id,
#         tags=["gemini", "streamlit", "document-analysis"],
#         metadata={"source": "streamlit_app"},
#         version="1.0.0",
#     ):
#         # Initialize client with a safe timeout
#         client = init_gemini_client_v1alpha()
        
#         if file_paths is None:
#             file_paths = []
#         elif isinstance(file_paths, str):
#             file_paths = [file_paths]
        
#         media_parts = []
        
#         # UI Status Container to show progress
#         with st.status("Processing document...", expanded=True) as status:
            
#             try:
#                 for file_path in file_paths:
#                     mime_type, _ = mimetypes.guess_type(file_path)
                    
#                     # --- PDF HANDLING ---
#                     if mime_type == "application/pdf" or file_path.lower().endswith(".pdf"):
#                         status.write(f"📄 Converting PDF: {Path(file_path).name}")
                        
#                         with open(file_path, 'rb') as f:
#                             pdf_bytes = f.read()
                        
#                         # OPTIMIZATION: Use 200 DPI (Sweet spot for Gemini) 
#                         # 300 DPI creates massive files that choke the API.
#                         # 200 DPI is still "High Res" for an LLM.
#                         start_time = time.time()
#                         images = convert_from_bytes(pdf_bytes, dpi=200, fmt='jpeg')
                        
#                         status.write(f"✅ Converted {len(images)} pages in {time.time() - start_time:.1f}s")
                        
#                         # Process images
#                         for i, img in enumerate(images):
#                             status.write(f"Processing Page {i+1}...")
                            
#                             img_byte_arr = io.BytesIO()
#                             # OPTIMIZATION: JPEG Quality 85 reduces size by 4x without visible loss
#                             img.save(img_byte_arr, format='JPEG', quality=85) 
#                             img_bytes = img_byte_arr.getvalue()
                            
#                             # Create part (We set resolution here via pixel density, not API param)
#                             part = types.Part.from_bytes(
#                                 data=img_bytes,
#                                 mime_type="image/jpeg"
#                             )
#                             media_parts.append(part)

#                     # --- IMAGE HANDLING ---
#                     else:
#                         status.write(f"🖼️ Reading image: {Path(file_path).name}")
#                         if not mime_type:
#                             mime_type = "application/octet-stream"

#                         with open(file_path, 'rb') as f:
#                             file_bytes = f.read()

#                         part = types.Part.from_bytes(
#                             data=file_bytes,
#                             mime_type=mime_type
#                         )
#                         media_parts.append(part)
                
#                 # Combine prompt with images
#                 contents = [input_prompt] + media_parts
                
#                 status.write(f"🚀 Sending request to Gemini ({len(media_parts)} parts)...")
                
#                 # Generate Content
#                 if generation_config:
#                     response = client.models.generate_content(
#                         model=model_id,
#                         contents=contents,
#                         config=generation_config,
#                     )
#                 else:
#                     response = client.models.generate_content(
#                         model=model_id,
#                         contents=contents,
#                     )
                
#                 status.update(label="Analysis Complete!", state="complete", expanded=False)
                
#                 # Extract Metadata
#                 usage_meta = response.usage_metadata
#                 usage_info = {
#                     "prompt_tokens": usage_meta.prompt_token_count if usage_meta else 0,
#                     "output_tokens": usage_meta.candidates_token_count if usage_meta else 0,
#                     "total_tokens": usage_meta.total_token_count if usage_meta else 0,
#                     "total_cost": 0.0 
#                 }

#                 return response.text, ground_truth, usage_info
                
#             except Exception as e:
#                 status.update(label="Error occurred", state="error")
#                 raise e

@observe(as_type="evaluator", name="evaluate-prediction")
def evaluate_with_gemini(prediction, ground_truth, session_id="streamlit_session"):
    """Evaluate prediction against ground truth using Gemini."""
    
    # Remove response_mime_type for preview models
    eval_generation_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        max_output_tokens=20000,  # Increased for detailed response
        system_instruction="""You are an expert evaluator for document analysis. 
Be thorough and fair in your assessment.
IMPORTANT: Return ONLY a valid JSON object. No markdown, no code blocks, no explanation outside JSON.""",
        # NO response_mime_type here!
    )

    eval_prompt = f"""
You are evaluating a document analysis output against the ground truth.

The evaluation must focus solely on the content and structure of the raw analysis output against the ground truth, disregarding any external annotations, comments, or corrections (e.g., 'teacher's marks').

Return a JSON object with these fields:
{{
    "score": <float between 0 and 1, where 1 is perfect match>,
    "grade": <letter grade: A, B, C, D, or F>,
    "reason": <detailed explanation of the overall score>,
    "content_accuracy": {{
        "score": <float 0-1>,
        "details": <explanation of text/content accuracy>
    }},
    "structure_accuracy": {{
        "score": <float 0-1>,
        "details": <explanation of structural accuracy>
    }},
    "completeness": {{
        "score": <float 0-1>,
        "details": <explanation of how complete the analysis is>
    }},
    "matches": [<list of correctly identified elements>],
    "misses": [<list of missed or incorrect elements>],
    "extra": [<list of elements in prediction but not in ground truth>],
    "suggestions": [<list of improvement suggestions>]
}}

CRITICAL RULES:
- Output ONLY the JSON object
- Do NOT wrap in ```json``` or any markdown
- Do NOT include any text before or after the JSON
- Start your response with {{ and end with }}

=== GROUND TRUTH ===
{ground_truth}

=== PREDICTION ===
{prediction}
"""

    eval_result_text, _, _ = call_gemini(
        input_prompt=eval_prompt,
        ground_truth=ground_truth,
        model_id="gemini-3-pro-preview",
        file_paths=None,
        generation_config=eval_generation_config,
        session_id=session_id
    )
    
    # Robust JSON parsing
    raw_output = eval_result_text
    print(f"Raw output type: {type(raw_output)}")
    print(f"Raw output: {raw_output[:500] if raw_output else 'None'}...")
    
    if raw_output is None:
        raise ValueError("Gemini returned None response")
    
    # Clean the output - handle various formats
    clean_json = raw_output.strip()
    
    # Remove markdown code blocks if present
    if clean_json.startswith("```json"):
        clean_json = clean_json[7:]
    if clean_json.startswith("```"):
        clean_json = clean_json[3:]
    if clean_json.endswith("```"):
        clean_json = clean_json[:-3]
    clean_json = clean_json.strip()
    
    # Try to find JSON object if there's extra text
    if not clean_json.startswith("{"):
        start_idx = clean_json.find("{")
        if start_idx != -1:
            clean_json = clean_json[start_idx:]
    
    if not clean_json.endswith("}"):
        end_idx = clean_json.rfind("}")
        if end_idx != -1:
            clean_json = clean_json[:end_idx + 1]

    try:
        result = json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Attempted to parse: {clean_json[:500]}...")
        raise ValueError(f"Gemini did not return valid JSON: {clean_json[:200]}...") from e

    # Ensure all expected fields exist with defaults
    result.setdefault("score", 0)
    result.setdefault("grade", "N/A")
    result.setdefault("reason", "No reason provided")
    result.setdefault("content_accuracy", {"score": 0, "details": "N/A"})
    result.setdefault("structure_accuracy", {"score": 0, "details": "N/A"})
    result.setdefault("completeness", {"score": 0, "details": "N/A"})
    result.setdefault("matches", [])
    result.setdefault("misses", [])
    result.setdefault("extra", [])
    result.setdefault("suggestions", [])

    langfuse.score_current_trace(
        name="evaluation_score",
        value=float(result["score"]),
        comment=result["reason"],
    )

    return result


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def get_file_base_name(filename):
    """Extract base name without extension from filename."""
    return Path(filename).stem


def get_score_color(score):
    """Return color based on score."""
    if score >= 0.8:
        return "#4CAF50"  # Green
    elif score >= 0.6:
        return "#8BC34A"  # Light Green
    elif score >= 0.4:
        return "#FFC107"  # Amber
    elif score >= 0.2:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def get_grade_emoji(grade):
    """Return emoji based on grade."""
    grade_emojis = {
        "A": "🏆",
        "B": "👍",
        "C": "👌",
        "D": "⚠️",
        "F": "❌"
    }
    return grade_emojis.get(grade, "📊")


def render_evaluation_results(result):
    """Render comprehensive evaluation results."""
    
    score = float(result["score"])
    grade = result.get("grade", "N/A")
    score_color = get_score_color(score)
    grade_emoji = get_grade_emoji(grade)
    
    # Custom CSS for evaluation display
    st.markdown("""
        <style>
        .eval-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .score-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            color: white;
            font-weight: bold;
        }
        .metric-bar {
            height: 8px;
            border-radius: 4px;
            background: #e0e0e0;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .metric-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 0.25rem;
        }
        .tag-success {
            background: #E8F5E9;
            color: #2E7D32;
            border: 1px solid #A5D6A7;
        }
        .tag-error {
            background: #FFEBEE;
            color: #C62828;
            border: 1px solid #EF9A9A;
        }
        .tag-warning {
            background: #FFF3E0;
            color: #E65100;
            border: 1px solid #FFCC80;
        }
        .tag-info {
            background: #E3F2FD;
            color: #1565C0;
            border: 1px solid #90CAF9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ========== HEADER SECTION ==========
    st.markdown("## Evaluation Results")
    st.markdown(f"*Evaluated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")
    
    st.divider()
    
    # ========== MAIN SCORE SECTION ==========
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Score Circle
        st.markdown(f"""
            <div style="text-align: center;">
                <div class="score-circle" style="background: linear-gradient(135deg, {score_color} 0%, {score_color}99 100%);">
                    <span style="font-size: 2.5rem;">{score:.0%}</span>
                    <span style="font-size: 0.9rem; opacity: 0.9;">Overall Score</span>
                </div>
                <div style="margin-top: 1rem;">
                    <span style="font-size: 3rem;">{grade_emoji}</span>
                    <h2 style="margin: 0; color: {score_color};">Grade: {grade}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Reason/Summary
        st.markdown("### Evaluation Summary")
        st.info(result["reason"])
        
        # Quick Stats
        st.markdown("### Quick Stats")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Matches", len(result.get("matches", [])))
        with stat_col2:
            st.metric("Misses", len(result.get("misses", [])))
        with stat_col3:
            st.metric("Extra", len(result.get("extra", [])))
    
    with col3:
        # Progress indicators for sub-scores
        st.markdown("### Sub-Scores")
        
        content_score = result.get("content_accuracy", {}).get("score", 0)
        structure_score = result.get("structure_accuracy", {}).get("score", 0)
        completeness_score = result.get("completeness", {}).get("score", 0)
        
        # Content Accuracy
        st.markdown(f"**Content Accuracy:** {content_score:.0%}")
        st.progress(content_score)
        
        # Structure Accuracy
        st.markdown(f"**Structure Accuracy:** {structure_score:.0%}")
        st.progress(structure_score)
        
        # Completeness
        st.markdown(f"**Completeness:** {completeness_score:.0%}")
        st.progress(completeness_score)
    
    st.divider()
    
    # ========== DETAILED BREAKDOWN ==========
    st.markdown("## Detailed Breakdown")
    
    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Sub-Score Details", "Matches & Misses", "Suggestions"])
    
    with detail_tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Content Accuracy")
            content_info = result.get("content_accuracy", {})
            score_val = content_info.get("score", 0)
            st.markdown(f"""
                <div class="eval-card">
                    <h1 style="color: {get_score_color(score_val)}; margin: 0;">{score_val:.0%}</h1>
                    <p style="color: #666; margin-top: 1rem;">{content_info.get('details', 'N/A')}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Structure Accuracy")
            structure_info = result.get("structure_accuracy", {})
            score_val = structure_info.get("score", 0)
            st.markdown(f"""
                <div class="eval-card">
                    <h1 style="color: {get_score_color(score_val)}; margin: 0;">{score_val:.0%}</h1>
                    <p style="color: #666; margin-top: 1rem;">{structure_info.get('details', 'N/A')}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### Completeness")
            completeness_info = result.get("completeness", {})
            score_val = completeness_info.get("score", 0)
            st.markdown(f"""
                <div class="eval-card">
                    <h1 style="color: {get_score_color(score_val)}; margin: 0;">{score_val:.0%}</h1>
                    <p style="color: #666; margin-top: 1rem;">{completeness_info.get('details', 'N/A')}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with detail_tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Correct Matches")
            matches = result.get("matches", [])
            if matches:
                for match in matches:
                    st.markdown(f'<span class="tag tag-success">✓ {match}</span>', unsafe_allow_html=True)
            else:
                st.info("No matches recorded")
        
        with col2:
            st.markdown("### Missed Elements")
            misses = result.get("misses", [])
            if misses:
                for miss in misses:
                    st.markdown(f'<span class="tag tag-error">✗ {miss}</span>', unsafe_allow_html=True)
            else:
                st.success("No misses - great job!")
        
        with col3:
            st.markdown("### Extra Elements")
            extras = result.get("extra", [])
            if extras:
                for extra in extras:
                    st.markdown(f'<span class="tag tag-warning">+ {extra}</span>', unsafe_allow_html=True)
            else:
                st.info("No extra elements detected")
    
    with detail_tab3:
        st.markdown("### Improvement Suggestions")
        suggestions = result.get("suggestions", [])
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"""
                    <div style="padding: 1rem; background: #F5F5F5; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #2196F3;">
                        <strong>Suggestion {i}:</strong> {suggestion}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No specific suggestions - the analysis looks good!")
    
    st.divider()
    
    # ========== SIDE BY SIDE COMPARISON ==========
    st.markdown("## Side-by-Side Comparison")
    
    compare_col1, compare_col2 = st.columns(2)
    
    with compare_col1:
        st.markdown("### Ground Truth")
        with st.container(height=400):
            st.markdown(st.session_state.ground_truth)
    
    with compare_col2:
        st.markdown("### Prediction")
        with st.container(height=400):
            st.markdown(st.session_state.prediction)
    
    st.divider()
    
    # ========== EXPORT SECTION ==========
    st.markdown("## Export Results")
    
    # Generate report
    report = f"""# Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Score: {score:.1%} (Grade: {grade})

### Summary
{result['reason']}

---

## Sub-Scores

| Metric | Score | Details |
|--------|-------|---------|
| Content Accuracy | {result.get('content_accuracy', {}).get('score', 0):.0%} | {result.get('content_accuracy', {}).get('details', 'N/A')} |
| Structure Accuracy | {result.get('structure_accuracy', {}).get('score', 0):.0%} | {result.get('structure_accuracy', {}).get('details', 'N/A')} |
| Completeness | {result.get('completeness', {}).get('score', 0):.0%} | {result.get('completeness', {}).get('details', 'N/A')} |

---

## Matches ({len(result.get('matches', []))})
{chr(10).join(['- ' + m for m in result.get('matches', [])]) or 'None'}

## Misses ({len(result.get('misses', []))})
{chr(10).join(['- ' + m for m in result.get('misses', [])]) or 'None'}

## Extra Elements ({len(result.get('extra', []))})
{chr(10).join(['- ' + m for m in result.get('extra', [])]) or 'None'}

---

## Suggestions
{chr(10).join(['- ' + s for s in result.get('suggestions', [])]) or 'None'}

---

## Ground Truth
{st.session_state.ground_truth}

## Prediction
{st.session_state.prediction}

"""
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        st.download_button(
            label="Download Report (Markdown)",
            data=report,
            file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with export_col2:
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with export_col3:
        # Create CSV-like summary
        csv_data = f"""Metric,Score,Details
Overall,{score},{result['reason'].replace(',', ';')}
Content Accuracy,{result.get('content_accuracy', {}).get('score', 0)},{result.get('content_accuracy', {}).get('details', 'N/A').replace(',', ';')}
Structure Accuracy,{result.get('structure_accuracy', {}).get('score', 0)},{result.get('structure_accuracy', {}).get('details', 'N/A').replace(',', ';')}
Completeness,{result.get('completeness', {}).get('score', 0)},{result.get('completeness', {}).get('details', 'N/A').replace(',', ';')}
"""
        st.download_button(
            label="Download Summary (CSV)",
            data=csv_data,
            file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #424242;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .file-info-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_id = st.selectbox(
            "Select Model",
            ["gemini-3-pro-preview", "gemini-2.0-flash", "gemini-2.5-flash"],
            index=0,
        )
        
        st.divider()
        
        st.subheader("Generation Settings")
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
        max_tokens = st.number_input("Max Output Tokens", 256, 65536, 65536, 256)
        
        enable_thinking = st.checkbox("Enable Thinking Mode", True)
        thinking_budget = 0
        if enable_thinking:
            thinking_budget = st.slider("Thinking Budget", 20000, 20000, -1, 1024)
        
        st.divider()
        
        st.subheader("Prompt Template")
        prompt_template = st.selectbox("Select Template", list(PROMPTS.keys()), 0)

    # Initialize session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "usage_info" not in st.session_state:
        st.session_state.usage_info = None
    if "ground_truth" not in st.session_state:
        st.session_state.ground_truth = None
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "prediction_source" not in st.session_state:
        st.session_state.prediction_source = "generated"
    # New session states for ground truth preparation (Tab 3)
    if "gt_prep_source_filename" not in st.session_state:
        st.session_state.gt_prep_source_filename = None
    if "gt_prep_content" not in st.session_state:
        st.session_state.gt_prep_content = ""
    if "gt_last_prediction_file" not in st.session_state:
        st.session_state.gt_last_prediction_file = None
    # Create a consistent session ID for all traces in this user session
    if "langfuse_session_id" not in st.session_state:
        st.session_state.langfuse_session_id = f"streamlit_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_id = st.session_state.langfuse_session_id

    # Main content tabs - Added new "Prepare Ground Truth" tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Upload & Analyze", 
        "Results", 
        "Prepare Ground Truth",
        "Evaluate", 
        "Evaluation Results"
    ])

    # Tab 1: Upload and Analyze
    file_name_x = "Unknown"
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<p class="section-header">Upload Document</p>', unsafe_allow_html=True)
            
            uploaded_doc = st.file_uploader(
                "Upload PDF or Image",
                type=["pdf", "png", "jpg", "jpeg", "webp", "gif"],
                key="doc_uploader"
            )

            print(uploaded_doc)

            if uploaded_doc:
                file_name_x = uploaded_doc.name
                st.success(f"Uploaded: {uploaded_doc.name}")
                if uploaded_doc.type.startswith("image/"):
                    st.image(uploaded_doc, caption="Document Preview", use_container_width=True)
                else:
                    st.info(f"📎 PDF file: {uploaded_doc.name} ({uploaded_doc.size / 1024:.1f} KB)")
        
        with col2:
            st.markdown('<p class="section-header">Prompt Configuration</p>', unsafe_allow_html=True)
            
            if prompt_template == "Custom":
                instruction_prompt = st.text_area("Custom Prompt", height=300, placeholder="Enter your custom prompt...")
            else:
                instruction_prompt = st.text_area("Prompt (editable)", value=PROMPTS[prompt_template], height=300)
        
        st.divider()
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_button = st.button(
                "Analyze Document",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_doc or not instruction_prompt
            )
        
        if analyze_button and uploaded_doc and instruction_prompt:
            with st.spinner("Processing document with Gemini..."):
                try:
                    temp_path = save_uploaded_file(uploaded_doc)
                    print("max_tokens:", max_tokens, "temperature:", temperature, "model:", model_id)
                    gen_config_params = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "system_instruction": "You are an expert Document Analyzer.",
                    }
                    
                    if True:
                        gen_config_params["thinking_config"] = types.ThinkingConfig(
                            thinking_level="high",
                            include_thoughts=True,
                        )
                    
                    generation_config = types.GenerateContentConfig(**gen_config_params)
                    
                    prediction, _, usage_info = call_gemini(
                        input_prompt=instruction_prompt,
                        ground_truth=None,
                        model_id=model_id,
                        file_paths=temp_path,
                        generation_config=generation_config,
                        session_id=session_id
                    )
                    
                    st.session_state.prediction = prediction
                    st.session_state.usage_info = usage_info
                    st.session_state.prediction_source = "generated"
                    st.session_state.evaluation_result = None
                    
                    os.unlink(temp_path)
                    langfuse.flush()
                    
                    st.success("Analysis complete! Check the Results tab.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    # Tab 2: Results
    with tab2:
        if st.session_state.prediction:
            st.markdown('<p class="section-header">Analysis Results</p>', unsafe_allow_html=True)
            
            # Show source indicator
            if st.session_state.prediction_source == "generated":
                st.info("Showing generated prediction from document analysis")
            else:
                st.info("Showing uploaded prediction file")
            
            if st.session_state.usage_info:
                usage = st.session_state.usage_info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Input Tokens", f"{usage['prompt_tokens']:,}")
                with col2:
                    st.metric("Output Tokens", f"{usage['output_tokens']:,}")
                with col3:
                    st.metric("Total Tokens", f"{usage['total_tokens']:,}")
                with col4:
                    st.metric("Est. Cost", f"${usage['total_cost']:.6f}")
            
            st.divider()
            
            display_mode = st.radio("Display Mode", ["Rendered Markdown", "Raw Text"], horizontal=True)
            
            if display_mode == "Rendered Markdown":
                st.markdown(st.session_state.prediction)
            else:
                st.code(st.session_state.prediction, language="markdown")
            
            st.divider()
            
            # Download buttons
            st.markdown("### Download Results")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="Download as Markdown",
                    data=st.session_state.prediction,
                    file_name=f"analysis_result_{file_name_x}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="Download as Text",
                    data=st.session_state.prediction,
                    file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("No results yet. Upload a document and analyze it first.")

    # Tab 3: Prepare Ground Truth (NEW TAB)
    # Tab 3: Prepare Ground Truth (NEW TAB)
    with tab3:
        st.markdown('<p class="section-header">Prepare Ground Truth</p>', unsafe_allow_html=True)
        st.markdown("Upload your source document and prediction file, edit the prediction to create ground truth, then download.")
        
        st.divider()
        
        col_left, col_right = st.columns([1, 1])
        
        # Left Column: Source Document (PDF/Image)
        with col_left:
            st.markdown("### Source Document")
            st.markdown("Upload the original PDF or image document for reference.")
            
            source_doc = st.file_uploader(
                "Upload PDF or Image",
                type=["pdf", "png", "jpg", "jpeg", "webp", "gif"],
                key="gt_source_doc"
            )
            
            if source_doc:
                st.session_state.gt_prep_source_filename = get_file_base_name(source_doc.name)
                
                st.markdown("#### Preview")
                if source_doc.type.startswith("image/"):
                    st.image(source_doc, caption="Document Preview", use_container_width=True)
                else:
                    try:
                        pdf_bytes = source_doc.getvalue()
                        images = convert_from_bytes(pdf_bytes)
                        
                        if len(images) > 1:
                            page_numbers = list(range(1, len(images) + 1))
                            page_tabs = st.tabs([f"Page {i}" for i in page_numbers])
                            
                            for tab_idx, tab in enumerate(page_tabs):
                                with tab:
                                    st.image(images[tab_idx], caption=f"Page {tab_idx + 1}", use_container_width=True)
                        else:
                            st.image(images[0], caption="PDF Preview", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not preview PDF: {str(e)}")
                        st.info("📎 PDF document uploaded but preview failed. Please check if poppler is installed.")
            else:
                st.info("Upload a source document to begin")
                st.session_state.gt_prep_source_filename = None
        
        # Right Column: Prediction File & Editor
        with col_right:
            st.markdown("### Edit Prediction → Ground Truth")
            st.markdown("Upload a prediction markdown file, edit it, and download as ground truth.")
            
            prediction_file = st.file_uploader(
                "Upload Prediction (Markdown or Text)",
                type=["md", "txt", "text"],
                key="gt_prediction_file"
            )
            
            # Handle file upload WITHOUT st.rerun() - let Streamlit handle naturally
            if prediction_file is not None:
                new_file_name = prediction_file.name
                # Only update content if it's a genuinely new file
                if st.session_state.get("gt_last_prediction_file") != new_file_name:
                    prediction_content = prediction_file.read().decode("utf-8")
                    # Update the editor content directly via the key
                    st.session_state.gt_editor_widget = prediction_content
                    st.session_state.gt_last_prediction_file = new_file_name
                    # NO st.rerun() here!
            else:
                # File was removed - only clear if we actually had a file before
                if st.session_state.get("gt_last_prediction_file") is not None:
                    st.session_state.gt_editor_widget = ""
                    st.session_state.gt_last_prediction_file = None
                    # NO st.rerun() here!
            
            st.markdown("#### Edit Content")
            st.write("")
            
            # Initialize the widget state if it doesn't exist
            if "gt_editor_widget" not in st.session_state:
                st.session_state.gt_editor_widget = ""
            
            # Text area WITH key - Streamlit manages the state automatically
            # The value is stored in st.session_state.gt_editor_widget
            st.text_area(
                "Edit the prediction to create ground truth:",
                height=400,
                placeholder="Upload a prediction file above or paste content here to edit...",
                key="gt_editor_widget"  # This is the key - value auto-syncs with session_state
            )
            
            # For compatibility with download section, sync to the old variable name
            st.session_state.gt_prep_content = st.session_state.gt_editor_widget
        
        st.divider()
        
        # Download Section (rest remains the same)
        st.markdown("### Download Ground Truth")
        
        # Determine filename
        if st.session_state.gt_prep_source_filename:
            ground_truth_filename = f"{st.session_state.gt_prep_source_filename}_ground_truth.md"
        else:
            ground_truth_filename = f"ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        col_info, col_download = st.columns([2, 1])
        
        with col_info:
            if st.session_state.gt_prep_source_filename:
                st.success(f"Output filename: **{ground_truth_filename}**")
            else:
                st.warning("Upload a source document (left panel) to auto-generate filename with matching base name.")
                st.info(f"Default filename: **{ground_truth_filename}**")
        
        with col_download:
            content_to_download = st.session_state.get("gt_editor_widget", "")
            download_disabled = not content_to_download.strip()
            
            st.download_button(
                label="Download Ground Truth",
                data=content_to_download,
                file_name=ground_truth_filename,
                mime="text/markdown",
                use_container_width=True,
                disabled=download_disabled,
                type="primary"
            )
        
        if download_disabled:
            st.warning("⚠️ Please add some content to the editor before downloading.")
        
        # Quick stats
        st.divider()
        st.markdown("### 📊 Content Stats")
        
        content = st.session_state.get("gt_editor_widget", "")
        if content:
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Characters", f"{len(content):,}")
            with stat_col2:
                st.metric("Words", f"{len(content.split()):,}")
            with stat_col3:
                st.metric("Lines", f"{len(content.splitlines()):,}")
            with stat_col4:
                st.metric("Size", f"{len(content.encode('utf-8')) / 1024:.2f} KB")
        else:
            st.info("No content to analyze yet.")

    # Tab 4: Evaluate
    with tab4:
        st.markdown('<p class="section-header">Evaluation Setup</p>', unsafe_allow_html=True)
        st.markdown("Upload both **prediction** and **ground truth** files for evaluation, or use the generated prediction from analysis.")
        
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        
        # Left column: Prediction
        with col1:
            st.markdown("### 🤖 Prediction")
            
            prediction_file = st.file_uploader(
                "Upload Prediction (Markdown or Text)",
                type=["md", "txt", "text"],
                key="prediction_upload"
            )
            
            if prediction_file:
                prediction_content = prediction_file.read().decode("utf-8")
                st.session_state.prediction = prediction_content
                st.success(f"{prediction_file.name}")
            
            # Preview prediction
            if st.session_state.prediction:
                with st.expander("📄 Preview Prediction", expanded=False):
                    st.markdown(st.session_state.prediction[:2000] + ("..." if len(st.session_state.prediction) > 2000 else ""))
        
        # Right column: Ground Truth
        with col2:
            st.markdown("### 🎯 Ground Truth")
            
            ground_truth_file = st.file_uploader(
                "Upload Ground Truth (Markdown or Text)",
                type=["md", "txt", "text"],
                key="ground_truth_upload"
            )
            
            if ground_truth_file:
                ground_truth_content = ground_truth_file.read().decode("utf-8")
                st.session_state.ground_truth = ground_truth_content
                st.success(f"{ground_truth_file.name}")
            
            # Preview ground truth
            if st.session_state.ground_truth:
                with st.expander("📄 Preview Ground Truth", expanded=False):
                    st.markdown(st.session_state.ground_truth[:2000] + ("..." if len(st.session_state.ground_truth) > 2000 else ""))
        
        st.divider()
        
        # Check for duplicate file uploads
        duplicate_files = False
        if prediction_file and ground_truth_file:
            if prediction_file.name == ground_truth_file.name:
                st.error("❌ Error: Cannot upload the same file for both prediction and ground truth. Please upload different files.")
                duplicate_files = True
        
        # Status summary
        st.markdown("### 📋 Evaluation Checklist")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            if st.session_state.prediction:
                source_label = "generated" if st.session_state.prediction_source == "generated" else "uploaded"
                st.success(f"✅ Prediction ready ({source_label})")
            else:
                st.error("❌ Prediction not available")
        
        with status_col2:
            if st.session_state.ground_truth:
                st.success("✅ Ground truth ready")
            else:
                st.error("❌ Ground truth not available")
        
        st.divider()
        
        # Evaluation button
        can_evaluate = st.session_state.prediction and st.session_state.ground_truth and not duplicate_files
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            evaluate_button = st.button(
                "🎯 Run Evaluation",
                type="primary",
                use_container_width=True,
                disabled=not can_evaluate
            )
        
        if not can_evaluate:
            missing = []
            if not st.session_state.prediction:
                missing.append("prediction")
            if not st.session_state.ground_truth:
                missing.append("ground truth")
            if duplicate_files:
                missing.append("different files for prediction and ground truth")
            st.warning(f"⚠️ Please provide: {', '.join(missing)}")
        
        if evaluate_button and can_evaluate:
            with st.spinner("🔄 Evaluating with Gemini..."):
                try:
                    eval_result = evaluate_with_gemini(
                        prediction=st.session_state.prediction,
                        ground_truth=st.session_state.ground_truth,
                        session_id=session_id
                    )
                    
                    st.session_state.evaluation_result = eval_result
                    langfuse.flush()
                    
                    st.success("✅ Evaluation complete! Check the 'Evaluation Results' tab.")
                    
                except Exception as e:
                    langfuse.flush()
                    st.error(f"❌ Evaluation error: {str(e)}")

    # Tab 5: Evaluation Results
    with tab5:
        if st.session_state.evaluation_result:
            render_evaluation_results(st.session_state.evaluation_result)
        else:
            st.info("📌 No evaluation results yet. Run an evaluation first from the 'Evaluate' tab.")
            
            # Show what's needed
            st.markdown("### 📋 Checklist")
            
            check1 = "✅" if st.session_state.prediction else "⬜"
            check2 = "✅" if st.session_state.ground_truth else "⬜"
            check3 = "✅" if st.session_state.evaluation_result else "⬜"
            
            st.markdown(f"""
            {check1} **Step 1:** Get prediction (Analyze document OR upload prediction file)
            
            {check2} **Step 2:** Upload ground truth (Evaluate tab)
            
            {check3} **Step 3:** Run evaluation (Evaluate tab)
            """)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            Built with Streamlit • Powered by Google Gemini • Traced with Langfuse
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
    