## Document Annotation Instructions

The goal of this task is to identify all meaningful blocks of content in a document, extract the transcribed text, assign the correct element type, and define the structural and semantic relationships between them.

### I. Output Format

All predictions **MUST** be outputted in a JSON array named `document_elements`. Each element in the array must be an object with the following required keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| **`id`** | `string` | A unique identifier (e.g., `B1`, `N_Start`, `T2`). |
| **`text`** | `string` | The exact transcribed content of the block. |
| **`type`** | `enum` | The element's function (see Section II). |
| **`bbox`** | `array` | Normalized coordinates `[xmin, ymin, xmax, ymax]`. All values **MUST** be integers between 0 and 100. |
| **`relations`** | `array` | A list of semantic connections (see Section III). |

### II. Element Type Definitions (`type`)

Use **only** the following seven element types:

| Element Type | Description |
| :--- | :--- |
| **`TITLE`** | The main heading of the document or a major section heading. |
| **`PARAGRAPH`** | A standard block of prose, including single sentences that stand alone. |
| **`LIST`** | A collection of items (bullet points or numbered list). Each list item should be its own block. |
| **`TABLE_CELL`** | A distinct cell within a formal grid/table structure. |
| **`KEY_VALUE_PAIR`** | Used for form-like structures, labels, or fields where there is a clear key (label) and its associated value. |
| **`DIAGRAM_NODE`** | A shape or text label representing a component or step within a visual diagram/flowchart. |
| **`DIAGRAM_ARROW`** | A line, arrow, or connector indicating a link or flow direction within a diagram/flowchart. |

### III. Relationship Definitions (`relations`)

The `relations` array contains objects with two keys: `target_id` and `relation_type`.

| Relation Type | Use Case |
| :--- | :--- |
| **`FLOWS_TO`** | Connects elements in a sequential order (e.g., `PARAGRAPH` to the next `PARAGRAPH`, or `DIAGRAM_NODE` to the next `DIAGRAM_NODE` via a `DIAGRAM_ARROW`). |
| **`IS_LABEL_FOR`** | Connects a caption, legend, or external label to the main element it describes (e.g., a caption block to a `TABLE_CELL` or a `DIAGRAM_NODE`). |
| **`VALUE_FOR`** | Links the "value" part of a form or key-value structure to its corresponding "key" (label). |
