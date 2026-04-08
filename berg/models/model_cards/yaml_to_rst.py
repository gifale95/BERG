#!/usr/bin/env python3
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Union

import yaml


def yaml_to_rst(yaml_file: str, output_file: Optional[str] = None) -> str:
    """
    Convert a YAML file in the specified format to an RST file for ReadTheDocs.

    Args:
        yaml_file: Path to the input YAML file
        output_file: Path to the output RST file. If None, the RST content is returned as a string.

    Returns:
        If output_file is None, returns the RST content as a string.
        Otherwise, writes to the output file and returns None.
    """
    # Read the YAML file
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Generate the RST content
    rst_content = []

    # Generate the title
    model_id = data.get("model_id", os.path.basename(yaml_file).replace(".yaml", ""))
    title_line = "=" * len(model_id)
    rst_content.extend([title_line, model_id, title_line, ""])

    # Model Summary section
    rst_content.extend(["Model Summary", "------------", ""])
    rst_content.append(".. list-table::")
    rst_content.append("   :widths: 30 70")
    rst_content.append("   :stub-columns: 1")
    rst_content.append("")

    # Add model summary items
    summary_items = [
        ("Modality", data.get("modality", "")),
        ("Training Dataset", data.get("training_dataset", "")),
    ]
    if "species" in data:
        summary_items.append(("Species", data.get("species", "")))
    if "stimuli" in data:
        summary_items.append(("Stimuli", data.get("stimuli", "")))
    if "model_type" in data:
        summary_items.append(("Model Type", data.get("model_type", "")))
    elif "model_architecture" in data:  # Fallback to old field name
        summary_items.append(("Model Architecture", data.get("model_architecture", "")))

    summary_items.append(("Creator", data.get("creator", "")))

    for item, value in summary_items:
        rst_content.append(f"   * - {item}")
        rst_content.append(f"     - {value}")

    rst_content.append("")

    # Description section
    rst_content.extend(["Description", "----------", ""])
    description = data.get("description", "").strip()
    # Process multiline description
    for line in description.split("\n"):
        rst_content.append(line)
    rst_content.append("")

    # Metadata section
    if "metadata" in data:
        rst_content.extend(["Metadata", "--------", ""])
        metadata = data.get("metadata", "").strip()

        # Parse metadata into a hierarchical structure
        lines = metadata.split("\n")
        entries = []  # List of (key, shape, desc, nesting_level)
        note_lines = []  # Collect NOTE lines to display above table

        i = 0
        indent_stack = []  # Track indent levels to determine nesting

        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()

            if not line_stripped:
                i += 1
                continue

            indent_level = len(line) - len(line.lstrip())

            # Check if this is a NOTE entry (special case)
            if line_stripped.upper().startswith("NOTE"):
                # This is a note - extract and save for later
                if ":" in line_stripped:
                    note_text = line_stripped.split(":", 1)[1].strip()
                    note_lines.append(note_text)
                i += 1
                continue

            # Check if this is a section header (quoted string ending with :)
            if line_stripped.startswith("'") and line_stripped.endswith("':"):
                # Section headers are treated as level 0 entries with type "section"
                section_name = line_stripped.rstrip(":").strip("'")
                entries.append((section_name, "section", "", 0))
                indent_stack = [(indent_level, 0)]  # Reset stack for new section
                i += 1
                continue

            # Parse entry: key : shape - description or key : shape
            if ":" in line_stripped:
                # Determine nesting level based on indent
                # Clear stack of items with equal or greater indent
                while indent_stack and indent_stack[-1][0] >= indent_level:
                    indent_stack.pop()

                # Current nesting level is the stack depth
                nesting_level = len(indent_stack)

                # Parse the entry
                if " - " not in line_stripped:
                    # Format: key : shape (no description)
                    # But first check if this is actually a valid key or just text with a colon
                    key_part, shape_part = line_stripped.split(":", 1)
                    key = key_part.strip().strip("'")
                    shape = shape_part.strip()

                    # Validate that this looks like a proper key:value entry
                    # Keys should be simple identifiers, not sentences
                    # Skip if key contains spaces and looks like a phrase or sentence
                    if " " in key and (
                        len(key)
                        > 30  # Long keys with spaces are likely descriptive text
                        or key.endswith(
                            ("ROIs", "info", "data", "values", "items")
                        )  # Descriptive headers
                    ):
                        # This is likely descriptive text, not a key - skip it
                        i += 1
                        continue

                    desc = ""
                    i += 1
                else:
                    # Format: key : shape - description
                    key_part, rest = line_stripped.split(":", 1)
                    key = key_part.strip().strip("'")

                    shape_part, desc_part = rest.split(" - ", 1)
                    shape = shape_part.strip()
                    desc = desc_part.strip()

                    # Check for continuation lines (multi-line descriptions)
                    continuation_lines = [desc]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        next_line_stripped = next_line.strip()

                        if not next_line_stripped:
                            j += 1
                            continue

                        next_indent = len(next_line) - len(next_line.lstrip())

                        # Check if this is a continuation line
                        # It's a continuation if indent is higher AND it doesn't match key:shape format
                        is_continuation = next_indent > indent_level

                        if is_continuation and ":" in next_line_stripped:
                            # Only break if this looks like a proper "key : shape - desc" entry
                            # Check for the pattern: word/phrase : something
                            if " - " in next_line_stripped:
                                # Has the full pattern, likely a new entry
                                parts = next_line_stripped.split(":", 1)
                                if len(parts) == 2:
                                    potential_key = parts[0].strip()
                                    # Simple key without spaces, or known valid keys
                                    if (
                                        not " " in potential_key
                                        or potential_key.replace("_", "")
                                        .replace("-", "")
                                        .isalnum()
                                    ):
                                        is_continuation = False

                        if is_continuation:
                            continuation_lines.append(next_line_stripped)
                            j += 1
                        else:
                            break

                    # Join continuation lines with newlines
                    if len(continuation_lines) > 1:
                        desc = "\n".join(continuation_lines)
                        i = j
                    else:
                        i += 1

                entries.append((key, shape, desc, nesting_level))

                # If this is a dict, add to stack for tracking children
                if shape.lower() == "dict":
                    indent_stack.append((indent_level, nesting_level))
            else:
                i += 1

        # Display NOTE if present
        if note_lines:
            rst_content.append(".. note::")
            rst_content.append("")
            for note_line in note_lines:
                rst_content.append(f"   {note_line}")
            rst_content.append("")

        # Generate definition lists with proper hierarchy
        if entries:
            prev_level = -1
            for key, shape, desc, nesting_level in entries:
                # Handle section headers specially - use bold header
                if shape == "section":
                    rst_content.append(f"**{key}**")
                    rst_content.append("")
                    prev_level = -1
                    continue

                # Handle subsection headers (no shape, only description)
                # These are like "full" or "repeat{N}" - they describe a group
                if not shape and desc:
                    indent = "    " * nesting_level
                    rst_content.append("")
                    rst_content.append(f"{indent}**{key}**: *{desc}*")
                    rst_content.append("")
                    prev_level = nesting_level
                    continue

                # Add blank line when returning to same or lower nesting level
                # This separates definition list items properly
                if prev_level >= 0 and nesting_level <= prev_level:
                    rst_content.append("")

                # Calculate indentation (4 spaces per level)
                indent = "    " * nesting_level

                # Format key with bold
                key_formatted = f"**{key}**"

                # Format the definition line: key : shape - description (all on one line)
                if desc:
                    # Single line format with description
                    if "\n" in desc:
                        # Multi-line description - first line inline, rest indented below
                        desc_lines = desc.split("\n")
                        rst_content.append(
                            f"{indent}{key_formatted} : ``{shape}`` - {desc_lines[0]}"
                        )
                        # Add remaining lines indented one level deeper
                        desc_indent = "    " * (nesting_level + 1)
                        for desc_line in desc_lines[1:]:
                            rst_content.append(f"{desc_indent}{desc_line}")
                    else:
                        # Single line description
                        rst_content.append(
                            f"{indent}{key_formatted} : ``{shape}`` - {desc}"
                        )
                else:
                    # No description, just key and shape (don't show if shape is empty)
                    if shape:
                        rst_content.append(f"{indent}{key_formatted} : ``{shape}``")

                prev_level = nesting_level

        # Remove trailing blank lines
        while rst_content and rst_content[-1] == "":
            rst_content.pop()

        rst_content.append("")

    # Input section
    rst_content.extend(["Input", "-----", ""])
    input_data = data.get("input", {})

    rst_content.append(".. list-table::")
    rst_content.append("   :widths: 20 80")
    rst_content.append("   :stub-columns: 1")
    rst_content.append("")

    # Type
    if input_data.get("type"):
        rst_content.append("   * - Type")
        rst_content.append(f"     - ``{input_data.get('type', '')}``")

    # Shape (if exists)
    if input_data.get("shape"):
        rst_content.append("   * - Shape")
        rst_content.append(f"     - ``{input_data.get('shape', '')}``")

    # Description
    if input_data.get("description"):
        input_desc = input_data.get("description", "").strip()
        rst_content.append("   * - Description")
        if "\n" in input_desc:
            desc_lines = input_desc.split("\n")
            rst_content.append(f"     - | {desc_lines[0]}")
            for line in desc_lines[1:]:
                rst_content.append(f"       | {line.rstrip()}")
        else:
            rst_content.append(f"     - {input_desc}")

    # Constraints (if exists)
    if input_data.get("constraints"):
        rst_content.append("   * - Constraints")
        constraints = input_data.get("constraints", [])
        if constraints:
            rst_content.append(f"     - * {constraints[0]}")
            for constraint in constraints[1:]:
                rst_content.append(f"       * {constraint}")

    # Example (if exists)
    if input_data.get("example"):
        input_example = input_data.get("example", "")
        # Convert non-string examples (e.g. lists) to their string representation
        if not isinstance(input_example, str):
            input_example = str(input_example)
        else:
            input_example = input_example.strip()
        rst_content.append("   * - Example")
        if "\n" in input_example:
            # Just show as plain text lines in the table
            example_lines = input_example.split("\n")
            rst_content.append(f"     - {example_lines[0]}")
            for line in example_lines[1:]:
                line = line.strip()
                if line:
                    rst_content.append(f"       {line}")
        else:
            rst_content.append(f"     - ``{input_example}``")

    rst_content.append("")

    # Output section
    rst_content.extend(["Output", "------", ""])
    output_data = data.get("output", {})

    rst_content.append(".. list-table::")
    rst_content.append("   :widths: 20 80")
    rst_content.append("   :stub-columns: 1")
    rst_content.append("")

    # Type
    if output_data.get("type"):
        rst_content.append("   * - Type")
        rst_content.append(f"     - ``{output_data.get('type', '')}``")

    # Shape
    if output_data.get("shape"):
        rst_content.append("   * - Shape")
        rst_content.append(f"     - ``{output_data.get('shape', '')}``")

    # Description
    if output_data.get("description"):
        output_description = output_data.get("description", "").strip()
        rst_content.append("   * - Description")
        if "\n" in output_description:
            desc_lines = output_description.split("\n")
            rst_content.append(f"     - | {desc_lines[0]}")
            for line in desc_lines[1:]:
                rst_content.append(f"       | {line.rstrip()}")
        else:
            rst_content.append(f"     - {output_description}")

    rst_content.append("")

    # Output dimensions
    if "dimensions" in output_data:
        rst_content.append("**Dimensions:**")
        rst_content.append("")
        rst_content.append(".. list-table::")
        rst_content.append("   :widths: 30 70")
        rst_content.append("   :header-rows: 1")
        rst_content.append("")
        rst_content.append("   * - Name")
        rst_content.append("     - Description")

        for dim in output_data["dimensions"]:
            rst_content.append(f"   * - {dim.get('name', '')}")
            # Handle multiline dimension descriptions properly
            dim_desc = dim.get("description", "").strip()
            if "\n" in dim_desc:
                # For multiline descriptions, use pipe notation
                desc_lines = dim_desc.split("\n")
                rst_content.append(f"     - | {desc_lines[0]}")
                for line in desc_lines[1:]:
                    line = line.strip()
                    if line:
                        rst_content.append(f"       | {line}")
            else:
                rst_content.append(f"     - {dim_desc}")

        rst_content.append("")

    # Parameters section
    rst_content.extend(["Parameters", "---------", ""])

    # Group parameters by function (handling comma-separated functions)
    param_by_function = {}
    for param_name, param_data in data.get("parameters", {}).items():
        functions = param_data.get("function", "")
        # Handle comma-separated functions
        if isinstance(functions, str):
            function_list = [f.strip() for f in functions.split(",")]
        else:
            function_list = [functions]

        for function in function_list:
            if function not in param_by_function:
                param_by_function[function] = []
            param_by_function[function].append((param_name, param_data))

    # Define function order
    function_order = ["get_encoding_model", "encode", "get_model_metadata"]

    # Create subsections for each function in order
    for function in function_order:
        if function not in param_by_function:
            continue

        params = param_by_function[function]

        if function == "get_encoding_model":
            display_name = "get_encoding_model"
            function_description = "This function loads the encoding model."
        elif function == "encode":
            display_name = "encode"
            function_description = "This function generates in silico neural responses using the encoding model previously loaded."
        elif function == "get_model_metadata":
            display_name = "get_model_metadata"
            function_description = "This function loads the encoding model's metadata without having to load the model itself."
        else:
            display_name = function
            function_description = ""

        rst_content.append(f"Parameters used in ``{display_name}``")
        rst_content.append("~" * (len(f"Parameters used in ``{display_name}``")))
        rst_content.append("")

        if function_description:
            rst_content.append(f"{function_description}")
            rst_content.append("")

        rst_content.append(".. list-table::")
        rst_content.append("   :widths: 20 80")
        rst_content.append("   :header-rows: 0")
        rst_content.append("")

        for param_name, param_data in params:
            # Special handling for selection parameter
            if param_name == "selection" and "properties" in param_data:
                rst_content.append(f"   * - **{param_name}**")
                rst_content.append(f"     - | **Type:** {param_data.get('type', '')}")

                required = param_data.get("required", False)
                rst_content.append(
                    f"       | **Required:** {'Yes' if required else 'No'}"
                )

                if "description" in param_data:
                    desc = param_data.get("description", "").strip()
                    desc_lines = desc.split("\n")
                    rst_content.append(f"       | **Description:** {desc_lines[0]}")
                    for line in desc_lines[1:]:
                        rst_content.append(f"       | {line}")

                rst_content.append("       | ")
                rst_content.append("       | **Properties:**")

                for prop_name, prop_data in param_data["properties"].items():
                    rst_content.append("       | ")
                    rst_content.append(f"       | **{prop_name}**")
                    rst_content.append(
                        f"       |     **Type:** {prop_data.get('type', '')}"
                    )

                    if "description" in prop_data:
                        prop_desc = prop_data.get("description", "").strip()
                        prop_desc_lines = prop_desc.split("\n")
                        rst_content.append(
                            f"       |     **Description:** {prop_desc_lines[0]}"
                        )
                        for line in prop_desc_lines[1:]:
                            rst_content.append(f"       |     {line}")

                    if "valid_values" in prop_data:
                        valid_values = prop_data["valid_values"]
                        if isinstance(valid_values, list):
                            formatted_values = []
                            for v in valid_values:
                                if isinstance(v, str):
                                    formatted_values.append(f'"{v}"')
                                else:
                                    formatted_values.append(str(v))
                            rst_content.append(
                                f"       |     **Valid values:** {', '.join(formatted_values)}"
                            )
                        else:
                            rst_content.append(
                                f"       |     **Valid values:** {valid_values}"
                            )

                    if "example" in prop_data:
                        example = prop_data["example"]
                        if isinstance(example, list):
                            if len(example) > 10:
                                example_str = str(example[:5])[:-1] + ", ... ]"
                            else:
                                example_str = str(example)
                            rst_content.append(
                                f"       |     **Example:** {example_str}"
                            )
                        else:
                            rst_content.append(f"       |     **Example:** {example}")
            else:
                # Regular parameter handling
                rst_content.append(f"   * - **{param_name}**")
                rst_content.append(f"     - | **Type:** {param_data.get('type', '')}")

                required = param_data.get("required", False)
                rst_content.append(
                    f"       | **Required:** {'Yes' if required else 'No'}"
                )

                if "description" in param_data:
                    desc = param_data.get("description", "").strip()
                    desc_lines = desc.split("\n")
                    rst_content.append(f"       | **Description:** {desc_lines[0]}")
                    for line in desc_lines[1:]:
                        rst_content.append(f"       | {line}")

                if "valid_values" in param_data:
                    valid_values = param_data["valid_values"]
                    if isinstance(valid_values, list):
                        formatted_values = []
                        for v in valid_values:
                            if isinstance(v, str):
                                formatted_values.append(f'"{v}"')
                            else:
                                formatted_values.append(str(v))
                        rst_content.append(
                            f"       | **Valid Values:** {', '.join(formatted_values)}"
                        )
                    else:
                        rst_content.append(f"       | **Valid Values:** {valid_values}")

                if "example" in param_data:
                    example = param_data.get("example", "")
                    if isinstance(example, str):
                        # Handle multiline string examples - just show as text in table
                        if "\n" in example:
                            rst_content.append(f"       | **Example:**")
                            for line in example.split("\n"):
                                line = line.strip()
                                if line:
                                    rst_content.append(f"       | {line}")
                        else:
                            rst_content.append(f'       | **Example:** "{example}"')
                    else:
                        rst_content.append(f"       | **Example:** {example}")

        rst_content.append("")
    # Methods section
    if "methods" in data:
        rst_content.extend(
            ["Model-specific utility methods", "------------------------------", ""]
        )
        methods = list(data["methods"].items())
        for method_idx, (method_name, method_data) in enumerate(methods):
            # Function name as subsection header
            func_title = f"``{method_name}()``"
            rst_content.append(func_title)
            rst_content.append("~" * len(func_title))
            rst_content.append("")

            # Description as plain prose
            if "description" in method_data:
                desc = method_data["description"].strip()
                for line in desc.split("\n"):
                    rst_content.append(line.strip())
                rst_content.append("")

            # Parameter table — only if parameters exist
            if "parameters" in method_data:
                rst_content.append(".. list-table::")
                rst_content.append("   :widths: 20 80")
                rst_content.append("   :header-rows: 0")
                rst_content.append("")
                for param_name, param_data in method_data["parameters"].items():
                    rst_content.append(f"   * - **{param_name}**")
                    rst_content.append(
                        f"     - | **Type:** ``{param_data.get('type', '')}``"
                    )
                    required = param_data.get("required", False)
                    rst_content.append(
                        f"       | **Required:** {'Yes' if required else 'No'}"
                    )
                    if "default" in param_data:
                        rst_content.append(
                            f"       | **Default:** {param_data['default']}"
                        )
                    if "description" in param_data:
                        desc = param_data.get("description", "").strip()
                        desc_lines = desc.split("\n")
                        rst_content.append(f"       | **Description:** {desc_lines[0]}")
                        for line in desc_lines[1:]:
                            rst_content.append(f"       | {line}")
                rst_content.append("")

            # Example code block
            if "example" in method_data:
                rst_content.append(".. code-block:: python")
                rst_content.append("")
                example = method_data["example"].strip()
                for line in example.split("\n"):
                    rst_content.append(f"    {line}")
                rst_content.append("")

            # Horizontal rule between methods, not after the last one
            if method_idx < len(methods) - 1:
                rst_content.append("----")
                rst_content.append("")

    # Performance section
    rst_content.extend(["Performance", "----------", ""])

    performance_data = data.get("performance", {})

    # Show non-plot metrics first
    has_metrics = False

    # Handle 'metrics' as a special case - can be dict, list, or other
    if "metrics" in performance_data:
        metrics = performance_data["metrics"]

        if isinstance(metrics, dict):
            # Dict format: {metric_name: value, ...}
            rst_content.append("**Metrics:**")
            rst_content.append("")
            for metric_key, metric_value in metrics.items():
                formatted_metric_key = metric_key.replace("_", " ").title()
                rst_content.append(f"* **{formatted_metric_key}**: {metric_value}")
            rst_content.append("")
            has_metrics = True

        elif isinstance(metrics, list):
            # List format: [{name: ..., value: ...}, ...]
            rst_content.append("**Metrics:**")
            rst_content.append("")
            for metric_item in metrics:
                if isinstance(metric_item, dict):
                    name = metric_item.get("name", "")
                    value = metric_item.get("value", "")
                    if name and value:
                        rst_content.append(f"* **{name}**: {value}")
                else:
                    # If list item is not a dict, just display it
                    rst_content.append(f"* {metric_item}")
            rst_content.append("")
            has_metrics = True

        else:
            # If metrics is neither dict nor list, treat it as a simple value
            rst_content.append("**Metrics:**")
            rst_content.append("")
            rst_content.append(f"* {metrics}")
            rst_content.append("")
            has_metrics = True

    # Show other non-plot, non-metrics items
    other_items = []
    for key, value in performance_data.items():
        if key not in ["accuracy_plots", "metrics"]:
            formatted_key = key.replace("_", " ").title()
            other_items.append((formatted_key, value))

    if other_items:
        if not has_metrics:
            rst_content.append("**Metrics:**")
            rst_content.append("")
        for key, value in other_items:
            rst_content.append(f"* **{key}**: {value}")
        rst_content.append("")

    if "accuracy_plots" in performance_data:
        rst_content.append("**Accuracy Plots (AWS directory):**")
        rst_content.append("")
        for plot in performance_data["accuracy_plots"]:
            rst_content.append(f"* ``{plot}``")
        rst_content.append("")

    # Example Usage section
    rst_content.extend(["Example Usage", "------------", ""])
    rst_content.append("")
    rst_content.append(".. code-block:: python")
    rst_content.append("")

    # Generate example code
    example_code = [
        "from berg import BERG",
        "",
        "# Initialize BERG",
        'berg = BERG(berg_dir="path/to/brain-encoding-response-generator")',
        "",
    ]

    # Collect parameters for each function
    get_model_params = []
    encode_params = []
    metadata_params = []
    has_selection = False
    selection_example = {}

    for param_name, param_data in data.get("parameters", {}).items():
        functions = param_data.get("function", "")
        # Handle comma-separated functions
        if isinstance(functions, str):
            function_list = [f.strip() for f in functions.split(",")]
        else:
            function_list = [functions]

        for function in function_list:
            if function == "get_encoding_model":
                if param_name == "selection":
                    has_selection = True
                    if "properties" in param_data:
                        for prop_name, prop_data in param_data["properties"].items():
                            if "example" in prop_data:
                                selection_example[prop_name] = prop_data["example"]
                # Include required params (except model_id and device) OR optional params with examples
                elif param_name not in ["model_id", "device"]:
                    if "example" in param_data:
                        # Include if: required OR (optional but has example different from default)
                        is_required = param_data.get("required", False)
                        example_val = param_data["example"]
                        default_val = param_data.get("default")

                        # Include if required, or if optional with example different from default
                        if is_required or (example_val != default_val):
                            if isinstance(example_val, str):
                                param_str = f'{param_name}="{example_val}"'
                            else:
                                param_str = f"{param_name}={example_val}"
                            # Avoid duplicates
                            if param_str not in get_model_params:
                                get_model_params.append(param_str)

            elif function == "encode":
                if param_name not in ["model", "stimulus", "return_metadata"]:
                    if "example" in param_data and param_data.get(
                        "example"
                    ) != param_data.get("default"):
                        if isinstance(param_data["example"], str):
                            param_str = f'{param_name}="{param_data["example"]}"'
                        else:
                            param_str = f"{param_name}={param_data['example']}"
                        # Avoid duplicates
                        if param_str not in encode_params:
                            encode_params.append(param_str)

            elif function == "get_model_metadata":
                if param_data.get("required", False) and param_name != "model_id":
                    if "example" in param_data:
                        example_val = param_data["example"]
                        if isinstance(example_val, str):
                            param_str = f'{param_name}="{example_val}"'
                        else:
                            param_str = f"{param_name}={example_val}"
                        # Avoid duplicates
                        if param_str not in metadata_params:
                            metadata_params.append(param_str)

    # Build get_encoding_model example
    example_code.append("# Load the model")
    example_code.append("model = berg.get_encoding_model(")
    example_code.append(f'    "{model_id}",')

    for param in get_model_params:
        example_code.append(f"    {param},")

    if has_selection and selection_example:
        example_code.append("    selection={")
        selection_items = list(selection_example.items())
        for idx, (key, value) in enumerate(selection_items):
            if isinstance(value, str):
                line = f'        "{key}": "{value}"'
            elif isinstance(value, list):
                if all(isinstance(x, str) for x in value):
                    formatted_list = (
                        "[" + ", ".join([f'"{item}"' for item in value]) + "]"
                    )
                else:
                    formatted_list = str(value)
                line = f'        "{key}": {formatted_list}'
            else:
                line = f'        "{key}": {value}'

            # Add comma if not the last item
            if idx < len(selection_items) - 1:
                line += ","
            example_code.append(line)
        example_code.append("    }")

    example_code.append(")")
    example_code.append("")

    # Add stimulus preparation - use example from YAML if available
    stimulus_param = data.get("parameters", {}).get("stimulus", {})
    stimulus_example = stimulus_param.get("example", "")
    input_data = data.get("input", {})
    input_example = input_data.get("example", "")

    # Determine stimulus type and create appropriate example
    if stimulus_example or input_example:
        # Use the example from the YAML
        example_to_use = stimulus_example if stimulus_example else input_example

        # Clean up the example if it's multiline
        if not isinstance(example_to_use, str):
            example_to_use = str(example_to_use)
        else:
            example_to_use = example_to_use.strip()

        # Check if this looks like a text/language stimulus
        if (
            "list[str]" in input_data.get("type", "").lower()
            or isinstance(example_to_use, str)
            and example_to_use.startswith("[")
        ):
            example_code.append("# Prepare the stimulus (text/sentences)")
            # Use the example directly
            if "\n" in example_to_use:
                # Multiline - add as is
                example_code.append(f"stimulus = {example_to_use}")
            else:
                example_code.append(f"stimulus = {example_to_use}")
        else:
            # Default to image stimulus
            example_code.append("# Prepare the stimulus images")
            example_code.append(
                "# Image shape should be [batch_size, 3 RGB channels, height, width]"
            )
            example_code.append(
                "stimulus = np.random.randint(0, 255, (100, 3, 256, 256))"
            )
    else:
        # Fallback to image stimulus
        example_code.append("# Prepare the stimulus images")
        example_code.append(
            "# Image shape should be [batch_size, 3 RGB channels, height, width]"
        )
        example_code.append("stimulus = np.random.randint(0, 255, (100, 3, 256, 256))")

    example_code.append("")

    # Build encode example
    example_code.append(
        "# Generates the in silico neural responses using the encoding model previously loaded"
    )
    example_code.append("responses = berg.encode(")
    example_code.append("    model,")
    example_code.append("    stimulus,")

    if encode_params:
        for param in encode_params:
            example_code.append(f"    {param}")
    else:
        example_code.append("    show_progress=True")

    example_code.append(")")
    example_code.append("")

    # Add output information
    output_data = data.get("output", {})
    if output_data:
        output_type = output_data.get("type", "")
        output_shape = output_data.get("shape", "")

        if output_type and output_shape:
            example_code.append(
                f"# The in silico fMRI responses will be a {output_type} of shape:"
            )
            example_code.append(f"# {output_shape}")

        dimensions = output_data.get("dimensions", [])
        if dimensions:
            example_code.append("# where:")
            for dim in dimensions:
                name = dim.get("name", "")
                desc = dim.get("description", "").strip()
                if name and desc and name != "batch_size":
                    if "lh_vertices" in name.lower():
                        example_code.append(
                            f"# - {name} is the number of selected left hemisphere (LH) vertices for which the in silico"
                        )
                        example_code.append("#   fMRI responses are generated.")
                    elif "rh_vertices" in name.lower():
                        example_code.append(
                            f"# - {name} is the number of selected right hemisphere (RH) vertices for which the in silico"
                        )
                        example_code.append("#   fMRI responses are generated.")
                    else:
                        # Handle multiline descriptions in code comments
                        if "\n" in desc:
                            desc_lines = desc.split("\n")
                            # First line
                            example_code.append(f"# - {name}: {desc_lines[0].strip()}")
                            # Remaining lines with proper indentation
                            for line in desc_lines[1:]:
                                line = line.strip()
                                if line:
                                    example_code.append(f"#   {line}")
                        else:
                            example_code.append(f"# - {name}: {desc}")

        example_code.append("")

    # Add metadata example with return_metadata
    example_code.append("# Generate in silico neural responses with metadata")
    example_code.append("responses, metadata = berg.encode(")
    example_code.append("    model,")
    example_code.append("    stimulus,")
    example_code.append("    return_metadata=True")
    example_code.append(")")
    example_code.append("")

    # Add get_model_metadata example if that function exists
    if "get_model_metadata" in param_by_function:
        example_code.append(
            "# Load the encoding model's metadata without having to load the model itself"
        )
        example_code.append("metadata = berg.get_model_metadata(")
        example_code.append(f'    "{model_id}",')
        for param in metadata_params:
            example_code.append(f"    {param}")
        example_code.append(")")
        example_code.append("")

    # Add the example code with proper indentation
    for line in example_code:
        rst_content.append(f"    {line}")

    # References section
    rst_content.extend(["", "References", "---------", ""])

    references = data.get("references", [])
    for ref in references:
        if isinstance(ref, dict):
            for key, value in ref.items():
                rst_content.append(f"* {key}: {value}")
        else:
            rst_content.append(f"* {ref}")

    # Convert the list to a string
    rst_text = "\n".join(rst_content)

    # Write to the output file or return as a string
    if output_file:
        with open(output_file, "w") as file:
            file.write(rst_text)
    else:
        return rst_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert YAML model specification to RST format"
    )
    parser.add_argument("yaml_file", help="Input YAML file path")

    args = parser.parse_args()

    output_file = (
        "docs/models/model_cards/"
        + (args.yaml_file.split("/")[-1]).split(".")[0]
        + ".rst"
    )
    if not output_file:
        output_file = os.path.splitext(args.yaml_file)[0] + ".rst"

    yaml_to_rst(args.yaml_file, output_file)
    print(f"Converted {args.yaml_file} to {output_file}")




# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/ecog-zada2025-gpt2_xl.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-cneuromod_algo2025-vibe.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-cneuromod_algo2025-text2fmri.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/brainscore_language.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/brainscore_vision.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-tuckute_2024-GPT2_XL.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/calcium_2p-wang_2025-3DCNN.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/eeg-things_eeg_2-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-mosaic-CNN8_multihead_subAll_verticesVisual.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-mosaic-CNN8_multihead_subNSD_verticesAll.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd_fsaverage-huze.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd_fsaverage-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd-fwrf.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-things_fmri_1-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/meg-things_meg_1-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/utah_array-tvsd-vit_b_32.yaml# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/meg-things_meg_1-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/utah_array-tvsd-vit_b_32.yaml