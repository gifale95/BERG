#!/usr/bin/env python3
import yaml
import re
from typing import Dict, List, Union, Any, Optional
import textwrap
import os

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
        
        # Parse metadata into sections OR flat structure with dict children
        lines = metadata.split("\n")
        sections = {}  # section_name -> list of (key, shape, desc, is_dict_child)
        current_section = None
        top_level_entries = []
        current_parent_indent = -1  # Track indent of last dict entry
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            if not line_stripped:
                i += 1
                continue
            
            indent_level = len(line) - len(line.lstrip())
            
            # Check if this is a section header (quoted string ending with :, low indent)
            if (line_stripped.startswith("'") and line_stripped.endswith("':") and indent_level <= 2):
                # This is a section header
                current_section = line_stripped.rstrip(":")
                sections[current_section] = []
                current_parent_indent = -1  # Reset parent tracking in new section
                i += 1
                continue
            
            # Parse entry: key : shape - description
            if ":" in line_stripped:
                # Check if it's "key : dict" or "key : shape" format (no description)
                if " - " not in line_stripped:
                    key_part, shape_part = line_stripped.split(":", 1)
                    key = key_part.strip().strip("'")
                    shape = shape_part.strip()
                    desc = ""
                    
                    # Determine if this belongs to current section or top-level
                    if current_section is None and indent_level <= 2:
                        is_dict_child = False
                        top_level_entries.append((key, shape, desc, is_dict_child))
                        # Track if this is a dict for subsequent children
                        if shape.lower() == "dict":
                            current_parent_indent = indent_level
                        else:
                            current_parent_indent = -1
                    elif current_section is not None:
                        # Is this a dict child? Section items are indent 4, dict children are indent 8+
                        is_dict_child = indent_level >= 8
                        sections[current_section].append((key, shape, desc, is_dict_child))
                        if not is_dict_child and shape.lower() == "dict":
                            current_parent_indent = indent_level
                        elif not is_dict_child:
                            current_parent_indent = -1
                    
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
                        is_new_key = (
                            next_indent <= indent_level + 5 or
                            ((":" in next_line_stripped) and 
                             next_indent < indent_level + 15)
                        )
                        
                        if not is_new_key and next_indent > indent_level:
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
                    
                    # Determine if this belongs to current section or top-level
                    if current_section is None and indent_level <= 2:
                        # Top-level flat structure
                        is_dict_child = False
                        top_level_entries.append((key, shape, desc, is_dict_child))
                        # Track if this is a dict for subsequent children
                        if shape.lower() == "dict":
                            current_parent_indent = indent_level
                        else:
                            current_parent_indent = -1
                    elif current_section is None and indent_level > 2 and current_parent_indent >= 0 and indent_level > current_parent_indent:
                        # This is a child of the previous dict in flat structure
                        is_dict_child = True
                        top_level_entries.append((key, shape, desc, is_dict_child))
                    elif current_section is not None:
                        # Is this a dict child? Section items are indent 4, dict children are indent 8+
                        is_dict_child = indent_level >= 8
                        sections[current_section].append((key, shape, desc, is_dict_child))
                        if not is_dict_child and shape.lower() == "dict":
                            current_parent_indent = indent_level
                        elif not is_dict_child:
                            current_parent_indent = -1
            else:
                i += 1
        
        # Generate RST output
        has_nested = False
        
        # Create table for top-level entries if they exist
        if top_level_entries:
            rst_content.append(".. list-table::")
            rst_content.append("   :widths: 30 20 50")
            rst_content.append("   :header-rows: 1")
            rst_content.append("")
            rst_content.append("   * - Key")
            rst_content.append("     - Shape/Type")
            rst_content.append("     - Description")
            
            for key, shape, desc, is_dict_child in top_level_entries:
                if is_dict_child:
                    # Dict child - add arrow indentation
                    has_nested = True
                    rst_content.append(f"   * - |nbsp| |nbsp| |nbsp| |nbsp| |rarr| {key}")
                elif shape.lower() == "dict":
                    rst_content.append(f"   * - **{key}**")
                else:
                    rst_content.append(f"   * - {key}")
                
                rst_content.append(f"     - ``{shape}``")
                
                if "\n" in desc:
                    desc_lines = desc.split("\n")
                    rst_content.append(f"     - {desc_lines[0]}")
                    for desc_line in desc_lines[1:]:
                        rst_content.append(f"       {desc_line}")
                else:
                    rst_content.append(f"     - {desc}")
            
            rst_content.append("")
        
        # Create separate tables for each section
        for section_name, entries in sections.items():
            rst_content.append(f"**{section_name}**")
            rst_content.append("")
            rst_content.append(".. list-table::")
            rst_content.append("   :widths: 30 20 50")
            rst_content.append("   :header-rows: 1")
            rst_content.append("")
            rst_content.append("   * - Key")
            rst_content.append("     - Shape/Type")
            rst_content.append("     - Description")
            
            for key, shape, desc, is_dict_child in entries:
                if is_dict_child:
                    # Dict child - add arrow indentation
                    has_nested = True
                    rst_content.append(f"   * - |nbsp| |nbsp| |nbsp| |nbsp| |rarr| {key}")
                else:
                    # Regular entry or dict parent
                    if shape.lower() == "dict":
                        rst_content.append(f"   * - **{key}**")
                    else:
                        rst_content.append(f"   * - {key}")
                
                rst_content.append(f"     - ``{shape}``")
                
                if "\n" in desc:
                    desc_lines = desc.split("\n")
                    rst_content.append(f"     - {desc_lines[0]}")
                    for desc_line in desc_lines[1:]:
                        rst_content.append(f"       {desc_line}")
                else:
                    rst_content.append(f"     - {desc}")
            
            rst_content.append("")
        
        # Add unicode definitions if we have nested items
        if has_nested:
            rst_content.insert(3, ".. |nbsp| unicode:: 0xA0")
            rst_content.insert(4, "   :trim:")
            rst_content.insert(5, "")
            rst_content.insert(6, ".. |rarr| unicode:: 0x2192")
            rst_content.insert(7, "   :trim:")
            rst_content.insert(8, "")
        
        rst_content.append("")
    
    # Input section
    rst_content.extend(["Input", "-----", ""])
    input_data = data.get("input", {})
    rst_content.append(f"**Type**: ``{input_data.get('type', '')}``  ")
    rst_content.append(f"**Shape**: ``{input_data.get('shape', '')}``  ")
    rst_content.append(f"**Description**: {input_data.get('description', '')}")
    rst_content.append("")
    
    # Input constraints
    if "constraints" in input_data:
        rst_content.append("**Constraints:**")
        rst_content.append("")
        for constraint in input_data["constraints"]:
            rst_content.append(f"* {constraint}")
        rst_content.append("")
    
    # Output section
    rst_content.extend(["Output", "------", ""])
    output_data = data.get("output", {})
    rst_content.append(f"**Type**: ``{output_data.get('type', '')}``  ")
    rst_content.append(f"**Shape**: ``{output_data.get('shape', '')}``  ")
    rst_content.append("**Description**:  ")
    
    # Handle multiline output description
    output_description = output_data.get("description", "").strip()
    
    if "\n" in output_description:
        output_lines = output_description.split("\n")
        
        for line in output_lines:
            line = line.strip()
            if not line:
                rst_content.append("")
                continue
            
            # Check if the line is a bullet point
            if line.startswith("-") or line.startswith("*"):
                if line.startswith("-"):
                    line = "* " + line[1:].strip()
                rst_content.append(line)
            else:
                rst_content.append(line)
    else:
        rst_content.append(output_description)
    
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
            rst_content.append(f"     - {dim.get('description', '')}")
        
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
                rst_content.append(f"       | **Required:** {'Yes' if required else 'No'}")
                
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
                    rst_content.append(f"       |     **Type:** {prop_data.get('type', '')}")
                    
                    if "description" in prop_data:
                        prop_desc = prop_data.get("description", "").strip()
                        prop_desc_lines = prop_desc.split("\n")
                        rst_content.append(f"       |     **Description:** {prop_desc_lines[0]}")
                        for line in prop_desc_lines[1:]:
                            rst_content.append(f"       |     {line}")
                    
                    if "valid_values" in prop_data:
                        valid_values = prop_data["valid_values"]
                        if isinstance(valid_values, list):
                            formatted_values = ", ".join([f'"{v}"' for v in valid_values])
                            rst_content.append(f"       |     **Valid values:** {formatted_values}")
                        else:
                            rst_content.append(f"       |     **Valid values:** {valid_values}")
                    
                    if "example" in prop_data:
                        example = prop_data["example"]
                        if isinstance(example, list):
                            if len(example) > 10:
                                example_str = str(example[:5])[:-1] + ", ... ]"
                            else:
                                example_str = str(example)
                            rst_content.append(f"       |     **Example:** {example_str}")
                        else:
                            rst_content.append(f"       |     **Example:** {example}")
            else:
                # Regular parameter handling
                rst_content.append(f"   * - **{param_name}**")
                rst_content.append(f"     - | **Type:** {param_data.get('type', '')}")
                
                required = param_data.get("required", False)
                rst_content.append(f"       | **Required:** {'Yes' if required else 'No'}")
                
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
                        rst_content.append(f"       | **Valid Values:** {', '.join(formatted_values)}")
                    else:
                        rst_content.append(f"       | **Valid Values:** {valid_values}")
                
                if "example" in param_data:
                    example = param_data.get('example', '')
                    if isinstance(example, str):
                        rst_content.append(f'       | **Example:** "{example}"')
                    else:
                        rst_content.append(f"       | **Example:** {example}")
        
        rst_content.append("")
    
    # Performance section
    rst_content.extend(["Performance", "----------", ""])
    
    performance_data = data.get("performance", {})
    
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
        "berg = BERG(berg_dir=\"path/to/brain-encoding-response-generator\")",
        ""
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
                elif param_data.get("required", False) and param_name not in ["model_id", "device"]:
                    if "example" in param_data:
                        example_val = param_data["example"]
                        if isinstance(example_val, str):
                            param_str = f'{param_name}="{example_val}"'
                        else:
                            param_str = f"{param_name}={example_val}"
                        # Avoid duplicates
                        if param_str not in get_model_params:
                            get_model_params.append(param_str)
            
            elif function == "encode":
                if param_name not in ["model", "stimulus", "return_metadata"]:
                    if "example" in param_data and param_data.get("example") != param_data.get("default"):
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
                    formatted_list = "[" + ", ".join([f'"{item}"' for item in value]) + "]"
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
    
    # Add stimulus preparation
    example_code.append("# Prepare the stimulus images")
    example_code.append("# Image shape should be [batch_size, 3 RGB channels, height, width]")
    example_code.append("images = np.random.randint(0, 255, (100, 3, 256, 256))")
    example_code.append("")
    
    # Build encode example
    example_code.append("# Generates the in silico neural responses to images using the encoding model previously loaded")
    example_code.append("responses = berg.encode(")
    example_code.append("    model,")
    example_code.append("    images,")
    
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
            example_code.append(f"# The in silico fMRI responses will be a {output_type} of shape:")
            example_code.append(f"# {output_shape}")
        
        dimensions = output_data.get("dimensions", [])
        if dimensions:
            example_code.append("# where:")
            for dim in dimensions:
                name = dim.get("name", "")
                desc = dim.get("description", "")
                if name and desc and name != "batch_size":
                    if "lh_vertices" in name.lower():
                        example_code.append(f"# - {name} is the number of selected left hemisphere (LH) vertices for which the in silico")
                        example_code.append("#   fMRI responses are generated.")
                    elif "rh_vertices" in name.lower():
                        example_code.append(f"# - {name} is the number of selected right hemisphere (RH) vertices for which the in silico")
                        example_code.append("#   fMRI responses are generated.")
                    else:
                        example_code.append(f"# - {name}: {desc}")
        
        example_code.append("")
    
    # Add metadata example with return_metadata
    example_code.append("# Generate in silico neural responses with metadata")
    example_code.append("responses, metadata = berg.encode(")
    example_code.append("    model,")
    example_code.append("    images,")
    example_code.append("    return_metadata=True")
    example_code.append(")")
    example_code.append("")
    
    # Add get_model_metadata example if that function exists
    if "get_model_metadata" in param_by_function:
        example_code.append("# Load the encoding model's metadata without having to load the model itself")
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
    
    parser = argparse.ArgumentParser(description="Convert YAML model specification to RST format")
    parser.add_argument("yaml_file", help="Input YAML file path")
    
    args = parser.parse_args()
    
    output_file = "docs/models/model_cards/" + (args.yaml_file.split("/")[-1]).split(".")[0] + ".rst"
    if not output_file:
        output_file = os.path.splitext(args.yaml_file)[0] + ".rst"
    
    yaml_to_rst(args.yaml_file, output_file)
    print(f"Converted {args.yaml_file} to {output_file}")
    
    
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/eeg-things_eeg_2-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-mosaic-CNN8_multihead_subAll_verticesVisual.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-mosaic-CNN8_multihead_subNSD_verticesAll.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd_fsaverage-huze.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd_fsaverage-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd-fwrf.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-things_fmri_1-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/meg-things_meg_1-vit_b_32.yaml
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/utah_array-tvsd-vit_b_32.yaml