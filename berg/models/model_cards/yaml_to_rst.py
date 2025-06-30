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
    
    # Handle multiline output description, preserving bullet points
    output_description = output_data.get("description", "").strip()
    
    if "\n" in output_description:
        output_lines = output_description.split("\n")
        
        in_bullet_list = False
        
        for line in output_lines:
            line = line.strip()
            if not line:
                # Add empty lines as is
                rst_content.append("")
                continue
                
            # Check if the line is a bullet point (starts with - or *)
            if line.startswith("-") or line.startswith("*"):
                # This is a bullet point line
                in_bullet_list = True
                # Convert to RST bullet format (*)
                if line.startswith("-"):
                    line = "* " + line[1:].strip()
                rst_content.append(line)
            else:
                in_bullet_list = False
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
    
    # Group parameters by function
    param_by_function = {}
    for param_name, param_data in data.get("parameters", {}).items():
        function = param_data.get("function", "")
        if function not in param_by_function:
            param_by_function[function] = []
        param_by_function[function].append((param_name, param_data))
    
    # Create subsections for each function
    for function, params in param_by_function.items():
        if function == "get_encoding_model":
            display_name = "get_encoding_model"
            # Add description for get_encoding_model function
            function_description = "This function loads the encoding model."
        elif function == "encode":
            display_name = "encode"
            # Add description for encode function
            function_description = "This function generates in silico neural responses using the encoding model previously loaded."
        else:
            display_name = function
            function_description = ""
            
        rst_content.append(f"Parameters used in ``{display_name}``")
        rst_content.append("~" * (len(f"Parameters used in ``{display_name}``")))
        rst_content.append("")
        
        # Add the function description if it exists
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
                # Add the main selection parameter entry
                rst_content.append(f"   * - **{param_name}**")
                
                # Start the parameter details cell with the vertical bar
                rst_content.append(f"     - | **Type:** {param_data.get('type', '')}")
                
                # Add required field
                required = param_data.get("required", False)
                rst_content.append(f"       | **Required:** {'Yes' if required else 'No'}")
                
                # Process description with special handling for multiline text
                if "description" in param_data:
                    desc = param_data.get("description", "").strip()
                    # Split by newlines and handle each line separately
                    desc_lines = desc.split("\n")
                    # First description line
                    rst_content.append(f"       | **Description:** {desc_lines[0]}")
                    # Any additional description lines
                    for line in desc_lines[1:]:
                        rst_content.append(f"       | {line}")
                
                # Add properties header with an empty line before it
                rst_content.append("       | ")
                rst_content.append("       | **Properties:**")
                
                # Process each property
                for prop_name, prop_data in param_data["properties"].items():
                    # Add an empty line before each property for better readability
                    rst_content.append("       | ")
                    rst_content.append(f"       | **{prop_name}**")
                    rst_content.append(f"       |     **Type:** {prop_data.get('type', '')}")
                    
                    # Process property description with careful handling of newlines
                    if "description" in prop_data:
                        prop_desc = prop_data.get("description", "").strip()
                        # Split and handle each line separately
                        prop_desc_lines = prop_desc.split("\n")
                        # First description line
                        rst_content.append(f"       |     **Description:** {prop_desc_lines[0]}")
                        # Any additional description lines
                        for line in prop_desc_lines[1:]:
                            rst_content.append(f"       |     {line}")
                    
                    # Add valid values if available
                    if "valid_values" in prop_data:
                        valid_values = prop_data["valid_values"]
                        if isinstance(valid_values, list):
                            # Format as comma-separated string wrapped in quotes
                            formatted_values = ", ".join([f'"{v}"' for v in valid_values])
                            rst_content.append(f"       |     **Valid values:** {formatted_values}")
                        else:
                            rst_content.append(f"       |     **Valid values:** {valid_values}")
                    
                    # Add example if available
                    if "example" in prop_data:
                        example = prop_data["example"]
                        if isinstance(example, list):
                            if len(example) > 10:
                                # Truncate long examples
                                example_str = str(example[:5])[:-1] + ", ... ]"
                            else:
                                example_str = str(example)
                            rst_content.append(f"       |     **Example:** {example_str}")
                        else:
                            rst_content.append(f"       |     **Example:** {example}")
            else:
                # Regular parameter handling
                rst_content.append(f"   * - **{param_name}**")
                
                # Format parameter details with proper indentation
                rst_content.append(f"     - | **Type:** {param_data.get('type', '')}")
                
                required = param_data.get("required", False)
                rst_content.append(f"       | **Required:** {'Yes' if required else 'No'}")
                
                if "description" in param_data:
                    desc = param_data.get("description", "").strip()
                    # Handle multiline descriptions
                    desc_lines = desc.split("\n")
                    rst_content.append(f"       | **Description:** {desc_lines[0]}")
                    for line in desc_lines[1:]:
                        rst_content.append(f"       | {line}")
                
                if "valid_values" in param_data:
                    valid_values = param_data["valid_values"]
                    if isinstance(valid_values, list):
                        rst_content.append(f"       | **Valid Values:** {', '.join(map(str, valid_values))}")
                    else:
                        rst_content.append(f"       | **Valid Values:** {valid_values}")
                
                if "example" in param_data:
                    rst_content.append(f"       | **Example:** {param_data.get('example', '')}")
        
        rst_content.append("")
    
    # Performance section
    rst_content.extend(["Performance", "----------", ""])
    
    performance_data = data.get("performance", {})
    
    if "accuracy_plots" in performance_data:
        rst_content.append("**Accuracy Plots:**")
        rst_content.append("")
        for plot in performance_data["accuracy_plots"]:
            rst_content.append(f"* ``{plot}``")
        rst_content.append("")
    
    # Example Usage section
    rst_content.extend(["Example Usage", "------------", ""])
    rst_content.append("")
    rst_content.append(".. code-block:: python")
    rst_content.append("")
    
    # Generate example code - FIXED: use berg_dir instead of berg
    example_code = [
        "from berg import BERG",
        "",
        "# Initialize BERG",
        "berg = BERG(berg_dir=\"path/to/brain-encoding-response-generator\")",
        ""
    ]
    
    # Dynamically create the model loading example based on actual parameters
    get_model_params = []
    encode_params = []
    has_selection = False
    selection_example = {}
    device_param = None
    
    # Find parameters used in get_encoding_model and encode
    for param_name, param_data in data.get("parameters", {}).items():
        if param_data.get("function") == "get_encoding_model":
            if param_name == "selection":
                has_selection = True
                # Build selection example from properties
                if "properties" in param_data:
                    for prop_name, prop_data in param_data["properties"].items():
                        if "example" in prop_data:
                            selection_example[prop_name] = prop_data["example"]
                        elif prop_name == "roi" and "valid_values" in prop_data:
                            # Fallback for roi if no example provided
                            selection_example["roi"] = prop_data["valid_values"][0]
            elif param_data.get("required", False):
                # For required parameters, use an example value if available
                if "example" in param_data:
                    example_val = param_data["example"]
                    # Format the value based on its type
                    if param_data.get("type") == "str":
                        if isinstance(example_val, str):
                            get_model_params.append(f"{param_name}={example_val}")
                        else:
                            get_model_params.append(f"{param_name}={example_val}")
                    else:
                        get_model_params.append(f"{param_name}={example_val}")
                else:
                    # Use a generic value if no example is provided
                    if param_data.get("type") == "str":
                        get_model_params.append(f"{param_name}=\"value\"")
                    elif param_data.get("type") == "int":
                        get_model_params.append(f"{param_name}=1")
                    else:
                        get_model_params.append(f"{param_name}=value")
        
        # Look for device parameter in encode function to add to get_encoding_model
        elif param_data.get("function") == "encode":
            if param_name == "device" and "example" in param_data:
                device_param = f"device=\"{param_data['example']}\""
            elif param_name != "stimulus":  # We'll handle stimulus separately
                # Collect other encode parameters for later use
                if "example" in param_data and param_data.get("example") != param_data.get("default"):
                    if param_data.get("type") == "str":
                        encode_params.append(f"{param_name}=\"{param_data['example']}\"")
                    else:
                        encode_params.append(f"{param_name}={param_data['example']}")
    
    # Build the model loading section
    example_code.append("# Load the model")
    
    # Always use multi-line format for better readability
    example_code.append("model = berg.get_encoding_model(")
    example_code.append(f"    \"{model_id}\",")
    
    # Add required parameters
    for param in get_model_params:
        example_code.append(f"    {param},")
    
    # Add selection if it exists
    if has_selection and selection_example:
        example_code.append("    selection={")
        for key, value in selection_example.items():
            if isinstance(value, str):
                example_code.append(f"        \"{key}\": \"{value}\"")
            elif isinstance(value, list):
                # Handle list values properly
                if all(isinstance(x, str) for x in value):
                    # List of strings
                    formatted_list = "[" + ", ".join([f'"{item}"' for item in value]) + "]"
                else:
                    # List of other types (like arrays)
                    formatted_list = str(value)
                example_code.append(f"        \"{key}\": {formatted_list}")
            else:
                example_code.append(f"        \"{key}\": {value}")
        example_code.append("    },")
    
    # Add device parameter
    if device_param:
        example_code.append(f"    {device_param}")
    
    example_code.append(")")
    example_code.append("")
    
    # Add information about the stimulus based on the input definition
    input_data = data.get("input", {})
    input_shape = input_data.get("shape", "")
    
    example_code.append("# Prepare the stimulus images")
    if isinstance(input_shape, list) and len(input_shape) > 0:
        # Create a more descriptive comment based on shape
        shape_description = str(input_shape).replace("'", "")
        if "3" in str(input_shape) and ("height" in str(input_shape) or "width" in str(input_shape)):
            example_code.append("# Image shape should be [batch_size, 3 RGB channels, height, width]")
        else:
            example_code.append(f"# Image shape should be {shape_description}")
    
    # Add example stimulus creation
    example_code.append("images = np.random.randint(0, 255, (100, 3, 256, 256))")
    example_code.append("")
    
    # Add encode call with improved formatting and comments
    example_code.append("# Generates the in silico neural responses to images using the encoding model previously loaded")
    example_code.append("responses = berg.encode(")
    example_code.append("    model,")
    example_code.append("    images,")
    
    # Add encode parameters if they exist
    if encode_params:
        for param in encode_params:
            example_code.append(f"    {param}")
    else:
        # Add show_progress as a common parameter
        example_code.append("    show_progress=True")
    
    example_code.append(")")
    example_code.append("")
    
    # Add output information based on the YAML output definition
    output_data = data.get("output", {})
    if output_data:
        # Add output description with improved formatting
        output_type = output_data.get("type", "")
        output_shape = output_data.get("shape", "")
        
        if output_type and output_shape:
            example_code.append(f"# The in silico fMRI responses will be a {output_type} of shape:")
            example_code.append(f"# {output_shape}")
        
        # Add dimension explanations (exclude batch_size as it's self-explanatory)
        dimensions = output_data.get("dimensions", [])
        if dimensions:
            example_code.append("# where:")
            for dim in dimensions:
                name = dim.get("name", "")
                desc = dim.get("description", "")
                if name and desc and name != "batch_size":
                    # Improve the description formatting
                    if "lh_vertices" in name.lower():
                        example_code.append(f"# - {name} is the number of selected left hemisphere (LH) vertices for which the in silico")
                        example_code.append("#   fMRI responses are generated.")
                    elif "rh_vertices" in name.lower():
                        example_code.append(f"# - {name} is the number of selected right hemisphere (RH) vertices for which the in silico")
                        example_code.append("#   fMRI responses are generated.")
                    else:
                        example_code.append(f"# - {name}: {desc}")
        
        example_code.append("")
    
    # Add metadata example
    example_code.append("# Generate in silico neural responses with metadata")
    example_code.append("responses, metadata = berg.encode(")
    example_code.append("    model,")
    example_code.append("    images,")
    example_code.append("    return_metadata=True")
    example_code.append(")")
    example_code.append("")
    
    # Add the example code
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
    parser.add_argument("output_file", nargs="?", help="Output RST file path (default: input filename with .rst extension)")
    
    args = parser.parse_args()
    
    output_file = args.output_file
    if not output_file:
        output_file = os.path.splitext(args.yaml_file)[0] + ".rst"
    
    yaml_to_rst(args.yaml_file, output_file)
    print(f"Converted {args.yaml_file} to {output_file}")

# Example usage from console:
# 
# 1. Convert a YAML file to RST (output has same name with .rst extension):
#    python yaml_to_rst.py fmri_nsd_fwrf.yaml
#
# 2. Convert a YAML file to RST with specific output path:
#    python yaml_to_rst.py fmri_nsd_fwrf.yaml docs/model_cards/fmri_nsd_fwrf.rst
    
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/fmri-nsd-fwrf.yaml source/models/model_cards/fmri-nsd-fwrf.rst
# python berg/models/model_cards/yaml_to_rst.py berg/models/model_cards/eeg-things_eeg_2-vit_b_32.yaml source/models/model_cards/eeg-things_eeg_2-vit_b_32.rst