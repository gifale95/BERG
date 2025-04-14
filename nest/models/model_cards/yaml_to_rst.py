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
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    # Generate the RST content
    rst_content = []
    
    # Generate the title
    model_id = data.get('model_id', os.path.basename(yaml_file).replace('.yaml', ''))
    title_line = '=' * len(model_id)
    rst_content.extend([title_line, model_id, title_line, ''])
    
    # Model Summary section
    rst_content.extend(['Model Summary', '------------', ''])
    rst_content.append('.. list-table::')
    rst_content.append('   :widths: 30 70')
    rst_content.append('   :stub-columns: 1')
    rst_content.append('')
    
    # Add model summary items
    summary_items = [
        ('Modality', data.get('modality', '')),
        ('Training Dataset', data.get('training_dataset', '')),
        ('Model Architecture', data.get('model_architecture', '')),
        ('Creator', data.get('creator', ''))
    ]
    
    for item, value in summary_items:
        rst_content.append(f'   * - {item}')
        rst_content.append(f'     - {value}')
    
    rst_content.append('')
    
    # Description section
    rst_content.extend(['Description', '----------', ''])
    description = data.get('description', '').strip()
    # Process multiline description
    for line in description.split('\n'):
        rst_content.append(line)
    rst_content.append('')
    
    # Input section
    rst_content.extend(['Input', '-----', ''])
    input_data = data.get('input', {})
    rst_content.append(f'**Type**: ``{input_data.get("type", "")}``  ')
    rst_content.append(f'**Shape**: ``{input_data.get("shape", "")}``  ')
    rst_content.append(f'**Description**: {input_data.get("description", "")}')
    rst_content.append('')
    
    # Input constraints
    if 'constraints' in input_data:
        rst_content.append('**Constraints:**')
        rst_content.append('')
        for constraint in input_data['constraints']:
            rst_content.append(f'* {constraint}')
        rst_content.append('')
    
    # Output section
    rst_content.extend(['Output', '------', ''])
    output_data = data.get('output', {})
    rst_content.append(f'**Type**: ``{output_data.get("type", "")}``  ')
    rst_content.append(f'**Shape**: ``{output_data.get("shape", "")}``  ')
    rst_content.append('**Description**:  ')
    
    # Handle multiline output description with special formatting for ROIs list
    output_description = output_data.get('description', '').strip()
    if '\n' in output_description:
        output_lines = output_description.split('\n')
        for i, line in enumerate(output_lines):
            line = line.strip()
            if line:
                # Preserve list formatting that may be in the description
                if 'ROI' in line and i > 0 and any(roi in line for roi in ['V1:', 'V2:', 'V3:']):
                    # Assuming this is a list of ROIs
                    for roi_line in output_lines[i:]:
                        if roi_line.strip():
                            rst_content.append(f'* {roi_line.strip()}')
                else:
                    rst_content.append(line)
    else:
        rst_content.append(output_description)
    
    rst_content.append('')
    
    # Output dimensions
    if 'dimensions' in output_data:
        rst_content.append('**Dimensions:**')
        rst_content.append('')
        rst_content.append('.. list-table::')
        rst_content.append('   :widths: 30 70')
        rst_content.append('   :header-rows: 1')
        rst_content.append('')
        rst_content.append('   * - Name')
        rst_content.append('     - Description')
        
        for dim in output_data['dimensions']:
            rst_content.append(f'   * - {dim.get("name", "")}')
            rst_content.append(f'     - {dim.get("description", "")}')
        
        rst_content.append('')
    
    # Parameters section
    rst_content.extend(['Parameters', '---------', ''])
    
    # Group parameters by function
    param_by_function = {}
    for param_name, param_data in data.get('parameters', {}).items():
        function = param_data.get('function', '')
        if function not in param_by_function:
            param_by_function[function] = []
        param_by_function[function].append((param_name, param_data))
    
    # Create subsections for each function
    for function, params in param_by_function.items():
        if function == 'get_encoding_model':
            display_name = 'get_encoding_model'
        elif function == 'encode':
            display_name = 'encode'
        else:
            display_name = function
            
        rst_content.append(f'Parameters used in ``{display_name}``')
        rst_content.append('~' * (len(f'Parameters used in ``{display_name}``')))
        rst_content.append('')
        rst_content.append('.. list-table::')
        rst_content.append('   :widths: 20 80')
        rst_content.append('   :header-rows: 0')
        rst_content.append('')
        
        for param_name, param_data in params:
            rst_content.append(f'   * - **{param_name}**')
            
            # Format parameter details with proper indentation
            details = []
            details.append(f'**Type:** {param_data.get("type", "")}')
            
            required = param_data.get('required', False)
            details.append(f'**Required:** {"Yes" if required else "No"}')
            
            if 'description' in param_data:
                details.append(f'**Description:** {param_data.get("description", "")}')
            
            if 'valid_values' in param_data:
                valid_values = param_data['valid_values']
                if isinstance(valid_values, list):
                    # Format the list of values nicely
                    if all(isinstance(x, str) for x in valid_values) and len(", ".join(valid_values)) > 50:
                        # For longer lists of string values, format them with line breaks
                        formatted_values = ", \n       |                  ".join(
                            [", ".join(valid_values[i:i+5]) for i in range(0, len(valid_values), 5)]
                        )
                        details.append(f'**Valid Values:** {formatted_values}')
                    else:
                        details.append(f'**Valid Values:** {", ".join(map(str, valid_values))}')
                else:
                    details.append(f'**Valid Values:** {valid_values}')
            
            if 'example' in param_data:
                details.append(f'**Example:** {param_data.get("example", "")}')
            
            # Add the formatted details with proper indentation
            for i, detail in enumerate(details):
                if i == 0:  # First line
                    rst_content.append(f'     - | {detail}')
                else:
                    rst_content.append(f'       | {detail}')
        
        rst_content.append('')
    
    # Performance section
    rst_content.extend(['Performance', '----------', ''])
    
    performance_data = data.get('performance', {})
    
    if 'accuracy_plots' in performance_data:
        rst_content.append('**Accuracy Plots:**')
        rst_content.append('')
        for plot in performance_data['accuracy_plots']:
            rst_content.append(f'* ``{plot}``')
        rst_content.append('')
    
    # Example Usage section
    rst_content.extend(['Example Usage', '------------', ''])
    rst_content.append('')
    rst_content.append('.. code-block:: python')
    rst_content.append('')
    
    # Generate example code
    example_code = [
        'from nest import NEST',
        '',
        '# Initialize NEST',
        'nest = NEST(nest_dir="path/to/nest")',
        ''
    ]
    
    # Dynamically create the model loading example based on actual parameters
    get_model_params = []
    encode_params = []
    
    # Find parameters used in get_encoding_model
    for param_name, param_data in data.get('parameters', {}).items():
        if param_data.get('function') == 'get_encoding_model':
            if param_data.get('required', False):
                # For required parameters, use an example value if available
                if 'example' in param_data:
                    example_val = param_data['example']
                    # Format the value based on its type
                    if param_data.get('type') == 'str':
                        if isinstance(example_val, str):
                            get_model_params.append(f'{param_name}="{example_val}"')
                        else:
                            get_model_params.append(f'{param_name}={example_val}')
                    else:
                        get_model_params.append(f'{param_name}={example_val}')
                else:
                    # Use a generic value if no example is provided
                    if param_data.get('type') == 'str':
                        get_model_params.append(f'{param_name}="value"')
                    elif param_data.get('type') == 'int':
                        get_model_params.append(f'{param_name}=1')
                    else:
                        get_model_params.append(f'{param_name}=value')
        
        # Identify what parameters are needed for encode
        elif param_data.get('function') == 'encode':
            if param_name != 'stimulus':  # We'll handle stimulus separately
                if not param_data.get('required', False) and 'default' in param_data:
                    # Only include optional parameters if they have defaults
                    if param_data.get('type') == 'str':
                        encode_params.append(f'{param_name}="{param_data["default"]}"')
                    else:
                        encode_params.append(f'{param_name}={param_data["default"]}')
    
    # Build the model loading line
    get_model_params_str = ", ".join(get_model_params)
    example_code.append(f'# Load the model')
    example_code.append(f'model = nest.get_encoding_model("{model_id}", {get_model_params_str})')
    example_code.append('')
    
    # Add information about the stimulus based on the input definition
    input_data = data.get('input', {})
    input_shape = input_data.get('shape', '')
    if isinstance(input_shape, list) and len(input_shape) > 0:
        example_code.append('# Prepare your stimuli')
        example_code.append(f'# stimulus shape should be {input_shape}')
        example_code.append('')
    
    # Add encode call
    encode_params_str = ", ".join(['stimulus'] + encode_params)
    example_code.append('# Generate responses')
    example_code.append(f'responses = nest.encode(model, {encode_params_str})')
    example_code.append('')
    
    # Add information about the output shape and format based on output definition
    output_data = data.get('output', {})
    output_shape = output_data.get('shape', '')
    if isinstance(output_shape, list) and len(output_shape) > 0:
        example_code.append(f'# responses shape will be {output_shape}')
        
        # Add comments explaining the dimensions if available
        dimensions = output_data.get('dimensions', [])
        if dimensions:
            example_code.append('# where:')
            for dim in dimensions:
                if dim.get('name') != 'batch_size':  # Skip batch_size as it's self-explanatory
                    name = dim.get('name', '')
                    desc = dim.get('description', '')
                    if name and desc:
                        example_code.append(f'# - {name} is {desc}')
    
    # Add metadata example if appropriate based on modality
    if data.get('modality') == 'eeg':
        example_code.extend([
            '',
            '# Get responses with metadata',
            'responses, metadata = nest.encode(model, stimulus, return_metadata=True)',
            '',
            '# Access channel names and time information',
            'channel_names = metadata[\'eeg\'][\'ch_names\']',
            'time_points = metadata[\'eeg\'][\'times\']  # in seconds'
        ])
    
    # Add the example code
    for line in example_code:
        rst_content.append(f'    {line}')
    
    # References section
    rst_content.extend(['', 'References', '---------', ''])
    
    references = data.get('references', [])
    for ref in references:
        rst_content.append(f'* {ref}')
    
    # Convert the list to a string
    rst_text = '\n'.join(rst_content)
    
    # Write to the output file or return as a string
    if output_file:
        with open(output_file, 'w') as file:
            file.write(rst_text)
    else:
        return rst_text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YAML model specification to RST format')
    parser.add_argument('yaml_file', help='Input YAML file path')
    parser.add_argument('output_file', nargs='?', help='Output RST file path (default: input filename with .rst extension)')
    
    args = parser.parse_args()
    
    output_file = args.output_file
    if not output_file:
        output_file = os.path.splitext(args.yaml_file)[0] + '.rst'
    
    yaml_to_rst(args.yaml_file, output_file)
    print(f"Converted {args.yaml_file} to {output_file}")

# Example usage from console:
# 
# 1. Convert a YAML file to RST (output has same name with .rst extension):
#    python yaml_to_rst.py fmri_nsd_fwrf.yaml
#
# 2. Convert a YAML file to RST with specific output path:
#    python yaml_to_rst.py fmri_nsd_fwrf.yaml docs/model_cards/fmri_nsd_fwrf.rst
    
# python nest/models/model_cards/yaml_to_rst.py nest/models/model_cards/fmri_nsd_fwrf.yaml -o nest/models/model_cards/fmri_nsd_fwrf.rst
# python nest/models/model_cards/yaml_to_rst.py nest/models/model_cards/eeg_things_eeg_2_vit_b_32.yaml -o nest/models/model_cards/eeg_things_eeg_2_vit_b_32.rst