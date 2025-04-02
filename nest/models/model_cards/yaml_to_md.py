import yaml
import os
import argparse
from collections import defaultdict

import yaml
import os
import argparse
from collections import defaultdict

def yaml_to_markdown(yaml_path, save_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    lines = []

    # Title
    model_id = data.get("model_id", "Unnamed Model")
    lines.append(f"# {model_id}")
    lines.append("")

    # Metadata Summary Table
    summary_keys = ["modality", "dataset", "features", "repeats", "subject_level"]
    lines.append("## Model Summary")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|-----|-------|")
    for key in summary_keys:
        value = data.get(key, "N/A")
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    # Description
    if "description" in data:
        lines.append("## Description")
        lines.append("")
        lines.append(data["description"].strip())
        lines.append("")

    # Input
    if "input" in data:
        lines.append("## Input")
        lines.append("")
        input_data = data["input"]
        lines.append(f"**Type**: `{input_data.get('type', 'N/A')}`  ")
        lines.append(f"**Shape**: `{input_data.get('shape', 'N/A')}`  ")
        if "description" in input_data:
            lines.append(f"**Description**: {input_data['description']}  ")
        if "constraints" in input_data:
            lines.append("**Constraints:**")
            for c in input_data["constraints"]:
                lines.append(f"- {c}")
        lines.append("")

    # Output
    if "output" in data:
        lines.append("## Output")
        lines.append("")
        output_data = data["output"]
        lines.append(f"**Type**: `{output_data.get('type', 'N/A')}`  ")
        lines.append(f"**Shape**: `{output_data.get('shape', 'N/A')}`  ")
        if "description" in output_data:
            lines.append(f"**Description**:  \n{output_data['description'].strip()}  ")

        if "dimensions" in output_data:
            lines.append("")
            lines.append("**Dimensions:**")
            lines.append("")
            lines.append("| Name | Description |")
            lines.append("|------|-------------|")
            for dim in output_data["dimensions"]:
                name = dim.get("name", "")
                desc = dim.get("description", "")
                lines.append(f"| `{name}` | {desc} |")
        lines.append("")

    # Parameters grouped by function
    if "parameters" in data:
        lines.append("## Parameters")
        lines.append("")
        param_groups = defaultdict(list)
        for name, param in data["parameters"].items():
            func = param.get("function", "other")
            param_groups[func].append((name, param))

        for func_name, params in param_groups.items():
            lines.append(f"### Parameters used in `{func_name}`")
            lines.append("")
            lines.append("| Name | Type | Required | Description | Example | Valid Values |")
            lines.append("|------|------|----------|-------------|---------|---------------|")
            for name, param in params:
                ptype = param.get("type", "")
                required = param.get("required", False)
                desc = param.get("description", "")
                example = param.get("example", "")
                valid_vals = param.get("valid_values", "")
                if isinstance(valid_vals, list):
                    valid_vals = ", ".join(str(v) for v in valid_vals)
                elif not valid_vals:
                    valid_vals = "-"
                lines.append(f"| `{name}` | `{ptype}` | `{required}` | {desc} | `{example}` | {valid_vals} |")
            lines.append("")

    # Performance
    if "performance" in data:
        lines.append("## Performance")
        lines.append("")
        performance = data["performance"]
        if "accuracy_plots" in performance:
            lines.append("**Accuracy Plots:**")
            for plot in performance["accuracy_plots"]:
                lines.append(f"- `{plot}`")
            lines.append("")
            
            
    if "references" in data:
        lines.append("## References")
        lines.append("")
        performance = data["performance"]

        for ref in data["references"]:
            lines.append(f"- {ref}")
        lines.append("")

    # Save to file
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown file saved to: {save_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YAML model metadata to Markdown.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file")
    parser.add_argument("save_path", type=str, help="Path to save the generated Markdown file")
    args = parser.parse_args()

    yaml_to_markdown(args.yaml_path, args.save_path)


# python nest/models/model_cards/yaml_to_md.py nest/models/model_cards/fmri_nsd_fwrf.yaml nest/models/model_cards/fmri_nsd_fwrf.md 
# python nest/models/model_cards/yaml_to_md.py nest/models/model_cards/eeg_things_eeg_2_vit_b_32.yaml nest/models/model_cards/eeg_things_eeg_2_vit_b_32.md