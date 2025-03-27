import yaml
import os
import argparse

def yaml_to_markdown(yaml_path, save_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    model_id = data.get("model_id", "N/A")
    version = data.get("version", "N/A")
    modality = data.get("modality", "N/A")
    dataset = data.get("dataset", "N/A")
    repeats = data.get("repeats", "N/A")
    features = data.get("features", "N/A")
    subject_level = data.get("subject_level", "N/A")

    md = f"# Model: `{model_id}`\n\n"
    md += f"**Version**: {version}  \n"
    md += f"**Modality**: {modality}  \n"
    md += f"**Dataset**: {dataset}  \n"
    md += f"**Repeats**: {repeats}  \n"
    md += f"**Features**: {features}  \n"
    md += f"**Subject-level**: {subject_level}  \n\n"

    md += "## Supported Parameters\n\n"
    params = data.get("supported_parameters", {})
    for param, info in params.items():
        md += f"### `{param}`\n"
        md += f"- **Type**: {info.get('type', 'N/A')}\n"
        md += f"- **Required**: {info.get('required', 'N/A')}\n"
        if 'valid_values' in info:
            vv = info['valid_values']
            vv = ', '.join(map(str, vv)) if isinstance(vv, list) else str(vv)
            md += f"- **Valid Values**: {vv}\n"
        md += f"- **Example**: `{info.get('example', 'N/A')}`\n"
        md += f"- **Description**: {info.get('description', '')}\n\n"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(md)

    print(f"Markdown file saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YAML model metadata to Markdown.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file")
    parser.add_argument("save_path", type=str, help="Path to save the generated Markdown file")
    args = parser.parse_args()

    yaml_to_markdown(args.yaml_path, args.save_path)


# python nest/models/model_cards/yaml_to_md.py nest/models/model_cards/fmri_nsd_fwrf.yaml nest/models/model_cards/fmri_nsd_fwrf.md 