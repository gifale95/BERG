from dataclasses import dataclass, fields, field
from huggingface_hub import get_collection, hf_hub_download
import json
import torch


@dataclass(frozen=True)
class Text2fMRIConfig:
    tr: float = 1.49  # Time Resolution in seconds
    num_subjects: int = 4  # Number of subjects
    num_rois: int = 1000  # Number of Regions of Interest
    num_transformer_layers: int = 4  # Number of transformer layers
    num_transformer_heads: int = 8  # Number of transformer heads
    transformer_dim: int = 1024  # d parameter of the transformer
    subject_embedding_dim: int = 128  # dimensionality for the subject embedding
    # Model Name to be passed into AutoCausalLLM in huggingface transformers
    extractor_LLM: str = "Qwen/Qwen2.5-0.5B"
    extractor_LLM_feature_size: int = 896  # Dimensionality of the output of the LLM
    extractor_LLM_dtype: torch.dtype = field(
        default=torch.float16, compare=False, hash=False)  # FP16 for speed/VRAM,
    # Number of last hidden states to extract and average over.
    extractor_LLM_num_last_hidden_states: int = 4


def get_pretrained_model_configs(collection_slug: str) -> dict[Text2fMRIConfig, str]:
    """
    Fetches all valid Text2fMRI models from a Hugging Face Collection.

    This function iterates through a collection, downloads the `config.json` for 
    each model, and attempts to map it to a `Text2fMRIConfig` object.

    Args:
        collection_slug (str): The Hugging Face collection ID (e.g., "username/my-collection").

    Returns:
        Dict[Text2fMRIConfig, str]: A dictionary mapping the configuration object 
        to the specific Hugging Face Repository ID (e.g., "username/model-name").

        Note: If multiple repositories share the exact same configuration (hash),
        the last one processed will overwrite the previous entry in the dictionary.
    """
    # 1. Get the list of models from the collection
    collection = get_collection(collection_slug)
    valid_models = {}

    # 2. Iterate over every model in the collection
    for item in collection.items:
        if item.item_type != "model":
            continue

        try:
            # 3. Download just the config.json (fast, small file)
            config_path = hf_hub_download(
                repo_id=item.item_id, filename="config.json")

            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # 4. Clean and Convert JSON to your Dataclass
            # We filter out keys in the JSON that aren't in your dataclass (like 'architectures')
            valid_keys = {f.name for f in fields(Text2fMRIConfig)}
            filtered_dict = {k: v for k,
                             v in config_dict.items() if k in valid_keys}

            if "extractor_LLM_dtype" in filtered_dict:
                dtype_str = filtered_dict["extractor_LLM_dtype"]
                if dtype_str == "float16":
                    filtered_dict["extractor_LLM_dtype"] = torch.float16
                elif dtype_str == "float32":
                    filtered_dict["extractor_LLM_dtype"] = torch.float32
                elif dtype_str == "bfloat16":
                    filtered_dict["extractor_LLM_dtype"] = torch.bfloat16

            # Create the config object
            config_obj = Text2fMRIConfig(**filtered_dict)

            # Map the config object to the Hugging Face Repo ID
            valid_models[config_obj] = item.item_id

        except Exception as e:
            print(f"Skipping {item.item_id}: Could not load config ({e})")

    return valid_models
