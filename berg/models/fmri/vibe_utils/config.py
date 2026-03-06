from dataclasses import dataclass, fields, field
from typing import Literal
from huggingface_hub import get_collection, hf_hub_download
import json
import torch


@dataclass(frozen=True)
class VIBEConfig:
    tr: float = 1.49  # Time Resolution in seconds
    num_subjects: int = 4  # Number of subjects
    num_rois: int = 1000  # Number of Regions of Interest
    modality_fusion_transformer_num_layers: int = 1 
    modality_fusion_transformer_num_heads: int = 4 
    modality_fusion_transformer_dim: int = 1024  # d parameter of the transformer
    modality_fusion_transformer_fuse_mode: Literal["concat", "mean"] = "concat"
    modality_fusion_transformer_num_projection_layers: int = 1
    predictor_transformer_num_heads: int = 8
    predictor_transformer_num_layers: int = 3
    # Model Name to be passed into AutoCausalLLM in huggingface transformers
    text_extractor: str = "Qwen/Qwen2.5-14B"
    text_extractor_feature_size: int = 5120  # Dimensionality of the output of the LLM
    text_extractor_dtype: torch.dtype = field(default=torch.float16, compare=False, hash=False)  # FP16 for speed/VRAM,
    # Number of last hidden states to extract and average over.
    text_extractor_num_last_hidden_states: int = 4

    video_extractor: str = "facebook/vjepa2-vitg-fpc64-256"
    video_extractor_feature_size: int = 1408  # Dimensionality of the output of the video extractor
    video_extractor_pool_size: int = 2  # Number of spatial dimensions to pool over
    video_extractor_dtype: torch.dtype = field(default=torch.bfloat16, compare=False, hash=False)  # FP16 for speed/VRAM
    video_extractor_num_last_hidden_states: int = 3
    video_extractor_chunk_length_seconds: int = 16
    video_extractor_batch_size: int = field(default=8, compare=False, hash=False)

    audio_extractor_last_layer_index: int = 2
    audio_extractor_batch_size: int = field(default=32, compare=False, hash=False)
    audio_extractor_feature_size: int = 1536


def get_pretrained_model_configs(collection_slug: str) -> dict[VIBEConfig, str]:
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
            valid_keys = {f.name for f in fields(VIBEConfig)}
            filtered_dict = {k: v for k,
                             v in config_dict.items() if k in valid_keys}

            # Normalize any dtype fields coming from JSON (strings) into torch.dtype
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            for key, value in list(filtered_dict.items()):
                if key.endswith("_dtype"):
                    if isinstance(value, str):
                        normalized = value.strip().lower()
                        if normalized in dtype_map:
                            filtered_dict[key] = dtype_map[normalized]
                    # If it's already a torch.dtype or an unrecognized string,
                    # leave it as-is so downstream can handle/raise.

            # Create the config object
            config_obj = VIBEConfig(**filtered_dict)

            # Map the config object to the Hugging Face Repo ID
            valid_models[config_obj] = item.item_id

        except Exception as e:
            print(f"Skipping {item.item_id}: Could not load config ({e})")

    return valid_models
