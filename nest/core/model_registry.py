import importlib
from typing import Dict, Optional, Set, Any

# Maps model ID → info about the model
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(
    model_id: str, 
    module_path: str, 
    class_name: Optional[str] = None, 
    modality: str = None, 
    dataset: str = None,
    yaml_path: str = None
):
    """
    Register a model with a given ID and module path.
    
    Args:
        model_id (str): Unique identifier for the model.
        module_path (str): Dotted import path to the model module.
        class_name (Optional[str]): Name of the model class (defaults to model_id).
        modality (Optional[str]): Associated data modality (e.g., 'fmri', 'eeg').
        dataset (Optional[str]): Dataset on which the model was trained.
        yaml_path (Optional[str]): Path to the YAML metadata file.
    """
    MODEL_REGISTRY[model_id] = {
        "module_path": module_path,
        "class_name": class_name or model_id,
        "modality": modality,
        "dataset": dataset,
        "yaml_path": yaml_path
    }


def get_model_class(model_id: str):
    """
    Dynamically import and return a model class by ID.

    Args:
        model_id (str): Unique identifier for the model.

    Returns:
        Type: The model class.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    info = MODEL_REGISTRY[model_id]
    module = importlib.import_module(info["module_path"])
    return getattr(module, info["class_name"])

def get_available_models():
    """
    List all registered model IDs.

    Returns:
        list: A list of registered model IDs.
    """
    return list(MODEL_REGISTRY.keys())
