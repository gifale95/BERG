import importlib
from typing import Dict, Type, Optional
import semver

# Maps model ID → {version: {"module_path": path, "class_name": name}}
MODEL_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {}


def register_model(model_id: str, version: str, module_path: str, class_name: Optional[str] = None, modality: str = None, yaml_path: str = None):
    """
    Register a model with a given ID, version, and module path.

    Args:
        model_id (str): Unique identifier for the model.
        version (str): Semantic version string (e.g., "1.0.0").
        module_path (str): Dotted import path to the model module.
        class_name (Optional[str]): Name of the model class (defaults to model_id).
        modality (Optional[str]): Associated data modality (e.g., 'fmri', 'eeg').
        yaml_path (Optional[str]): Path to the YAML metadata file.
    """
    if model_id not in MODEL_REGISTRY:
        MODEL_REGISTRY[model_id] = {}
    
    MODEL_REGISTRY[model_id][version] = {
        "module_path": module_path,
        "class_name": class_name or model_id,
        "modality": modality,
        "yaml_path": yaml_path
    }

def get_model_class(model_id: str, version: str = "latest"):
    """
    Dynamically import and return a model class by ID and version.

    Args:
        model_id (str): Unique identifier for the model.
        version (str): Specific version or "latest" for most recent.

    Returns:
        Type: The model class.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    versions = MODEL_REGISTRY[model_id]
    if not versions:
        raise ValueError(f"No versions available for model '{model_id}'")
    
    if version == "latest":
        # Get the latest version using semantic versioning rules
        version = max(versions.keys(), key=lambda v: semver.VersionInfo.parse(v))
    
    if version not in versions:
        raise ValueError(f"Version '{version}' not found for model '{model_id}'")
    
    info = versions[version]
    module = importlib.import_module(info["module_path"])
    return getattr(module, info["class_name"])

def get_available_models():
    """
    List all registered model IDs.

    Returns:
        list: A list of registered model IDs.
    """
    return list(MODEL_REGISTRY.keys())

def get_model_versions(model_id: str):
    """
    Get all available versions for a registered model ID.

    Args:
        model_id (str): Unique identifier for the model.

    Returns:
        list: A list of version strings for the model.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    return list(MODEL_REGISTRY[model_id].keys())