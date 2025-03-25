import importlib
from typing import Dict, Type, Optional
import semver

# Maps model ID → {version: {"module_path": path, "class_name": name}}
MODEL_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {}


def register_model(model_id: str, version: str, module_path: str, class_name: Optional[str] = None, modality: str = None, yaml_path: str = None):
    """
    Register a model with the registry.
    
    Args:
        model_id: Unique identifier for the model
        version: Semantic version string (e.g., "1.0.0")
        module_path: Import path to the module containing the model
        class_name: Class name (defaults to model_id if not provided)
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
    Get a model class by ID and version.
    
    Args:
        model_id: Unique identifier for the model
        version: Specific version or "latest" to get the most recent
    
    Returns:
        The model class
    
    Raises:
        ValueError: If model or version not found
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
    """Get a list of all registered model IDs."""
    return list(MODEL_REGISTRY.keys())

def get_model_versions(model_id: str):
    """Get all available versions for a model ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    return list(MODEL_REGISTRY[model_id].keys())