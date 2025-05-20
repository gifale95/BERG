import importlib
from typing import Dict, Optional, Set, Any

# Maps model ID → info about the model
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(
    model_id: str, 
    module_path: str, 
    class_name: Optional[str] = None, 
    modality: str = None, 
    training_dataset: str = None,
    yaml_path: str = None
):
    """
    Register a model with a given ID and module path.
    
    Parameters
    ----------
    model_id : str
        Unique identifier for the model. Used as the key in the model registry.
    module_path : str
        Dotted import path to the model module (e.g., 'nest.models.fmri.nsd_fwrf').
    class_name : str, optional
        Name of the model class. If not provided, defaults to model_id.
    modality : str, optional
        Associated data modality (e.g., 'fmri', 'eeg').
    training_dataset : str, optional
        Dataset on which the model was trained (e.g., 'NSD', 'THINGS_EEG_2').
    yaml_path : str, optional
        Path to the YAML metadata file that describes the model.
    """
    MODEL_REGISTRY[model_id] = {
        "module_path": module_path,
        "class_name": class_name or model_id,
        "modality": modality,
        "training_dataset": training_dataset,
        "yaml_path": yaml_path
    }


def get_model_class(model_id: str):
    """
    Dynamically import and return a model class by ID.
    
    Parameters
    ----------
    model_id : str
        Unique identifier for the model to import.
    
    Returns
    -------
    Type
        The model class that can be instantiated to create a model object.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    info = MODEL_REGISTRY[model_id]
    module = importlib.import_module(info["module_path"])
    return getattr(module, info["class_name"])

def get_available_models():
    """
    List all registered model IDs.
    
    Returns
    -------
    list
        A list of registered model IDs that can be used with get_model_class()
        or passed to NEST.get_encoding_model().
    """
    return list(MODEL_REGISTRY.keys())
