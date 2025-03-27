from typing import Dict, Any, List, Optional, Union
import numpy as np
import os

from nest.core.model_registry import get_model_class, get_available_models, get_model_versions
from nest.core.exceptions import ModelNotFoundError
from nest.interfaces.base_model import BaseModelInterface
from nest.core.model_registry import MODEL_REGISTRY


class NEST:
    def __init__(self, nest_dir: str):
        """
        Initialize the NEST toolkit.

        Args:
            nest_dir (str): Path to the NEST directory containing model files.
        """
        self.nest_dir = nest_dir
        
    def which_modalities(self) -> List[str]:
        """
        Get the list of available neural data modalities.

        Returns:
            List[str]: Names of supported modalities (e.g., 'fmri', 'eeg').
        """
        # Derive from model registry metadata
        modalities = set()
        
        for model_id in get_available_models():
            # Get modalities from each version of the model
            versions = MODEL_REGISTRY[model_id]
            for version_info in versions.values():
                if "modality" in version_info and version_info["modality"]:
                    modalities.add(version_info["modality"])
                else:
                    # Fallback: try to extract from model_id
                    parts = model_id.split('_')
                    if parts:
                        modalities.add(parts[0])
        
        return list(modalities)
        
    
    def get_encoding_model(self, model_id: str, version: str = "latest", **kwargs):
        """
        Load and return a specific encoding model instance.

        Args:
            model_id (str): Unique identifier of the model.
            version (str, optional): Model version to load, or 'latest'. Defaults to "latest".
            **kwargs: Additional model-specific initialization parameters.

        Returns:
            BaseModelInterface: Instantiated encoding model.
        """
        try:
            model_class = get_model_class(model_id, version)
            model = model_class(**kwargs)
            model.load_model()
            return model
        except ValueError as e:
            raise ModelNotFoundError(str(e))
        except Exception as e:
            raise
    
    def encode(self, model: BaseModelInterface, stimulus: np.ndarray, return_metadata: bool = False, **kwargs):
        """
        Generate in silico neural responses using the given model.

        Args:
            model (BaseModelInterface): An instantiated model.
            stimulus (np.ndarray): Input stimulus array.
            return_metadata (bool, optional): Whether to return metadata. Defaults to False.
            **kwargs: Additional arguments for response generation.

        Returns:
            Simulated neural responses, optionally with model metadata.
        """
        
        if return_metadata:
            return model.generate_response(stimulus, **kwargs), model.get_metadata()
        else:
            return model.generate_response(stimulus, **kwargs)
    
    def get_model_info(self, model_id: str, version: str = "latest") -> Dict[str, Any]:
        """
        Retrieve metadata and usage info for a specified model.

        Args:
            model_id (str): Unique identifier of the model.
            version (str, optional): Reserved for future use. Defaults to "latest".

        Returns:
            Dict[str, Any]: Model metadata and usage example.
        """
        if model_id not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        try:
            return BaseModelInterface.describe(model_id)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model description for '{model_id}': {str(e)}")
        
        
    def list_models(self) -> Dict[str, List[str]]:
        """
        List all registered models with their available versions.

        Returns:
            Dict[str, List[str]]: Mapping of model IDs to version lists.
        """
        result = {}
        for model_id in get_available_models():
            result[model_id] = get_model_versions(model_id)
        return result