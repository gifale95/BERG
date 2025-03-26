from typing import Dict, Any, List, Optional, Union
import numpy as np
import os

from new_nest.core.model_registry import get_model_class, get_available_models, get_model_versions
from new_nest.core.exceptions import ModelNotFoundError
from new_nest.interfaces.base_model import BaseModelInterface
from new_nest.core.model_registry import MODEL_REGISTRY


class NEST:
    def __init__(self, nest_dir: str):
        """
        Initialize NEST.
        
        Args:
            nest_dir: Path to the NEST directory containing models
        """
        self.nest_dir = nest_dir
        
    def which_modalities(self) -> List[str]:
        """
        Return available neural data modalities.
        
        Returns:
            List of modality names
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
        Create and return an instance of the specified model.
        
        Args:
            model_id: Unique identifier for the model
            version: Specific version or "latest" for the most recent
            **kwargs: Model-specific parameters
        
        Returns:
            Instantiated model
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
        Generate in silico neural responses.
        
        Args:
            model: Model instance
            stimulus: Input stimulus array
            return_metadata: If it should also return model metadata
            **kwargs: Additional parameters for response generation
        
        Returns:
            Neural responses
        """
        
        if return_metadata:
            return model.generate_response(stimulus, **kwargs), model.get_metadata()
        else:
            return model.generate_response(stimulus, **kwargs)
    
    def get_model_info(self, model_id: str, version: str = "latest") -> Dict[str, Any]:
        """
        Get information about a model based on its YAML metadata.

        Args:
            model_id: Unique identifier for the model
            version: Reserved for future use (currently ignored)

        Returns:
            Dict containing model metadata and usage example
        """
        if model_id not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        try:
            return BaseModelInterface.describe(model_id)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model description for '{model_id}': {str(e)}")
        
        
    def list_models(self) -> Dict[str, List[str]]:
        """
        List all available models and their versions.
        
        Returns:
            Dict mapping model IDs to lists of available versions
        """
        result = {}
        for model_id in get_available_models():
            result[model_id] = get_model_versions(model_id)
        return result