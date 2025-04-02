from typing import Any, Dict, List
import numpy as np
from nest.core.exceptions import ModelNotFoundError
from nest.core.model_registry import (
    MODEL_REGISTRY,
    get_available_models,
    get_model_class
)
from nest.interfaces.base_model import BaseModelInterface


class NEST:
    def __init__(self, nest_dir: str):
        """
        Initialize the NEST toolkit.

        Args:
            nest_dir (str): Path to the NEST directory containing model files.
        """
        self.nest_dir = nest_dir
        
    def get_model_catalog(self, print_format: bool = False) -> Dict[str, List[str]]:
        """
        Get a catalog of available models organized by modality and dataset.
        
        Args:
            print_format (bool): If True, print a formatted catalog.
            
        Returns:
            Dict[str, List[str]]: Mapping from modalities to lists of datasets.
        """
        
        # Organize models by modality and dataset
        catalog = {}
        
        for model_id, info in MODEL_REGISTRY.items():
            modality = info.get("modality")
            dataset = info.get("dataset")
            
            # Handle missing modality
            if not modality:
                parts = model_id.split('_')
                modality = parts[0] if parts else "unknown"
                
            # Handle missing dataset
            if not dataset:
                parts = model_id.split('_')
                dataset = parts[1] if len(parts) > 1 else "unknown"
            
            # Add to catalog
            if modality not in catalog:
                catalog[modality] = set()
            
            catalog[modality].add(dataset)
        
        # Convert sets to sorted lists for more predictable output
        formatted_catalog = {modality: sorted(datasets) for modality, datasets in catalog.items()}
        
        # Print formatted catalog if requested
        if print_format:
            print("Available Modalities and Datasets:")
            print("=================================")
            
            for modality, datasets in sorted(formatted_catalog.items()):
                print(f"• {modality.upper()}")
                for dataset in datasets:
                    print(f"  └─ {dataset}")
                print()
        
        return formatted_catalog
        
    def list_models(self) -> Dict[str, List[str]]:
        """
        List all registered models.

        Returns:
            Dict[str, List[str]]: Mapping of model IDs.
        """
        return get_available_models()
        
    
    def get_encoding_model(self, model_id: str, device:str="auto", **kwargs):
        """
        Load and return a specific encoding model instance.

        Args:
            model_id (str): Unique identifier of the model.
            **kwargs: Additional model-specific initialization parameters.

        Returns:
            BaseModelInterface: Instantiated encoding model.
        """
        try:
            model_class = get_model_class(model_id)
            model = model_class(nest_dir=self.nest_dir, device=device, **kwargs)
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
    
    def describe(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata and usage info for a specified model.

        Args:
            model_id (str): Unique identifier of the model.

        Returns:
            Dict[str, Any]: Model metadata and usage example.
        """
        if model_id not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        try:
            return BaseModelInterface.describe_from_id(model_id)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model description for '{model_id}': {str(e)}")