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
        
        Parameters
        ----------
        nest_dir : str
            Path to the NEST directory containing model files and weights.
            This directory should contain the organized structure of encoding
            models by modality and dataset.
        """
        self.nest_dir = nest_dir
        
    def get_model_catalog(self, print_format: bool = False) -> Dict[str, List[str]]:
        """
        Get a catalog of available models organized by modality and dataset.
        
        Parameters
        ----------
        print_format : bool, default=False
            If True, print a formatted hierarchical catalog to the console.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping modalities (e.g., 'fmri', 'eeg') to lists of 
            available datasets for each modality.
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
        List all registered models in the NEST registry.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary containing information about all registered models,
            including their IDs and associated model_info.
        """
        return get_available_models()
        
    
    def get_encoding_model(self, model_id: str, device:str="auto", **kwargs):
        """
        Load and return a specific encoding model instance.
        
        Parameters
        ----------
        model_id : str
            Unique identifier of the model to load.
        device : str, default="auto"
            Target device for computation ("cpu", "cuda", or "auto").
            If "auto", the system will use GPU acceleration if available.
        **kwargs
            Additional model-specific initialization parameters.
            These vary by model and are documented in each model's
            YAML configuration file.
        
        Returns
        -------
        BaseModelInterface
            Instantiated and loaded encoding model ready for generating
            neural responses.
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
        
        Parameters
        ----------
        model : BaseModelInterface
            An instantiated and loaded encoding model.
        stimulus : np.ndarray
            Input stimulus array. Typically has shape (batch_size, channels, height, width)
            for image stimuli, but exact requirements vary by model.
        return_metadata : bool, default=False
            Whether to return model metadata along with the responses.
        **kwargs
            Additional arguments for response generation that are specific
            to the model being used.
        
        Returns
        -------
        np.ndarray or tuple
            If return_metadata is False:
                Simulated neural responses only.
            If return_metadata is True:
                A tuple of (responses, metadata), where responses is the simulated
                neural activity and metadata is a dictionary of model-specific
                information.
        """
        
        if return_metadata:
            return model.generate_response(stimulus, **kwargs), model.get_metadata()
        else:
            return model.generate_response(stimulus, **kwargs)
    
    def describe(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieve model info and usage information for a specified model.
        
        Parameters
        ----------
        model_id : str
            Unique identifier of the model to describe.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive model information.
        """
        if model_id not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        try:
            return BaseModelInterface.describe_from_id(model_id)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model description for '{model_id}': {str(e)}")