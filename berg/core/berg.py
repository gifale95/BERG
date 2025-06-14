from typing import Any, Dict, List
import numpy as np
from berg.core.exceptions import ModelNotFoundError, InvalidParameterError
from berg.core.model_registry import (
    MODEL_REGISTRY,
    get_available_models,
    get_model_class
)
from berg.interfaces.base_model import BaseModelInterface


class BERG:
    def __init__(self, berg_dir: str):
        """
        Initialize the BERG toolkit.
        
        Parameters
        ----------
        berg_dir : str
            Path to the BERG directory containing model files and weights.
            This directory should contain the organized structure of encoding
            models by modality and dataset.
        """
        self.berg_dir = berg_dir
        
    def get_model_catalog(self, print_format: bool = False) -> Dict[str, List[str]]:
        """
        Get a catalog of available models organized by modality and training dataset.
        
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
            training_dataset = info.get("training_dataset")
            
            # Handle missing modality
            if not modality:
                parts = model_id.split('_')
                modality = parts[0] if parts else "unknown"
                
            # Handle missing dataset
            if not training_dataset:
                parts = model_id.split('_')
                training_dataset = parts[1] if len(parts) > 1 else "unknown"
            
            # Add to catalog
            if modality not in catalog:
                catalog[modality] = set()
            
            catalog[modality].add(training_dataset)
        
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
        List all registered models in the BERG registry.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary containing information about all registered models,
            including their IDs and associated model_info.
        """
        return get_available_models()
        

    def get_encoding_model(self, model_id: str, device: str = "auto", selection: dict = None, **kwargs):
        """
        Load and return a specific encoding model instance.

        Parameters
        ----------
        model_id : str
            Unique identifier of the model to load.
        device : str, default="auto"
            Target device for computation ("cpu", "cuda", or "auto").
            If "auto", the system will use GPU acceleration if available.
        selection : dict, optional
            Optional selection dictionary to specify which parts of the model output to include.
            Keys may include:
            - "roi": str, e.g. "V1", "FFA-2", etc.
            - "channels": list of EEG channel names to include
            - "timepoints": binary one-hot encoded vector (length must match number of timepoints)
            Refer to the model's YAML config for valid options.
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
            model = model_class(berg_dir=self.berg_dir, device=device, selection=selection, **kwargs)
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
        # Generate responses
        responses = model.generate_response(stimulus, **kwargs)
        
        if return_metadata:
            try:
                # Get the model's class and call get_metadata with the model instance
                model_class = model.__class__
                metadata = model_class.get_metadata(model_instance=model)
                return responses, metadata
            except Exception as e:
                # If that fails, provide a warning and return responses only
                print(f"Warning: Could not retrieve metadata ({str(e)}). Returning responses only.")
                return responses
        else:
            return responses
        
    def get_model_metadata(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for a model with specific parameters without loading the model.
        
        Parameters
        ----------
        model_id : str
            Unique identifier of the model.
        **kwargs
            Parameters needed for metadata retrieval (e.g., subject, roi)
            These parameters depend on the specific model and are documented
            in the model's YAML configuration.
        
        Returns
        -------
        Dict[str, Any]
            Model metadata dictionary.
        """
        if model_id not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
            
        try:
            model_class = get_model_class(model_id)
            return model_class.get_metadata(berg_dir=self.berg_dir, **kwargs)
        except Exception as e:
            yaml_path = MODEL_REGISTRY[model_id]["yaml_path"]
            error_msg = f"Error retrieving metadata: {str(e)}\n"
            error_msg += f"Check the model's YAML file at {yaml_path} for correct parameters."
            raise InvalidParameterError(error_msg)
        
    
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