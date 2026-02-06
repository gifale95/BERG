import os
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Union
import pkgutil

from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model, MODEL_REGISTRY

from berg.core.parameter_validator import validate_selection_keys
from berg.core.exceptions import InvalidParameterError

from brainscore_vision import load_model
from brainio.stimuli import StimulusSet
import brainscore_vision.models as bs_models

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# Load model info from YAML
def load_model_info():
    yaml_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "model_cards", 
        "brainscore.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register the gateway model
register_model(
    model_id="brainscore",
    module_path="berg.models.ephys.brainscore",
    class_name="BrainScoreGateway",
    modality="ephys",
    training_dataset="BrainScore",
    yaml_path="berg/models/model_cards/brainscore.yaml"
)


class BrainScoreGateway(BaseModelInterface):
    """
    Gateway to BrainScore vision models.
    
    This class provides a unified interface to access BrainScore's collection
    of neural network models trained on biological neural recordings.
    """
    
    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    
    def __init__(
        self,
        berg_dir: str,
        model_id: str,
        device: str = "auto",
        selection: dict = None
    ):
        """
        Initialize BrainScore gateway.
        
        Parameters
        ----------
        berg_dir : str
            Path to BERG directory
        model_id : str
            Model identifier in format "brainscore-{model_name}"
        device : str
            Device for computation ("cpu", "cuda", or "auto")
        selection : dict, optional
            Selection parameters:
                - roi: str (e.g., "V1", "V4", "IT") - if None, all regions returned
                - time_bins: list of tuples (e.g., [(100, 200)])
        """
        self.berg_dir = berg_dir
        self.model_id_full = model_id
        self.device = device
        self.selection = selection
        
        # Validate parameters
        self._validate_parameters()
        
        # Parse BrainScore model name from model_id 
        if model_id.startswith("brainscore-"):
            lowercase_name = model_id.replace("brainscore-", "")
        else:
            lowercase_name = model_id
        
        # Get original case for BrainScore API
        self.brainscore_model_name = get_original_case_model_name(lowercase_name)
        
        # Parse selection parameters
        self.roi = None
        self.time_bins = None
        if selection:
            self.roi = selection.get('roi', None)
            self.time_bins = selection.get('time_bins', None)
        
        self.model = None
        
        
    def _validate_parameters(self):
            """
            Validate user-provided parameters.
            Ensures that region and time_bins match expected values.
            """
            
            if self.selection is not None:
                # Validate selection keys
                validate_selection_keys(self.selection, self.SELECTION_KEYS)
                
                # Validate region
                if "roi" in self.selection:
                    roi = self.selection["roi"]
                    
                    if not isinstance(roi, str):
                        raise InvalidParameterError("roi must be provided as a string")
                    
                    if roi not in self.VALID_ROIS:
                        raise InvalidParameterError(
                            f"Invalid roi: '{roi}'. "
                            f"Valid rois are: {self.VALID_ROIS}"
                        )
                
                # Validate time_bins
                if "time_bins" in self.selection:
                    time_bins = self.selection["time_bins"]
                    
                    if not isinstance(time_bins, list):
                        raise InvalidParameterError("time_bins must be provided as a list of tuples")
                    
                    for tb in time_bins:
                        if not isinstance(tb, (list, tuple)) or len(tb) != 2:
                            raise InvalidParameterError(
                                f"Each time bin must be a tuple of (start_ms, end_ms). Got: {tb}"
                            )
                        
                        start, end = tb
                        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                            raise InvalidParameterError(
                                f"Time bin values must be numeric. Got: {tb}"
                            )
                        
                        if start >= end:
                            raise InvalidParameterError(
                                f"Time bin start must be less than end. Got: {tb}"
                            )
                        
                        # TODO: ONCE WE KNOW MIN AND MAX ALSO ADD THIS
        
    def load_model(self):
        """Load the BrainScore model."""
        
        print(f"Loading BrainScore model: {self.brainscore_model_name}")
        self.model = load_model(self.brainscore_model_name)
        
        # Configure recording only if region specified
        if self.roi:
            recording_target = getattr(self.model.RecordingTarget, self.roi)
            
            # Use default time_bins if not provided
            if self.time_bins is None:
                self.time_bins = [(70, 170)]  # TODO: WHAT IS THE DEFAULT?
            
            self.model.start_recording(
                recording_target=recording_target,
                time_bins=self.time_bins
            )
            
            print(f"Recording configured: region={self.roi}, time_bins={self.time_bins}")
        else:
            print("No region selected - will extract all available regions")
    

    def generate_response(
        self,
        stimulus: Union[str, List[str]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate neural responses for given stimuli.
        
        Parameters
        ----------
        stimulus : str or list[str]
            Path to image directory or list of image file paths
        show_progress : bool
            Whether to show progress bar
        
        Returns
        -------
        np.ndarray
            Neural responses, shape (n_images, n_units)
        """
        # Get image paths
        if isinstance(stimulus, str):
            image_paths = sorted(Path(stimulus).glob("*.jpg")) + \
                        sorted(Path(stimulus).glob("*.png")) + \
                        sorted(Path(stimulus).glob("*.jpeg"))
        else:
            image_paths = [Path(p) for p in stimulus]
        
        # Create StimulusSet
        stimulus_ids = [p.stem for p in image_paths]
        stimuli = StimulusSet(pd.DataFrame({
            "stimulus_id": stimulus_ids,
            "filename": [p.name for p in image_paths],
        }))
        stimuli.stimulus_paths = {
            sid: str(path.absolute()) 
            for sid, path in zip(stimulus_ids, image_paths)
        }
        
        # Extract layers
        if self.roi:
            # Extract only the layer for selected region
            layer_for_region = self.model.layer_model.region_layer_map[self.roi]
            layers = [layer_for_region]
            
            if show_progress:
                print(f"Extracting activations from layer '{layer_for_region}' for {len(image_paths)} images...")
        else:
            # Extract all unique layers for all regions
            layers = list(dict.fromkeys(self.model.layer_model.region_layer_map.values()))
            
            if show_progress:
                print(f"Extracting activations from {len(layers)} layers (all regions) for {len(image_paths)} images...")
        
        activations = self.model.activations_model._extractor.from_stimulus_set(
            stimulus_set=stimuli,
            layers=layers
        )
        
        return activations.values
    
    @classmethod
    def get_model_id(cls) -> str:
        """Return the model's unique identifier."""
        return cls.MODEL_ID
    
    @classmethod
    def get_metadata(
        cls,
        berg_dir: str = None,
        model_instance: 'BrainScoreGateway' = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get model metadata.
        
        BrainScore models provide minimal metadata.
        """
        if model_instance:
            return {
                "brainscore_model": model_instance.brainscore_model_name,
                "region": model_instance.region,
                "time_bins": model_instance.time_bins,
            }
        return {}
    
    def cleanup(self):
        """Release resources."""
        self.model = None


# Discovery function
def discover_brainscore_models() -> List[str]:
    """
    Discover all available BrainScore vision models.
    
    Returns lowercase model names for display, and maintains a mapping
    to original case for API calls.
    """
    models = []
    for importer, modname, ispkg in pkgutil.iter_modules(bs_models.__path__):
        if not modname.startswith('_'):
            models.append(modname.lower())
    
    return sorted(models)



def get_original_case_model_name(lowercase_name: str) -> str:
    """Map lowercase model name back to original case for BrainScore API."""
    for importer, modname, ispkg in pkgutil.iter_modules(bs_models.__path__):
        if modname.lower() == lowercase_name:
            return modname
    
    raise ValueError(f"BrainScore model '{lowercase_name}' not found")

