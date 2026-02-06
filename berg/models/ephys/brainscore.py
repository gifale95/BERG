import os
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Union
import pkgutil

from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model, MODEL_REGISTRY

from brainscore_vision import load_model
from brainio.stimuli import StimulusSet
import brainscore_vision.models as bs_models

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


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
    
    MODEL_ID = "brainscore"
    
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
                - region: str (e.g., "V1", "V4", "IT") - REQUIRED
                - time_bins: list of tuples (e.g., [(100, 200)])
        """
        self.berg_dir = berg_dir
        self.model_id_full = model_id
        self.device = device
        
        # Parse BrainScore model name from model_id
        if model_id.startswith("brainscore-"):
            lowercase_name = model_id.replace("brainscore-", "")
        else:
            lowercase_name = model_id

        # Get original case for BrainScore API
        self.brainscore_model_name = get_original_case_model_name(lowercase_name)
        
        # Parse selection parameters - region is REQUIRED
        if not selection or 'region' not in selection:
            raise ValueError("BrainScore models require 'region' in selection parameter (e.g., selection={'region': 'V1'})")
        
        self.region = selection['region']
        self.time_bins = selection.get('time_bins', None)
        
        # Model will be loaded in load_model()
        self.model = None
        
    def load_model(self):
        """Load the BrainScore model."""
        print(f"Loading BrainScore model: {self.brainscore_model_name}")
        self.model = load_model(self.brainscore_model_name)
        
        # Configure recording if region specified
        if self.region:
            recording_target = getattr(self.model.RecordingTarget, self.region)
            
            # Use default time_bins if not provided
            if self.time_bins is None:
                self.time_bins = [(70, 170)]  # BrainScore default
            
            self.model.start_recording(
                recording_target=recording_target,
                time_bins=self.time_bins
            )
            
            print(f"Recording configured: region={self.region}, time_bins={self.time_bins}")
    
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
        
        # Extract only the layer(s) for the selected region
        layer_for_region = self.model.layer_model.region_layer_map[self.region]
        layers = [layer_for_region]
        
        if show_progress:
            print(f"Extracting activations from layer '{layer_for_region}' for {len(image_paths)} images...")
        
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

