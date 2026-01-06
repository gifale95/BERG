import os
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model
from berg.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from berg.core.parameter_validator import (
    validate_subjects,
    validate_selection_keys,
    validate_roi,
    validate_binary_array,
)

# Import fnn library
try:
    from fnn import microns
except ImportError:
    raise ImportError(
        "fnn is required for this model. Please install it with: "
        "pip install git+https://github.com/cajal/fnn.git"
    )


# Valid fields for each session/scan combination
VALID_FIELDS = {
    "session4_scan7": [1, 2, 3, 4, 5, 6, 7, 8],
    "session5_scan6": [1, 2, 3, 4, 5, 6, 7, 8],
    "session5_scan7": [1, 2, 3, 4, 5, 6, 7, 8],
    "session6_scan2": [1, 2, 3, 4, 5, 6, 7, 8],
    "session6_scan4": [1, 2, 3, 4, 5, 6, 7, 8],
    "session6_scan6": [1, 2, 3, 4, 5, 6, 7, 8],
    "session6_scan7": [1, 2, 3, 4, 5, 6, 7, 8],
    "session7_scan3": [1, 2, 3, 4, 5, 6, 7, 8],
    "session7_scan5": [1, 2, 3, 4, 5, 6, 7, 8],
    "session8_scan5": [1, 2, 3, 4, 5, 6, 7, 8],
    "session9_scan3": [1, 2, 3, 4, 5, 6],
    "session9_scan4": [1, 2, 3, 4, 5, 6],
    "session9_scan6": [1, 2, 3, 4],
}


def load_model_info():
    """Load model information from YAML file."""
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "calcium_2p-natural_videos-3DCNN.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


model_info = load_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.calcium_2p.calcium_2p_natural_videos_3dcnn",
    class_name="CalciumEncodingModel",
    modality=model_info.get("modality", "calcium_2p"),
    training_dataset=model_info.get("training_dataset", "natural_videos"),
    yaml_path=os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "calcium_2p-natural_videos-3DCNN.yaml"
    )
)


class CalciumEncodingModel(BaseModelInterface):
    """
    Two-photon calcium imaging encoding model using 3D CNN architecture
    for mouse visual cortex. The model code can be found here: https://github.com/cajal/fnn.
    """
    
    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    VALID_FIELDS = model_info["parameters"]["selection"]["properties"]["field"]["valid_values"]
    
    def __init__(
        self,
        subject: str,
        selection: Optional[Dict] = None,
        device: str = "auto",
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the mouse calcium imaging encoding model.
        
        Parameters
        ----------
        subject : str
            Session and scan identifier in format "sessionX_scanY" (e.g., "session8_scan5").
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            - roi: List of brain area labels (e.g., ['V1', 'LM'])
            - field: List of imaging field numbers (e.g., [1, 2, 3])
            - unit_index: Binary one-hot encoded vector indicating neurons to include
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
        berg_dir : str, optional
            Path to the BERG directory containing model weights and metadata files.
        """
        self.subject_input = subject
        self.berg_dir = berg_dir
        self.model = None
        self.unit_ids = None
        self.metadata = None
        
        # Parameters from selection
        self.selection = selection
        self.roi_list = None
        self.field_list = None
        self.neuron_mask = None
        self.selected_indices = None
        
        # Validate parameters
        self._validate_parameters()
        
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def _validate_parameters(self):
        """Validate the subject and selection parameters."""
        # Validate subject
        self.subjects = validate_subjects(self.subject_input, self.VALID_SUBJECTS)
        if len(self.subjects) != 1:
            raise InvalidParameterError(
                f"Only single subject supported, got {len(self.subjects)} subjects"
            )
        self.subject = self.subjects[0]
        
        # Parse session and scan from subject string
        self.session, self.scan = self._parse_subject_string(self.subject)
        
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)
            
            # Validate ROI selection
            if "roi" in self.selection:
                roi_list = self.selection["roi"]
                if not isinstance(roi_list, list):
                    raise InvalidParameterError(
                        f"Parameter 'roi' must be a list, got {type(roi_list)}"
                    )
                for roi in roi_list:
                    validate_roi(roi, self.VALID_ROIS)
                self.roi_list = roi_list
            
            # Validate field selection
            if "field" in self.selection:
                field_list = self.selection["field"]
                if not isinstance(field_list, list):
                    raise InvalidParameterError(
                        f"Parameter 'field' must be a list, got {type(field_list)}"
                    )
                
                # Check that all fields exist for this session/scan
                valid_fields_for_subject = VALID_FIELDS[self.subject]
                unavailable_fields = [f for f in field_list if f not in valid_fields_for_subject]
                if unavailable_fields:
                    raise InvalidParameterError(
                        f"Field(s) {unavailable_fields} not available for {self.subject}. "
                        f"Available fields: {valid_fields_for_subject}"
                    )
                
                self.field_list = field_list
            
            # Validate neuron selection
            if "unit_index" in self.selection:
                neuron_mask = self.selection["unit_index"]
                # Will validate length after loading metadata
                self.neuron_mask = neuron_mask
    
    def _parse_subject_string(self, subject_str: str) -> tuple:
        """
        Parse a subject string like "session8_scan5" into session and scan numbers.
        
        Parameters
        ----------
        subject_str : str
            Subject identifier in format "sessionX_scanY"
            
        Returns
        -------
        tuple
            (session, scan) as integers
        """
        parts = subject_str.split('_')
        if len(parts) != 2 or not parts[0].startswith('session') or not parts[1].startswith('scan'):
            raise InvalidParameterError(
                f"Subject string must be in format 'sessionX_scanY', got '{subject_str}'"
            )
        
        try:
            session = int(parts[0].replace('session', ''))
            scan = int(parts[1].replace('scan', ''))
        except ValueError:
            raise InvalidParameterError(
                f"Could not parse session/scan numbers from '{subject_str}'"
            )
        
        return session, scan
    
    def load_model(self, device: str = "auto") -> None:
        """
        Load the pre-trained model weights and initialize the inference engine.
        
        Parameters
        ----------
        device : str, default="auto"
            Device to load the model on.
        """
        
        # Construct path to model weights
        weights_dir = os.path.join(
            self.berg_dir,
            'encoding_models',
            'modality-calcium_2p',
            'train_dataset-natural_videos',
            'model-3DCNN',
            'encoding_models_weights'
        )
        
        if not os.path.exists(weights_dir):
            raise ModelLoadError(f"Model weights directory not found: {weights_dir}")
        
        # Load model using fnn library
        self.model, self.unit_ids = microns.scan(
            session=self.session,
            scan_idx=self.scan,
            directory=weights_dir)

        # Move model to device if possible
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        
        # Load metadata
        self._load_metadata()
        
        # Now validate neuron mask length if provided
        if self.neuron_mask is not None:
            n_neurons = len(self.unit_ids)
            validated_mask = validate_binary_array(
                self.neuron_mask,
                expected_length=n_neurons,
                parameter_name="unit_index"
            )
            self.neuron_mask = validated_mask.astype(bool)
        
        # Compute selected indices based on selection criteria
        self._compute_selected_indices()
    
    def _load_metadata(self):
        """Load metadata for the current session/scan."""
        metadata_path = os.path.join(
            self.berg_dir,
            'encoding_models',
            'modality-calcium_2p',
            'train_dataset-natural_videos',
            'model-3DCNN',
            'metadata',
            f'session{self.session}_scan{self.scan}_metadata.npy'
        )
        
        if not os.path.exists(metadata_path):
            raise ModelLoadError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = np.load(metadata_path, allow_pickle=True).item()
    
    def _compute_selected_indices(self):
        """Compute indices of neurons to include based on selection criteria."""
        n_neurons = len(self.unit_ids)
        selected_mask = np.zeros(n_neurons, dtype=bool)
        
        # Apply ROI filter
        if self.roi_list is not None:
            roi_mask = np.zeros(n_neurons, dtype=bool)
            for roi in self.roi_list:
                roi_mask |= self.metadata['calcium_2p']['roi'][roi].astype(bool)
            selected_mask |= roi_mask
        
        # Apply field filter
        if self.field_list is not None:
            field_mask = np.zeros(n_neurons, dtype=bool)
            for field in self.field_list:
                field_key = f'field_{field}'
                field_mask |= self.metadata['calcium_2p']['field_masks'][field_key].astype(bool)
            selected_mask |= field_mask
        
        # Apply neuron mask filter
        if self.neuron_mask is not None:
            selected_mask |= self.neuron_mask
        
        # If no selection criteria provided, select all neurons
        if self.roi_list is None and self.field_list is None and self.neuron_mask is None:
            selected_mask = np.ones(n_neurons, dtype=bool)
        
        # Get indices
        self.selected_indices = np.where(selected_mask)[0]
        
        if len(self.selected_indices) == 0:
            raise InvalidParameterError(
                "Selection criteria resulted in zero neurons. Please adjust your selection."
            )
    
    def generate_response(
        self,
        stimulus: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate in silico calcium imaging responses for a video stimulus.
        
        Parameters
        ----------
        stimulus : np.ndarray
            Grayscale video frames. Shape: (n_frames, 144, 256) with integer
            values in range [0, 255].
        show_progress : bool, default=True
            Whether to show a progress bar during encoding.
        
        Returns
        -------
        np.ndarray
            Predicted neural responses with shape (n_frames, n_selected_neurons)
            and dtype float32.
        """
        # Validate stimulus
        if not isinstance(stimulus, np.ndarray):
            raise StimulusError("Stimulus must be a numpy array")
        
        if len(stimulus.shape) != 3:
            raise StimulusError(
                f"Stimulus must be 3D (n_frames, height, width), got shape {stimulus.shape}"
            )
        
        if stimulus.shape[1] != 144 or stimulus.shape[2] != 256:
            raise StimulusError(
                f"Stimulus frames must be 144x256 pixels, got {stimulus.shape[1]}x{stimulus.shape[2]}"
            )
        
        # Ensure uint8 dtype
        if stimulus.dtype != np.uint8:
            stimulus = stimulus.astype(np.uint8)
        
        # Run inference
        try:
            responses = self.model.predict(stimuli=stimulus)
        except Exception as e:
            raise ModelLoadError(f"Model prediction failed: {str(e)}")
        
        # Convert to numpy if needed
        if isinstance(responses, torch.Tensor):
            responses = responses.cpu().numpy()
        
        # Convert to float32
        responses = responses.astype(np.float32)
        
        # Apply neuron selection
        if self.selected_indices is not None:
            responses = responses[:, self.selected_indices]
        
        return responses
    
    @classmethod
    def get_metadata(
        cls,
        berg_dir: Optional[str] = None,
        subject: Optional[str] = None,
        model_instance: Optional[BaseModelInterface] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        berg_dir : str, optional
            Path to BERG directory containing metadata.
        subject : str, optional
            Subject identifier (e.g., "session8_scan5").
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        
        Returns
        -------
        dict
            Metadata dictionary containing calcium imaging and encoding model information.
        """
        # Extract parameters from model instance if provided
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
            subject = model_instance.subject
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir
            subject = cls.subject
        
        # Validate required parameters
        missing_params = []
        if berg_dir is None:
            missing_params.append('berg_dir')
        if subject is None:
            missing_params.append('subject')
        
        if missing_params:
            raise InvalidParameterError(
                f"Required parameters missing: {', '.join(missing_params)}"
            )
        
        # Validate subject
        if isinstance(subject, str):
            subjects = validate_subjects(subject, cls.VALID_SUBJECTS)
            if len(subjects) != 1:
                raise InvalidParameterError(
                    f"Only single subject supported for get_metadata, got {len(subjects)}"
                )
            subject = subjects[0]
        
        # Parse session and scan
        parts = subject.split('_')
        session = int(parts[0].replace('session', ''))
        scan = int(parts[1].replace('scan', ''))
        
        # Build metadata path
        metadata_path = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-calcium_2p',
            'train_dataset-natural_videos',
            'model-3DCNN',
            'metadata',
            f'session{session}_scan{scan}_metadata.npy'
        )
        
        # Load metadata
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique string identifier.
        
        Returns
        -------
        str
            Model ID string.
        """
        return cls.MODEL_ID
    
    def cleanup(self) -> None:
        """
        Release memory and resources associated with the model.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Move to CPU if possible
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            
            # Clear references
            self.model = None
            self.unit_ids = None
            self.metadata = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()