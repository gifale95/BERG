import os
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional
from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model
from berg.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from berg.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_roi,
    validate_binary_array,
)

# Import MOSAIC
try:
    import mosaic
    from mosaic.utils.inference import MosaicInference
    from PIL import Image
except ImportError:
    raise ImportError(
        "MOSAIC is required for this model. Please install it with: pip install mosaic"
    )


# Load model_info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-mosaic_nsd-cnn8_nsd_all.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.mosaic_nsd",
    class_name="FMRIEncodingModel",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "mosaic_nsd"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-mosaic_nsd-cnn8_nsd_all.yaml")
)


class FMRIEncodingModel(BaseModelInterface):
    """
    fMRI encoding model using MOSAIC CNN8 architecture
    for Natural Scenes Dataset (NSD) subjects.
    """
    
    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    N_VERTICES = 57051  # Total cortical vertices
    
    def __init__(
        self, 
        subject: int, 
        selection: Optional[Dict] = None,
        device: str = "auto", 
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the MOSAIC fMRI encoding model for a specific subject.
        
        Parameters
        ----------
        subject : int
            Subject number from the NSD dataset (1-8).
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            - roi: List of region labels (e.g., ['L_V1', 'R_V1'])
            - vertices: Binary one-hot encoded vector (57051,) indicating vertices to include
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
        berg_dir : str, optional
            Path to the BERG directory containing metadata files.
        """
        self.subject = subject
        self.berg_dir = berg_dir
        self.model = None
        self.inference = None
        
        # Parameters from selection
        self.selection = selection
        self.roi_list = None
        self.vertex_index = None
        self.vertex_mask = None  # Combined mask for output selection
        
        # Validate Parameters
        self._validate_parameters()
        
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def _validate_parameters(self):
        """
        Validate the subject and selection values against the model info.
        
        Verifies that the provided subject ID, ROI names, and vertex indices
        are among the supported values defined in the model's YAML.
        """
        # Validate subject
        validate_subject(self.subject, self.VALID_SUBJECTS)
        
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)
            
            # Individual validations
            if "roi" in self.selection:
                # Validate each ROI in the list
                roi_list = self.selection["roi"]
                if not isinstance(roi_list, list):
                    raise InvalidParameterError(
                        f"Parameter 'roi' must be a list, got {type(roi_list)}"
                    )
                for roi in roi_list:
                    validate_roi(roi, self.VALID_ROIS)
                self.roi_list = roi_list
            
            if "voxel_index" in self.selection:
                vertex_index = self.selection["voxel_index"]
                # Use the standard binary array validator
                validated_array = validate_binary_array(
                    vertex_index,
                    expected_length=self.N_VERTICES,
                    parameter_name="voxel_index"
                )
                self.vertex_index = validated_array.astype(bool)
                print(self.vertex_index)
        
    def load_model(self, device: str = "auto") -> None:
        """
        Load MOSAIC model weights and prepare for inference.
        
        Parameters
        ----------
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
        """
        try:
            # Load the MOSAIC model
            self.model = mosaic.from_pretrained(
                backbone_name="CNN8",
                framework="multihead",
                subjects="NSD",
                vertices="all",
                folder="./mosaic_models"
            )
            
            # Initialize inference wrapper
            self.inference = MosaicInference(
                model=self.model,
                batch_size=32,
                device=self.device
            )
            
            # Prepare vertex selection mask if ROI or vertex selection is specified
            if self.selection is not None:
                self._prepare_vertex_mask()
            
            print(f"MOSAIC model loaded on {self.device} for subject {self.subject}")
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load MOSAIC model: {str(e)}")
    
    def _prepare_vertex_mask(self):
        """
        Prepare the combined vertex selection mask from ROI and/or vertex indices.
        
        This method combines ROI-based selection and direct vertex index selection
        into a single boolean mask that will be used to slice the model output.
        """
        # Initialize mask as all False
        combined_mask = np.zeros(self.N_VERTICES, dtype=bool)
        
        # Add ROI-based selection
        if self.roi_list is not None:
            # Load metadata to get ROI masks
            metadata = self.get_metadata(
                berg_dir=self.berg_dir,
                subject=self.subject
            )
            
            roi_all_vertices = metadata["fmri"]["roi_all_vertices"]
            
            # Combine all selected ROIs
            for roi in self.roi_list:
                if roi not in roi_all_vertices:
                    raise InvalidParameterError(
                        f"ROI '{roi}' not found in metadata. Available ROIs: {list(roi_all_vertices.keys())}"
                    )
                roi_mask = roi_all_vertices[roi].astype(bool)
                combined_mask = combined_mask | roi_mask
        
        # Add direct vertex index selection
        if self.vertex_index is not None:
            combined_mask = combined_mask | self.vertex_index
        
        # Store the combined mask
        self.vertex_mask = combined_mask
        
        # Check if any vertices are selected
        if not np.any(self.vertex_mask):
            raise InvalidParameterError(
                "No vertices selected. Please check your ROI and vertex index specifications."
            )
    
    def generate_response(
        self, 
        stimulus: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate in silico fMRI responses for a batch of images.
        
        Parameters
        ----------
        stimulus : np.ndarray
            Images for which the in silico neural responses are generated. Must be
            a 4-D numpy array of shape (Batch size x 3 RGB Channels x Width x
            Height) consisting of integer values in range [0, 255].
            Images should have square dimensions.
        show_progress : bool, default=True
            Whether to show a progress bar during encoding.
        
        Returns
        -------
        np.ndarray
            Predicted fMRI responses with shape (batch_size, n_vertices).
            The number of vertices depends on the selection (up to 57,051).
        """
        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
            raise StimulusError(
                "Stimulus must be a 4D numpy array (batch, channels, height, width)"
            )
        
        if stimulus.shape[1] != 3:
            raise StimulusError(
                f"Stimulus must have 3 color channels, got {stimulus.shape[1]}"
            )
        
        # Convert numpy array to PIL Images for MOSAIC
        # MOSAIC expects list of PIL Images
        images = []
        for i in range(stimulus.shape[0]):
            # Convert from (C, H, W) to (H, W, C)
            img_array = np.transpose(stimulus[i], (1, 2, 0))
            # Ensure uint8
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            # Convert to PIL Image
            pil_img = Image.fromarray(img_array, mode='RGB')
            images.append(pil_img)
        
        # Run inference through MOSAIC
        # Specify the subject as a list with single integer
        results = self.inference.run(
            images=images,
            names_and_subjects={"NaturalScenesDataset": [self.subject]}
        )
        
        # Extract responses for the specified subject
        # Results are organized as: results["NaturalScenesDataset"][f"sub-{subject:02d}"]
        dataset_results = results["NaturalScenesDataset"]
        subject_key = f"sub-{self.subject:02d}"
        
        if subject_key not in dataset_results:
            raise ModelLoadError(
                f"Subject {subject_key} not found in MOSAIC results. "
                f"Available subjects: {list(dataset_results.keys())}"
            )
        
        # Get the tensor and convert to numpy
        insilico_fmri_responses = dataset_results[subject_key]
        if isinstance(insilico_fmri_responses, torch.Tensor):
            insilico_fmri_responses = insilico_fmri_responses.cpu().numpy()
        
        # Apply vertex selection if specified
        if self.vertex_mask is not None:
            insilico_fmri_responses = insilico_fmri_responses[:, self.vertex_mask]
        
        # Convert to float32
        insilico_fmri_responses = insilico_fmri_responses.astype(np.float32)
        
        return insilico_fmri_responses
    
    @classmethod
    def get_metadata(
        cls, 
        berg_dir: Optional[str] = None,
        subject: Optional[int] = None,
        model_instance: Optional[BaseModelInterface] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        berg_dir : str, optional
            Path to BERG directory containing metadata.
        subject : int, optional
            Subject number (1-8).
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        **kwargs
            Additional parameters.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing fMRI info, ROI masks, and noise ceilings.
        """
        # If model_instance is provided, extract parameters from it
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
            subject = model_instance.subject
        
        # If this method is called on an instance (rather than the class)
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
        validate_subject(subject, cls.VALID_SUBJECTS)
        
        # Build metadata path
        file_name = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-fmri',
            'train_dataset-mosaic',
            'model-mosaic',
            'metadata',
            'NSD',
            f'sub-{subject:02d}.npy'
        )
        
        # Load metadata if file exists
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(
                f"Metadata file not found: {file_name}"
            )
    
    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique string identifier.
        
        Returns
        -------
        str
            Model ID string that identifies this model in the registry.
        """
        return cls.MODEL_ID
    
    def cleanup(self) -> None:
        """
        Release memory and resources associated with the model.
        
        Frees GPU memory by moving models to CPU and clearing CUDA cache
        if available, preventing memory leaks when working with multiple models.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Free GPU memory if using CUDA
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            
            # Clear references to large objects
            self.model = None
            self.inference = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
