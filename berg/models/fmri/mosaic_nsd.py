import os
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional, Union, List
from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model
from berg.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from berg.core.parameter_validator import (
    validate_subjects,
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
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-mosaic-CNN8_multihead_subNSD_verticesAll.yaml")
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
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-mosaic-CNN8_multihead_subNSD_verticesAll.yaml")
)


class FMRIEncodingModel(BaseModelInterface):
    """
    fMRI encoding model using MOSAIC CNN8 architecture
    for Natural Scenes Dataset (NSD) subjects.
    """
    
    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = [s for s in model_info["parameters"]["subject"]["valid_values"] if isinstance(s, int)]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    N_VERTICES_MODEL = 57051  # Model output size
    N_VERTICES_FULL = 91282   # Full fsLR32k brain space
    
    def __init__(
        self, 
        subject: Union[int, List[int], str],
        selection: Optional[Dict] = None,
        device: str = "auto", 
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the MOSAIC fMRI encoding model for one or more subjects.
        
        Parameters
        ----------
        subject : int, list of int, or "all"
            Subject number(s) from the NSD dataset (1-8), or "all" for all subjects.
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            Applied equally to all subjects when multiple subjects are specified.
            - roi: List of region labels (e.g., ['L_V1', 'R_V1'])
            - voxel_index: Binary one-hot encoded vector (57051,) indicating vertices to include
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
        berg_dir : str, optional
            Path to the BERG directory containing metadata files.
        """
        self.subject_input = subject
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
        self.subjects = validate_subjects(self.subject_input, self.VALID_SUBJECTS)
        
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
                # Validate as binary array in full 91k space
                validated_array = validate_binary_array(
                    vertex_index,
                    expected_length=self.N_VERTICES_FULL,
                    parameter_name="voxel_index"
                )
                self.vertex_index = validated_array.astype(bool)
        
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
            
            subject_str = "all subjects" if len(self.subjects) == len(self.VALID_SUBJECTS) else f"subject(s) {self.subjects}"
            print(f"MOSAIC model loaded on {self.device} for {subject_str}")
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load MOSAIC model: {str(e)}")
    
    def _prepare_vertex_mask(self):
        """
        Prepare vertex selection from ROI and/or vertex indices in full 91k space.
        Stores indices that will be used to slice predictions after expanding to 91k.
        Uses the first subject in the list for loading ROI metadata.
        """
        from mosaic.models.transforms import SelectROIs
        
        # Get the vertex mapping (57k model space → 91k full space)
        all_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 23)])
        self.vertex_mapping = np.array(all_selector.selected_roi_indices)
        
        # Validate that vertex indices don't exceed model prediction range
        max_model_vertex = self.vertex_mapping.max()
        
        # Initialize combined indices set
        selected_indices = set()
        
        # Add ROI-based selection
        if self.roi_list is not None:
            # Load metadata to get ROI indices
            metadata = self.get_metadata(
                berg_dir=self.berg_dir,
                subject=self.subjects[0]
            )
            
            # Combine all selected ROIs
            for roi in self.roi_list:
                if roi not in metadata["fmri"]["roi"]:
                    raise InvalidParameterError(
                        f"ROI '{roi}' not found in metadata. Available ROIs: {list(metadata['fmri']['roi'].keys())}"
                    )
                roi_indices = metadata["fmri"]["roi"][roi]
                selected_indices.update(roi_indices)
        
        # Add direct vertex index selection
        if self.vertex_index is not None:
            vertex_indices = np.where(self.vertex_index)[0]
            
            # Validate indices are within model prediction range
            if np.any(vertex_indices > max_model_vertex):
                invalid_indices = vertex_indices[vertex_indices > max_model_vertex]
                raise InvalidParameterError(
                    f"voxel_index contains indices beyond model prediction range. "
                    f"Max allowed: {max_model_vertex}, found: {invalid_indices[:5].tolist()}..."
                )
            
            selected_indices.update(vertex_indices)
        
        # Convert to sorted array and store
        self.vertex_mask = np.array(sorted(selected_indices))
        
        # Check if any vertices are selected
        if len(self.vertex_mask) == 0:
            raise InvalidParameterError(
                "No vertices selected. Please check your ROI and vertex index specifications."
            )
    
    def generate_response(
        self, 
        stimulus: np.ndarray,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate in silico fMRI responses for a batch of images.
        
        Parameters
        ----------
        stimulus : np.ndarray
            Images for which neural responses are generated. Shape: (batch, 3, height, width)
            with integer values in range [0, 255].
        show_progress : bool, default=True
            Whether to show a progress bar during encoding.
        
        Returns
        -------
        dict
            Nested dict with structure:
            {"NaturalScenesDataset": {"sub-01": array, "sub-02": array, ...}}
            where each array has shape (batch_size, n_vertices) and dtype float32.
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
        results = self.inference.run(
            images=images,
            names_and_subjects={"NaturalScenesDataset": self.subjects}
        )
        
        # Extract responses for the specified subjects
        # Results are organized as: results["NaturalScenesDataset"][f"sub-{subject:02d}"]
        dataset_results = results["NaturalScenesDataset"]
        
        processed_results = {}
        for subject_id in self.subjects:
            subject_key = f"sub-{subject_id:02d}"
            
            insilico_fmri_responses = dataset_results[subject_key]
            if isinstance(insilico_fmri_responses, torch.Tensor):
                insilico_fmri_responses = insilico_fmri_responses.cpu().numpy()
            
            # Convert to float32
            insilico_fmri_responses = insilico_fmri_responses.astype(np.float32)
            
            # Apply vertex selection if specified
            if self.vertex_mask is not None:
                # Expand predictions to full 91k space
                predictions_full = np.full((insilico_fmri_responses.shape[0], self.N_VERTICES_FULL), np.nan, dtype=np.float32)
                predictions_full[:, self.vertex_mapping] = insilico_fmri_responses
                
                # Slice using selected vertex indices
                insilico_fmri_responses = predictions_full[:, self.vertex_mask]
            
            processed_results[subject_key] = insilico_fmri_responses
        
        return {"NaturalScenesDataset": processed_results}
    
    @classmethod
    def get_metadata(
        cls, 
        berg_dir: Optional[str] = None,
        subject: Optional[Union[int, List[int], str]] = None,
        model_instance: Optional[BaseModelInterface] = None,
        **kwargs
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        berg_dir : str, optional
            Path to BERG directory containing metadata.
        subject : int, list of int, or "all", optional
            Subject number(s) (1-8) or "all".
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        
        Returns
        -------
        dict
            If single subject: metadata dictionary
            If multiple subjects: dict mapping subject keys to metadata dicts
            {"sub-01": metadata_dict, "sub-02": metadata_dict, ...}
        """
        # If model_instance is provided, extract parameters from it
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
            subject = model_instance.subjects
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir
            subject = cls.subjects
        
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
        
        if isinstance(subject, (int, list, str)):
            subjects = validate_subjects(subject, cls.VALID_SUBJECTS)
        else:
            subjects = subject
        
        metadata_dict = {}
        for subj_id in subjects:
            file_name = os.path.join(
                berg_dir,
                'encoding_models',
                'modality-fmri',
                'train_dataset-mosaic',
                'model-mosaic',
                'metadata',
                'NSD',
                f'sub-{subj_id:02d}.npy'
            )
            
            if os.path.exists(file_name):
                metadata = np.load(file_name, allow_pickle=True).item()
                metadata_dict[f"sub-{subj_id:02d}"] = metadata
            else:
                raise FileNotFoundError(
                    f"Metadata file not found: {file_name}"
                )
        
        if len(subjects) == 1:
            return metadata_dict[f"sub-{subjects[0]:02d}"]
        else:
            return metadata_dict
    
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