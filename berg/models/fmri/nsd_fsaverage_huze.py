import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from typing import Dict, Any, Optional
from berg.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from berg.core.model_registry import register_model
from berg.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_binary_array,
    get_selected_indices,
    validate_roi,
)
from berg.interfaces.base_model import BaseModelInterface
from berg.models.fmri.huze.model import _load_one_model, TowPartModel, BrainEncodingModel


# Load model info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-nsd_fsaverage-huze.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.nsd_fsaverage_huze",  # Replace with actual path
    class_name="HUZE",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "nsd_fsaverage"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-nsd_fsaverage-huze.yaml")
)



class HUZE(BaseModelInterface):
    """
    Memory Encoding Model (MEM) for predicting fMRI responses across the entire cortex.

    This is the pre-trained Memory Encoding Model by Yang et al. (2023) that won 
    the Algonauts 2023 visual brain competition with a score of 70.8. The model 
    was trained with memory information from up to 32 previous image frames, 
    enabling predictions across both visual and non-visual brain regions.

    The architecture uses a two-part model design for computational efficiency
    when handling ~160k brain vertices with memory integration from training.
    
    Architecture:
    - Vision Transformer (ViT) backbone for feature extraction
    - Memory compressor module for processing previous frames
    - Time embedding for temporal context
    - Subject-aware conditioning
    - Two-part model architecture for scalability
    
    References
    ----------
    Yang, H., Gee, J., & Shi, J. (2023). Memory Encoding Model. 
    arXiv:2308.01175 [cs.CV]
    
    """

    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    VERTICES_LENGTH = 163842

    def __init__(self, subject: int, selection: Dict, device: str = "auto", berg_dir: Optional[str] = None, **kwargs):
        """
        Initialize the Memory Encoding Model for a specific NSD subject.

        Each subject has individually trained model weights optimized for their
        specific brain anatomy and response patterns from the NSD dataset.

        Parameters
        ----------
        subject : int
            Subject ID from the NSD dataset (1-8). Each subject has individually
            trained model weights optimized for their specific brain anatomy and
            response patterns.
        selection : dict
            Specifies which brain regions/vertices to include in model responses.
            Options include:
            - roi: Region of interest name (e.g., "V1", "FFA-1", "lateral")
            - lh_vertices: Binary mask for left hemisphere vertices  
            - rh_vertices: Binary mask for right hemisphere vertices
        device : str
            Device to run the model on ('cpu', 'cuda', or 'auto').
        berg_dir : str, optional
            Path to BERG directory containing model weights and metadata.
            Required for loading pre-trained model parameters.
        **kwargs
            Additional model-specific parameters for future extensions.
        """
        self.subject = subject
        self.berg_dir = berg_dir
        self.model = None
        
        # Parameters from selection
        self.selection = selection
        self.roi = None
        self.selected_lh_vertices = None
        self.selected_rh_vertices = None
        
        # Other parameters:
        current_dir = os.path.dirname(__file__)  # path to nsd_mem.py
        self.cfg_path = os.path.join(current_dir, "huze", "config.yaml")
        
        # Validate Parameters
        self._validate_parameters()

        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device


    def _validate_parameters(self):
        """
        Validate the input parameters against the model specs.
        """
        if self.subject not in self.VALID_SUBJECTS:
            raise InvalidParameterError(
                f"Subject must be one of {self.VALID_SUBJECTS}, got {self.subject}"
            )

        # For selection Paramter 
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Validate ROI
            if "roi" in self.selection:
                self.roi = validate_roi(
                    self.selection["roi"], self.VALID_ROIS
                )

            # Validate LH vertices
            if "lh_vertices" in self.selection:
                lh_vertices_array = validate_binary_array(
                    self.selection["lh_vertices"],
                    self.VERTICES_LENGTH,
                    "lh_vertices"
                )
                self.selected_lh_vertices = get_selected_indices(lh_vertices_array)

            # Validate RH vertices
            if "rh_vertices" in self.selection:
                rh_vertices_array = validate_binary_array(
                    self.selection["rh_vertices"],
                    self.VERTICES_LENGTH,
                    "rh_vertices"
                )
                self.selected_rh_vertices = get_selected_indices(rh_vertices_array)
        
        
    def _transform_image(self, x: np.ndarray) -> torch.Tensor:
        """
        Preprocess input images for the Memory Encoding Model.
        
        Parameters
        ----------
        x : np.ndarray
            Input images as numpy array with shape (B, C, H, W) and values [0, 255]
        """
        # Convert to torch tensor and normalize to [0, 1]
        x = torch.from_numpy(x).float() / 255.0  # Shape: (B, C, H, W)
        
        # Resize to (224, 224) only if not already that size
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Apply ImageNet normalization
        means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - means) / stds
        
        # Move to device after all transformations
        x = x.to(self.device)
        
        return x

    def load_model(self) -> None:
        """
        Load the two-part Memory Encoding Model architecture and weights.

        This method loads the complete MEM model consisting of:
        1. Part 1: Handles subset of brain vertices with memory integration
        2. Part 2: Handles remaining vertices  
        3. Voxel indices mapping for combining predictions
        """
        try:

            # Select the used vertices
            # If the ROI is provided, select the LH and RH vertices based on the chosen ROI
            if self.roi is not None:
                metadata_dir = os.path.join(
                    self.berg_dir, 'encoding_models', 'modality-fmri',
                    'train_dataset-nsd_fsaverage', 'model-huze',
                    'metadata', f'metadata_subject-{self.subject:02d}.npy'
                )
                metadata_dict = np.load(metadata_dir, allow_pickle=True).item()
                self.selected_lh_vertices = metadata_dict['fmri']\
                    ['lh_fsaverage_rois'][self.roi]
                self.selected_rh_vertices = metadata_dict['fmri']\
                    ['rh_fsaverage_rois'][self.roi]
                    
            # Select vertices based on one-hot encoded vector only if the ROI is not provided
            else:
                # If selected vertices is not set, use all vertice
                if self.selected_lh_vertices is None:
                        self.selected_lh_vertices = range(self.VERTICES_LENGTH)
                if self.selected_rh_vertices is None:
                        self.selected_rh_vertices = range(self.VERTICES_LENGTH)
                        
            
            # Load the model
            self.model = self._load_model(self.device)
            
            # Load the transform
            self.transform = self._transform_image

            print(f"Model loaded on {self.device} for subject {self.subject}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
        
        
    def _load_model(self, device):
        """
        Load individual model components and combine into two-part architecture.

        Loads the pre-trained model parts and voxel indices, then combines them
        into a unified TowPartModel that can predict responses across the full
        brain surface while incorporating memory information.

        Parameters
        ----------
        device : str
            Target device for model computation
        """
        
        # Load model checkpoints
        model_path1 = os.path.join(
            self.berg_dir, 
            f"encoding_models/modality-fmri/train_dataset-nsd_fsaverage/model-huze/encoding_models_weights/subj{self.subject:02d}_part1.pth"
        )
        
        model_path2 = os.path.join(
            self.berg_dir, 
            f"encoding_models/modality-fmri/train_dataset-nsd_fsaverage/model-huze/encoding_models_weights/subj{self.subject:02d}_part2.pth"
        )
        
        # Load models
        model1: BrainEncodingModel = _load_one_model(model_path1, f"subj{self.subject:02d}", self.cfg_path)
        model2: BrainEncodingModel = _load_one_model(model_path2, f"subj{self.subject:02d}", self.cfg_path)
        
        # Get model Indices
        voxel_indices_path = os.path.join(
            self.berg_dir, 
            "encoding_models/modality-fmri/train_dataset-nsd_fsaverage/model-huze/encoding_models_weights/part1_voxel_indices.pt"
        )
        voxel_indices = torch.load(voxel_indices_path, weights_only=False)[f"subj{self.subject:02d}"]
        
        # Initalize model
        model = TowPartModel(model1, model2, voxel_indices)
        model = model.to(device).eval()
        
        return model
        
        

    def generate_response(
        self,
        stimulus: np.ndarray,
        show_progress: bool = True) -> np.ndarray:
        """
        Generate fMRI response predictions for input images using the pre-trained model.

        Processes images through the Vision Transformer backbone and two-part 
        architecture to predict brain responses across selected vertices. Images
        are processed in batches for memory efficiency.

        Parameters
        ----------
        stimulus : np.ndarray
            Input images with shape (batch_size, 3, height, width) and integer
            values in range [0, 255]. Images should be square dimensions.
        show_progress : bool, default=True
            Whether to display progress bar during batch processing.
            
        Returns
        -------
        tuple of np.ndarray
            (left_hemisphere, right_hemisphere) predictions where each array
            has shape (batch_size, n_vertices). Responses are in z-scored units
            following NSD preprocessing conventions.
        """
        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
            raise StimulusError(
                "Stimulus must be a 4D numpy array (batch, channels, height, width)"
            )
        
        # Preprocess stimulus
        images = self.transform(stimulus)
        
        # Extract features and generate responses in batches
        batch_size = 100
        n_batches = int(np.ceil(len(images) / batch_size))
        
        if show_progress:
            progress_bar = tqdm(range(n_batches), desc='Encoding fMRI responses')
        else:
            progress_bar = range(n_batches)
        
        all_outputs = []

        # Check if vertex selection is applied
        lh_vertices_selected = (self.selected_lh_vertices is not None and 
                            not (isinstance(self.selected_lh_vertices, range) and 
                                    self.selected_lh_vertices == range(self.VERTICES_LENGTH)))

        rh_vertices_selected = (self.selected_rh_vertices is not None and 
                            not (isinstance(self.selected_rh_vertices, range) and 
                                    self.selected_rh_vertices == range(self.VERTICES_LENGTH)))

        if lh_vertices_selected or rh_vertices_selected:
            # Create combined indices for the model (remap to 0-327684 space)
            if lh_vertices_selected:
                selected_lh_model_indices = torch.tensor(self.selected_lh_vertices, dtype=torch.long)
            else:
                selected_lh_model_indices = torch.arange(self.VERTICES_LENGTH, dtype=torch.long)
            
            if rh_vertices_selected:
                selected_rh_model_indices = torch.tensor(self.selected_rh_vertices, dtype=torch.long) + self.VERTICES_LENGTH
            else:
                selected_rh_model_indices = torch.arange(self.VERTICES_LENGTH, self.VERTICES_LENGTH * 2, dtype=torch.long)
            
            # Combine both hemispheres
            all_selected_voxel_indices = torch.cat([selected_lh_model_indices, selected_rh_model_indices])
            
            print(f"Computing responses for {len(all_selected_voxel_indices)} voxels instead of all {self.VERTICES_LENGTH * 2} voxels")
            print(f"LH: {len(selected_lh_model_indices)} vertices, RH: {len(selected_rh_model_indices)} vertices")
        else:
            # No selection - use original behavior
            all_selected_voxel_indices = None
            print("Computing responses for all vertices (no selection applied)")
            

        with torch.no_grad():
            for b in progress_bar:
                # Image batch indices
                idx_start = b * batch_size
                idx_end = idx_start + batch_size
                # Extract features
                img_batch = images[idx_start:idx_end]
                features = self.model(img_batch, voxel_indices=all_selected_voxel_indices)
                all_outputs.append(features.cpu())
                if show_progress and isinstance(progress_bar, tqdm):
                    encoded_images = min((b + 1) * batch_size, len(images))
                    progress_bar.set_postfix({
                        'Encoded images': encoded_images,
                        'Total images': len(images)
                    })
        
        # Concatenate all outputs into one tensor
        model_outputs = torch.cat(all_outputs, dim=0)

        if lh_vertices_selected or rh_vertices_selected:
            # Split back into hemispheres based on how many we selected from each
            n_lh_selected = len(selected_lh_model_indices)
            n_rh_selected = len(selected_rh_model_indices)
            
            lh_insilico_fmri = model_outputs[:, :n_lh_selected]
            rh_insilico_fmri = model_outputs[:, n_lh_selected:n_lh_selected + n_rh_selected]
        else:
            # Original behavior - split in half
            voxels = model_outputs.shape[1]
            voxels_half = voxels // 2
            lh_insilico_fmri = model_outputs[:, :voxels_half]
            rh_insilico_fmri = model_outputs[:, voxels_half:]

        # Convert to numpy arrays and return
        lh_insilico_fmri = lh_insilico_fmri.numpy()
        rh_insilico_fmri = rh_insilico_fmri.numpy()
        return (lh_insilico_fmri, rh_insilico_fmri)



    @classmethod
    def get_metadata(cls, berg_dir=None, subject=None, model_instance=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.

        Parameters
        ----------
        berg_dir : str
            Path to the BERG directory where metadata is stored.
        subject : int
            Subject number.
        model_instance : BaseModelInterface, optional
            If provided, parameters can be extracted directly from the model instance.
        **kwargs
            Additional model-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """

        # Extract parameters from instance if available
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
            subject = model_instance.subject

        # Also allow metadata retrieval from class instance
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir
            subject = cls.subject

        # Validate required parameters
        missing = []
        if berg_dir is None: missing.append("berg_dir")
        if subject is None: missing.append("subject")

        if missing:
            raise InvalidParameterError(f"Required parameters missing: {', '.join(missing)}")

        # Optional: validate against allowed values
        validate_subject(subject, cls.VALID_SUBJECTS)

        # Build metadata path
        filename = os.path.join(
            berg_dir,
            "encoding_models",
            "modality-fmri",             
            "train_dataset-nsd_fsaverage",       
            "model-huze",              
            "metadata",
            f'metadata_subject-{subject:02d}.npy')

        # Load metadata
        if os.path.exists(filename):
            metadata = np.load(filename, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found at: {filename}")

    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique identifier.

        Returns
        -------
        str
            Model ID string from the YAML config.
        """
        return cls.MODEL_ID

    def cleanup(self) -> None:
        """
        Release resources (e.g., GPU memory) when finished.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Free GPU memory if using CUDA
            if hasattr(self.model, 'to'):
                self.model.to('cpu')

            # Clear references
            self.model = None

            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()