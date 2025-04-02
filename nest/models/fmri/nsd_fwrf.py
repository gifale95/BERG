import os
import numpy as np
import torch
import yaml
from torchvision import transforms as trn
from typing import Dict, Any, Optional
from nest.interfaces.base_model import BaseModelInterface
from nest.core.model_registry import register_model
from nest.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from nest.models.fmri.fwrf.torch_gnet import Encoder
from nest.models.fmri.fwrf.torch_mpf import Torch_LayerwiseFWRF
from nest.models.fmri.fwrf.load_nsd import image_feature_fn
from nest.models.fmri.fwrf.torch_joint_training_unpacked_sequences import *

# Load model metadata from YAML
def load_model_metadata():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri_nsd_fwrf.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load metadata once at the top
metadata = load_model_metadata()

# Register this model with the registry using metadata
register_model(
    model_id=metadata["model_id"],
    module_path="nest.models.fmri.nsd_fwrf",
    class_name="FMRIEncodingModel",
    modality=metadata.get("modality", "fmri"),
    dataset=metadata.get("dataset", "nsd"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri_nsd_fwrf.yaml")
)


class FMRIEncodingModel(BaseModelInterface):
    """
    fMRI encoding model using feature-weighted receptive fields (fwrf)
    for the Natural Scenes Dataset (NSD).
    """
    
    print(metadata)
    
    MODEL_ID = metadata["model_id"]
    VALID_SUBJECTS = metadata["parameters"]["subject"]["valid_values"]
    VALID_ROIS = metadata["parameters"]["roi"]["valid_values"]
    
    
    def __init__(self, subject: int, roi: str, device:str="auto", nest_dir: Optional[str] = None):
        """
        Initialize the fMRI encoding model for a specific subject and ROI.

        Args:
            subject (int): Subject number (1–8).
            roi (str): Region of interest (e.g., "V1", "FFA", "lateral").
            nest_dir (Optional[str]): Path to the NEST directory.
        """
        self.img_chan = 3
        self.resize_px = 227
        self.subject = subject
        self.roi = roi
        self.nest_dir = nest_dir
        self.model = None
        self._validate_parameters()
        
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def _validate_parameters(self):
        """
        Validate the subject and ROI values against the metadata.
        """
        if self.subject not in self.VALID_SUBJECTS:
            raise InvalidParameterError(
                f"Subject must be one of {self.VALID_SUBJECTS}, got {self.subject}"
            )
        
        if self.roi not in self.VALID_ROIS:
            raise InvalidParameterError(
                f"ROI must be one of {self.VALID_ROIS}, got {self.roi}"
            )

    def load_model(self, device: str = "auto") -> None:
        """
        Load model weights and prepare the encoder and fwrf components.

        Args:
            device (str): Target device ("cpu", "cuda", or "auto").
        """
        try:
            
            
            
            # Build model weight paths
            if self.roi in ["lateral", "ventral"]:
                model_paths = [
                    os.path.join(self.nest_dir, 'encoding_models', 'modality-fmri',
                                 'train_dataset-nsd', 'model-fwrf', 'encoding_models_weights',
                                 f'weights_sub-{self.subject:02d}_roi-{self.roi}_split-{i}.pt')
                    for i in [1, 2]
                ]
            else:
                model_paths = [os.path.join(self.nest_dir, 'encoding_models', 'modality-fmri',
                                            'train_dataset-nsd', 'model-fwrf', 'encoding_models_weights',
                                            f'weights_sub-{self.subject:02d}_roi-{self.roi}.pt')]

            # Load models
            trained_models = [torch.load(path, map_location=torch.device("cpu"), weights_only=False) for path in model_paths]
            stim_mean = trained_models[0]["stim_mean"]  # is the same across models so we can take the first one

            # Model instantiation
            self._initialize_models(trained_models, stim_mean)

            print(f"Model loaded on {self.device} for subject {self.subject}, ROI {self.roi}")
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
        
        
    def _initialize_models(self, trained_models, stim_mean):
        """
        Initializes the fMRI encoding model components.

        This function sets up the shared encoder model and the subject-specific 
        feature-weighted receptive field (fwrf) models, loads trained weights, 
        and ensures models are in evaluation mode.

        Args:
            trained_models (List[dict]): List of loaded weight checkpoints.
            stim_mean (torch.Tensor): Mean image used for input normalization.

        Steps:
        1. Create dummy input images to initialize the encoder.
        2. Instantiate a shared encoder model to extract feature maps.
        3. Pass dummy images through the encoder to obtain feature maps.
        4. Initialize subject-specific fwrf models to map features to voxel responses.
        5. Load pre-trained weights for the encoder and fwrf models.
        6. Set all models to evaluation mode for inference.
        """


        # Dummy images for model initialization for proper setup
        dummy_images = np.random.randint(0, 255, (20, self.img_chan, self.resize_px, self.resize_px))

        # Shared encoder model across ROI and subjects to extract features
        self.shared_model = [
            Encoder(mu=stim_mean,  # to normalize input
                    trunk_width=64,  # conv width
                    use_prefilter=1).to(self.device)  # prefilter: Initial Conv Maps
            for _ in trained_models
        ]
        
        
        for model in self.shared_model:
            _, fmaps, _ = model(torch.from_numpy(dummy_images).to(self.device))

        # Subject-specific fwrf models
        # Nonlinearity (Pre and Post-Processing)
        _log_act_fn = lambda _x: torch.log(1 + torch.abs(_x)) * torch.tanh(_x) 
        
        self.subject_fwrfs = [
            {self.subject: Torch_LayerwiseFWRF(
                fmaps,  # Feature Maps from Encoder
                nv=len(trained["best_params"]["fwrfs"][self.subject]["b"]),  # Number of Voxels to predict
                pre_nl=_log_act_fn,  # Pre Non-Linearity
                post_nl=_log_act_fn,   # Post Non-Linearity
                dtype=np.float32).to(self.device)}
            for trained in trained_models
        ]

        # Load weights
        for i, trained_model in enumerate(trained_models):
            self.shared_model[i].load_state_dict(trained_model["best_params"]["enc"])
            for s, sd in self.subject_fwrfs[i].items():
                sd.load_state_dict(trained_model["best_params"]["fwrfs"][s])

        # Set evaluation mode
        for model in self.shared_model:
            model.eval()
        for subject_dict in self.subject_fwrfs:
            for sd in subject_dict.values():
                sd.eval()


    def generate_response(
            self, 
            stimulus: np.ndarray) -> np.ndarray:
            """
            Generate in silico fMRI responses for a batch of visual stimuli.

            Args:
                stimulus (np.ndarray): Input array of shape (B, C, H, W).

            Returns:
                np.ndarray: fMRI responses of shape (B, V), where:
                    B = batch size,
                    V = number of predicted voxels.
            """
            # Validate stimulus
            if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
                raise StimulusError(
                    "Stimulus must be a 4D numpy array (batch, channels, height, width)"
                )
                
                
            # Preprocess images
            transform = trn.Compose([
                trn.Resize((self.resize_px,self.resize_px))
            ])
            images = torch.from_numpy(stimulus)
            images = transform(images)
            images = np.asarray(images)
            images = image_feature_fn(images)
            
            ### Model functions ###
            def _model_fn(_ext, _con, _x):
                _y, _fm, _h = _ext(_x)
                if isinstance(_con, dict):
                    return torch.cat([model(_fm) for model in _con.values()], dim=-1)
                else:
                    return _con(_fm)

            def _pred_fn(_ext, _con, xb):
                xb = torch.from_numpy(xb).to(self.device)
                return _model_fn(_ext, _con, xb)

            
            # Generate the in silico fMRI responses
            with torch.no_grad():
                if self.roi in ['lateral', 'ventral']:
                    insilico_fmri_responses_1 = subject_pred_pass(
                        _pred_fn, 
                        self.shared_model[0],
                        self.subject_fwrfs[0], 
                        images,
                        batch_size=100)
                    insilico_fmri_responses_2 = subject_pred_pass(
                        _pred_fn, 
                        self.shared_model[1],
                        self.subject_fwrfs[1], 
                        images,
                        batch_size=100)
                    
                    insilico_fmri_responses = np.append(insilico_fmri_responses_1,
                        insilico_fmri_responses_2, 1)
                else:
                    insilico_fmri_responses = subject_pred_pass(_pred_fn,
                        self.shared_model[0],
                        self.subject_fwrfs[0][self.subject], 
                        images,
                        batch_size=100)
                    
            # Convert the in silico fMRI responses to float 32
            insilico_fmri_responses = insilico_fmri_responses.astype(np.float32)

            ### Output ###
            return insilico_fmri_responses
        
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata for the current subject and ROI.

        Returns:
            Dict[str, Any]: Metadata dictionary (e.g., voxel indices, ROI info).
        """
        
        file_name = os.path.join(self.nest_dir, 
                                 'encoding_models', 
                                 'modality-fmri',
                                 'train_dataset-nsd', 
                                 'model-fwrf', 
                                 'metadata',
                                 'metadata_sub-' + format(self.subject,'02') + '_roi-' + self.roi + '.npy')
        

        metadata = np.load(file_name, allow_pickle=True).item()
        
        return metadata
          

    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique string identifier.

        Returns:
            str: Model ID.
        """
        return cls.MODEL_ID
    
    def cleanup(self) -> None:
        """
        Release memory and resources associated with the model.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Free GPU memory if using CUDA
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            
            # Clear references to large objects
            self.model = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()