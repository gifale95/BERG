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
from nest.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_roi,
)

# Load model model_info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-nsd-fwrf.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="nest.models.fmri.nsd_fwrf",
    class_name="FMRIEncodingModel",
    modality=model_info.get("modality", "fmri"),
    dataset=model_info.get("dataset", "nsd"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-nsd-fwrf.yaml")
)


class FMRIEncodingModel(BaseModelInterface):
    """
    fMRI encoding model using feature-weighted receptive fields (fwrf)
    for the Natural Scenes Dataset (NSD).
    """
    
    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    
    def __init__(self, subject: int, selection: Dict, device:str="auto", nest_dir: Optional[str] = None):
        """
        Initialize the fMRI encoding model for a specific subject and ROI.
        
        Parameters
        ----------
        subject : int
            Subject number from the NSD dataset (1-8).
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            - roi: Region of interest (e.g., "V1", "FFA-1", "lateral").
        nest_dir : str, optional
            Path to the NEST directory containing model files and weights.
        """
        self.img_chan = 3
        self.resize_px = 227
        self.subject = subject
        self.nest_dir = nest_dir
        self.model = None
        
        # Parameters from selection
        self.selection = selection
        self.roi = None
        
        # Validate Parameters
        self._validate_parameters()
        
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def _validate_parameters(self):
        """
        Validate the subject and ROI values against the model info.
        
        Verifies that the provided subject ID and ROI name are among
        the supported values defined in the model's modelinfo.
        """
        
        # Validate subject
        validate_subject(self.subject, self.VALID_SUBJECTS)
        
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Individual validations
            if "roi" in self.selection:
                self.roi = validate_roi(
                    self.selection["roi"], self.VALID_ROIS
                )
        # Ensure selection is provided
        else:
            raise InvalidParameterError("Parameter 'selection' is required but was not provided")
        


    def load_model(self, device: str = "auto") -> None:
        """
        Load model weights and prepare the encoder and fwrf components.
        
        Parameters
        ----------
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
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
        
        Parameters
        ----------
        trained_models : list
            List of loaded weight checkpoints containing model parameters.
        stim_mean : torch.Tensor
            Mean image used for input normalization.
        
        Notes
        -----
        This function:
        1. Creates dummy input images to initialize the encoder
        2. Instantiates shared encoder models to extract feature maps
        3. Initializes subject-specific fwrf models to map features to voxel responses
        4. Loads pre-trained weights for both encoder and fwrf components
        5. Sets all models to evaluation mode for inference
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
        
        Parameters
        ----------
        stimulus : np.ndarray
            Images for which the in silico neural responses are generated. Must be
            a 4-D numpy array of shape (Batch size x 3 RGB Channels x Width x
            Height) consisting of integer values in the range [0, 255].
            Furthermore, the images must be of square size (i.e., equal width and
            height).
        
        Returns
        -------
        np.ndarray
            Predicted fMRI responses with shape (batch_size, n_voxels).
            The number of voxels varies by ROI and subject.
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
        

    @classmethod
    def get_metadata(cls, model_instance=None, nest_dir=None, subject=None, roi=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        This method works both as an instance method on a loaded model
        and as a class method to retrieve metadata without loading the model.
        
        Parameters
        ----------
        model_instance : FMRIEncodingModel, optional
            If provided, extract necessary parameters from this model instance.
        nest_dir : str, optional
            Path to NEST directory. Required when not using model_instance.
        subject : int, optional
            Subject number. Required when not using model_instance.
        roi : str, optional
            Region of interest. Required when not using model_instance.
        **kwargs
            Additional parameters
                
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """
        # If called with a model instance, extract parameters from it
        if model_instance is not None:
            nest_dir = model_instance.nest_dir
            subject = model_instance.subject
            roi = model_instance.roi
        # Check if this is an instance method call
        elif hasattr(cls, 'nest_dir') and hasattr(cls, 'subject') and hasattr(cls, 'roi'):
            nest_dir = cls.nest_dir
            subject = cls.subject
            roi = cls.roi
        # For backward compatibility, check for args
        elif 'args' in kwargs and kwargs['args'] and isinstance(kwargs['args'][0], cls):
            instance = kwargs['args'][0]
            nest_dir = instance.nest_dir
            subject = instance.subject
            roi = instance.roi
        
        # At this point, parameters should be provided
        if nest_dir is None or subject is None or roi is None:
            raise InvalidParameterError("Parameters 'nest_dir', 'subject', and 'roi' are required")
        
        # Validate parameters
        validate_subject(subject, cls.VALID_SUBJECTS)
        validate_roi(roi, cls.VALID_ROIS)
        
        # Build metadata path
        file_name = os.path.join(nest_dir,
                            'encoding_models', 
                            'modality-fmri',
                            'train_dataset-nsd', 
                            'model-fwrf', 
                            'metadata',
                            f'metadata_sub-{subject:02d}_roi-{roi}.npy')
        
        # Load metadata if file exists
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found for subject {subject}, roi {roi}")

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
        if available, preventing memory leaks when working with multiple
        models.
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