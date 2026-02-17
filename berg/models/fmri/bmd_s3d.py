import os
import numpy as np
import torch
import torchvision
import yaml
from typing import Dict, Any, Optional
from berg.core.model_registry import register_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from berg.interfaces.base_model import BaseModelInterface
from berg.core.exceptions import (
    InvalidParameterError,
    ModelLoadError,
    StimulusError,
)
from berg.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_binary_array,
    get_selected_indices,
    validate_roi,
)

# Load model model_info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-bmd-s3d.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.bmd_s3d",
    class_name="FMRIEncodingModel",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "bmd_s3d"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-bmd-s3d.yaml")
)


class FMRIEncodingModel(BaseModelInterface):
    """
    fMRI encoding model trained on the BOLD Moments Dataset (BMD).
    """

    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    VOXELS_LENGTH = [
        108219, # Subject 1
        108603, # Subject 2
        108366, # Subject 3
        108283, # Subject 4
        108201, # Subject 5
        108449, # Subject 6
        108126, # Subject 7
        108407, # Subject 8
        108250, # Subject 9
        107987, # Subject 10
    ]

    def __init__(self, subject: int, selection: Dict, device:str="auto", berg_dir: Optional[str] = None):
        """
        Initialize the fMRI encoding model for a specific subject.

        Parameters
        ----------
        subject : int
            Subject number from the BMD dataset (1-10).
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
        selection : dict, optional
            Specifies for which vertices to generate the in silico fMRI responses.
            - roi: The region-of-interest (ROI) for which the in silico fMRI
                responses are generated.
            - voxels: Binary one-hot encoded vector with ones indicating
                the voxels for which the in silico fMRI responses are
                generated. This vector must have exactly the same length as the
                number of voxels, which varies for each subject:
                - Subject 1:  108,219 voxels
                - Subject 2:  108,603 voxels
                - Subject 3:  108,366 voxels
                - Subject 4:  108,283 voxels
                - Subject 5:  108,201 voxels
                - Subject 6:  108,449 voxels
                - Subject 7:  108,126 voxels
                - Subject 8:  108,407 voxels
                - Subject 9:  108,250 voxels
                - Subject 10: 107,987 voxels
                The voxels from the one-hot encoded vector are only selected if
                the "roi" key is not provided, or has value None.
        berg_dir : str, optional
            Path to the BERG directory containing model files and weights.
        """

        self.subject = subject
        self.berg_dir = berg_dir
        self.model = None

        # Parameters from selection
        self.selection = selection
        self.roi = None
        self.selected_voxels = None

        # Validate Parameters
        self._validate_parameters()

        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device


    def _validate_parameters(self):
        """
        Validate user-provided parameters against supported model yaml.

        Verifies that the provided subject ID and ROI name are among
        the supported values defined in the model's modelinfo.
        """

        # Validate subject
        validate_subject(self.subject, self.VALID_SUBJECTS)

        # Validate selection keys
        if self.selection is not None:
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Validate ROI
            if "roi" in self.selection:
                self.roi = validate_roi(
                    self.selection["roi"], self.VALID_ROIS
                )

            # Validate voxels
            if "voxels" in self.selection:
                voxels_array = validate_binary_array(
                    self.selection["voxels"],
                    self.VOXELS_LENGTH[self.subject-1],
                    "voxels"
                )
                self.selected_voxels = get_selected_indices(voxels_array)

    def load_model(self, device: str = "auto") -> None:
        """
        Load model weights, preprocessing pipeline, and regression weights.

        Loads the vision transformer backbone, preprocessing components (scaler,
        PCA), and trained regression weights for the specified subject. Sets up
        all necessary components for generating fMRI responses.

        Parameters
        ----------
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.
        """

        try:

            # Select the used voxels
            # If the ROI is provided, select the voxels based on the chosen ROI
            if self.roi is not None:
                metadata_dir = os.path.join(
                    self.berg_dir, 'encoding_models', 'modality-fmri',
                    'train_dataset-bmd', 'model-s3d',
                    'metadata', f'metadata_sub-{self.subject:02d}.npy'
                )
                metadata_dict = np.load(metadata_dir, allow_pickle=True).item()
                self.selected_voxels = metadata_dict['fmri']['rois'][self.roi]
            # Select voxels based on one-hot encoded vector only if the ROI is
            # not provided
            else:
                # If selected voxels is not set, use all voxels
                if self.selected_voxels is None:
                    self.selected_voxels = range(self.VOXELS_LENGTH[self.subject-1])

            # Load the vision transformer
            self.feature_extractor = self._load_feature_extractor(self.device)

            # Define the image preprocessing transform # !!!
            self.transform = torchvision.models.video.S3D_Weights.KINETICS400_V1.transforms()

            # Load the scaler, PCA, and trained regression weights
            self.scaler, self.pca, self.reg = self._load_encoding_weights()

            print(f"Model loaded on {self.device} for subject {self.subject}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")


    def _load_feature_extractor(self, device):
        """
        Load the ViT feature extractor for selected intermediate layers.
        
        Parameters
        ----------
        device : str
            Computation device ("cpu" or "cuda").
        
        Returns
        -------
        torch.nn.Module
            Torch feature extractor model in eval mode, configured to
            extract representations from 12 transformer layers.
        """

        # Load the model architecture
        model = torchvision.models.video.s3d(weights='KINETICS400_V1')
        
        # Select the used layers for feature extraction
        model_layers = {
            'features.2': 'layer2',
            'features.5.cat': 'layer5',
            'features.7': 'layer7',
            'features.9.cat': 'layer9',
            'features.11.cat': 'layer11',
            'features.13': 'layer13',
            'avgpool': 'avgpool'
            }
        feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
        feature_extractor.to(device)
        feature_extractor.eval()
        
        return feature_extractor


    def _load_encoding_weights(self):
        """
        Loads and configures StandardScaler and PCA models with
        pre-computed parameters for feature normalization and
        dimensionality reduction.

        Loads the weights for the linear mapping from visual features
        to fMRI responses.

        Returns
        -------
        tuple
            A tuple containing (scaler, pca, regression_weights) where:
            - scaler : StandardScaler - Fitted feature normalization object.
            - pca : PCA - Fitted principal component analysis model.
            - regression_weights: scikit-learn LinearRegression model.
        """

        # Load the weights
        reg_weight_dir = os.path.join(
            self.berg_dir, 'encoding_models', 'modality-fmri',
            'train_dataset-bmd', 'model-s3d', 'encoding_models_weights',
            'weights_sub-'+format(self.subject, '02')+'.npy'
        )
        weights_reg = np.load(reg_weight_dir, allow_pickle=True).item()
        pca_weight_dir = os.path.join(
            self.berg_dir, 'encoding_models', 'modality-fmri',
            'train_dataset-bmd', 'model-s3d', 'encoding_models_weights',
            'pca_weights.npy'
        )
        weights_pca = np.load(pca_weight_dir, allow_pickle=True).item()

        # Scaler
        scaler = StandardScaler()
        scaler.scale_ = weights_pca['scaler_param']['scale_']
        scaler.mean_ = weights_pca['scaler_param']['mean_']
        scaler.var_ = weights_pca['scaler_param']['var_']
        scaler.n_features_in_ = weights_pca['scaler_param']['n_features_in_']
        scaler.n_samples_seen_ = weights_pca['scaler_param']['n_samples_seen_']

        # PCA
        pca = TruncatedSVD(n_components=100, random_state=20200220)
        pca.components_ = weights_pca['pca_param']['components_']
        pca.explained_variance_ = weights_pca['pca_param']['explained_variance_']
        pca.explained_variance_ratio_ = weights_pca['pca_param']['explained_variance_ratio_']
        pca.singular_values_ = weights_pca['pca_param']['singular_values_']
        pca.n_features_in_ = weights_pca['pca_param']['n_features_in_']

        # Linear regression parameters
        reg = LinearRegression()
        reg.coef_ = weights_reg['coef_'][self.selected_voxels]
        reg.intercept_ = weights_reg['intercept_'][self.selected_voxels]
        reg.n_features_in_ = weights_reg['n_features_in_']

        return scaler, pca, reg


    def generate_response(
            self, 
            stimulus: np.ndarray,
            show_progress: bool = True) -> np.ndarray:
        """
        Generate in silico fMRI responses for a batch of videos.

        Parameters
        ----------
        stimulus : np.ndarray
            Videos for which the in silico neural responses are generated. Must be
            a 5-D numpy array of shape (Batch size x N video frames x 3 RGB Channels
            x Width x Height) consisting of integer values in the range [0, 255].
            Furthermore, the videos must be of square size (i.e., equal width and
            height).
        show_progress : bool, default=True
            Whether to display a progress bar during encoding.

        Returns
        -------
        insilico_fmri : np.ndarray
            In silico fMRI response array, with shape (batch_size, n_voxels),
            where the number of voxels depends on the selection parameter.
        """

        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 5:
            raise StimulusError(
                "Stimulus must be a 5D numpy array (batch, frames, channels, height, width)"
            )

        # Select 14 equally spaced frames from the video
        num_samples = 14
        num_frames = stimulus.shape[1]
        if num_samples > num_frames:
            raise ValueError("The video does not haave at least 14 frames.")
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        videos = stimulus[:,indices]

        # Preprocess the videos
        videos = self.transform(torch.from_numpy(videos).contiguous())

        # Extract features and generate responses in batches
        batch_size = 10
        n_batches = int(np.ceil(len(videos) / batch_size))

        if show_progress:
            progress_bar = tqdm(range(n_batches), desc='Encoding fMRI responses')
        else:
            progress_bar = range(n_batches)

        insilico_fmri = None

        with torch.no_grad():
            for b in progress_bar:
                # Image batch indices
                idx_start = b * batch_size
                idx_end = idx_start + batch_size

                # Extract features
                video_batch = videos[idx_start:idx_end].to(self.device)
                features = self.feature_extractor(video_batch)

                # Flatten features
                features = torch.hstack([torch.flatten(l, start_dim=1) for l in features.values()])
                features = features.detach().cpu().numpy()

                # Process features
                features = self.scaler.transform(features)
                features = self.pca.transform(features)
                features = features.astype(np.float32)

                # Generate the in silico fMRI responses
                insilico_fmri_batch = self.reg.predict(features).astype(np.float32)

                # Combine with previous batches
                if insilico_fmri is None:
                    insilico_fmri = insilico_fmri_batch
                else:
                    insilico_fmri = np.append(
                        insilico_fmri,
                        insilico_fmri_batch,
                        axis=0
                    )

                if show_progress and isinstance(progress_bar, tqdm):
                    encoded_videos = min((b + 1) * batch_size, len(videos))
                    progress_bar.set_postfix({
                        'Encoded videos': encoded_videos, 
                        'Total videos': len(videos)
                    })

        return insilico_fmri


    @classmethod
    def get_metadata(cls, berg_dir=None, subject=None, model_instance=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        berg_dir : str
            Path to BERG directory.
        subject : int
            Subject number.
        model_instance : BaseModelInterface
            If provided, extract parameters from this model instance.
        **kwargs
            Additional parameters.
                
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
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
        if berg_dir is None: missing_params.append('berg_dir')
        if subject is None: missing_params.append('subject')

        if missing_params:
            raise InvalidParameterError(f"Required parameters missing: {', '.join(missing_params)}")

        # Validate parameters
        validate_subject(subject, cls.VALID_SUBJECTS)

        # Build metadata path
        file_name = os.path.join(berg_dir,
                            'encoding_models', 
                            'modality-fmri',
                            'train_dataset-bmd', 
                            'model-s3d', 
                            'metadata',
                            f'metadata_sub-{subject:02d}.npy')

        # Load metadata if file exists
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found for subject {subject}")


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