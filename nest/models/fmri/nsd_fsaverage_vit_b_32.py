import os
import numpy as np
import torch
import torchvision
import yaml
from torchvision import transforms as trn
from typing import Dict, Any, Optional
from nest.core.model_registry import register_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from nest.interfaces.base_model import BaseModelInterface
from nest.core.exceptions import (
    InvalidParameterError,
    ModelLoadError,
    StimulusError,
)
from nest.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_binary_array,
    get_selected_indices,
    validate_roi,
)

# Load model model_info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-nsd_fsaverage-vit_b_32.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="nest.models.fmri.nsd_fsaverage_vit_b_32",
    class_name="FMRIEncodingModel",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "nsd_fsaverage"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-nsd_fsaverage-vit_b_32.yaml")
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
    VERTICES_LENGTH = 163842

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
            Specifies for which vertices to generate the in silico fMRI responses.
            - roi: The region-of-interest (ROI) for which the in silico fMRI
                responses (of both hemispherese) are generated.
            - lh_vertices: Binary one-hot encoded vector with ones indicating
                the left hemisphere (LH) vertices for which the in silico fMRI
                responses are generated. This vector must have exactly the same
                length as the number of LH fsaverage vertices (163,842). The
                vertices from the one-hot encoded vector are only selected if
                the "roi" key is not provided, or has value None.
            - rh_vertices: Binary one-hot encoded vector with ones indicating
                the right hemisphere (RH) vertices for which the in silico fMRI
                responses are generated. This vector must have exactly the same
                length as the number of RH fsaverage vertices (163,842). The
                vertices from the one-hot encoded vector are only selected if
                the "roi" key is not provided, or has value None.
        nest_dir : str, optional
            Path to the NEST directory containing model files and weights.
        """

        self.subject = subject
        self.nest_dir = nest_dir
        self.model = None

        # Parameters from selection
        self.selection = selection
        self.roi = None
        self.selected_lh_vertices = None
        self.selected_rh_vertices = None

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


    def load_model(self, device: str = "auto") -> None:
        """
        Load model weights, preprocessing pipeline, and regression layers.

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

            # Select the used vertices
            # If the ROI is provided, select the LH and RH vertices based on the
            # chosen ROI
            if self.roi is not None:
                metadata_dir = os.path.join(
                    self.nest_dir, 'encoding_models', 'modality-fmri',
                    'train_dataset-nsd_fsaverage', 'model-vit_b_32',
                    'metadata', f'metadata_subject-{self.subject:02d}.npy'
                )
                metadata_dict = np.load(metadata_dir, allow_pickle=True).item()
                self.selected_lh_vertices = metadata_dict['fmri']\
                    ['lh_fsaverage_rois'][self.roi]
                self.selected_rh_vertices = metadata_dict['fmri']\
                    ['rh_fsaverage_rois'][self.roi]
            # Select vertices based on one-hot encoded vector only if the ROI is
            # not provided
            else:
                # If selected vertices is not set, use all vertice
                if self.selected_lh_vertices is None:
                        self.selected_lh_vertices = range(self.VERTICES_LENGTH)
                if self.selected_rh_vertices is None:
                        self.selected_rh_vertices = range(self.VERTICES_LENGTH)

            # Load the vision transformer
            self.feature_extractor = self._load_feature_extractor(self.device)

            # Define the image preprocessing transform
            self.transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()

            # Load the scaler, PCA, and trained regression weights
            self.scaler, self.pca, self.lh_reg, self.rh_reg = \
                self._load_encoding_weights()

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
        model = torchvision.models.vit_b_32(weights='DEFAULT')
        
        # Select the used layers for feature extraction
        model_layers = [
            'encoder.layers.encoder_layer_0.add_1',
            'encoder.layers.encoder_layer_1.add_1',
            'encoder.layers.encoder_layer_2.add_1',
            'encoder.layers.encoder_layer_3.add_1',
            'encoder.layers.encoder_layer_4.add_1',
            'encoder.layers.encoder_layer_5.add_1',
            'encoder.layers.encoder_layer_6.add_1',
            'encoder.layers.encoder_layer_7.add_1',
            'encoder.layers.encoder_layer_8.add_1',
            'encoder.layers.encoder_layer_9.add_1',
            'encoder.layers.encoder_layer_10.add_1',
            'encoder.layers.encoder_layer_11.add_1'
        ]
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
        weights_dir = os.path.join(
            self.nest_dir, 'encoding_models', 'modality-fmri',
            'train_dataset-nsd_fsaverage', 'model-vit_b_32',
            'encoding_models_weights', 'weights_subject-'+
            format(self.subject, '02')+'.npy'
        )
        weights = np.load(weights_dir, allow_pickle=True).item()

        # Scaler
        scaler = StandardScaler()
        scaler.scale_ = weights['scaler_param']['scale_']
        scaler.mean_ = weights['scaler_param']['mean_']
        scaler.var_ = weights['scaler_param']['var_']
        scaler.n_features_in_ = weights['scaler_param']['n_features_in_']
        scaler.n_samples_seen_ = weights['scaler_param']['n_samples_seen_']

        # PCA
        pca = PCA(n_components=250, random_state=20200220)
        pca.components_ = weights['pca_param']['components_']
        pca.explained_variance_ = weights['pca_param']['explained_variance_']
        pca.explained_variance_ratio_ = weights['pca_param']['explained_variance_ratio_']
        pca.singular_values_ = weights['pca_param']['singular_values_']
        pca.mean_ = weights['pca_param']['mean_']
        pca.n_components_ = weights['pca_param']['n_components_']
        pca.n_samples_ = weights['pca_param']['n_samples_']
        pca.noise_variance_ = weights['pca_param']['noise_variance_']
        pca.n_features_in_ = weights['pca_param']['n_features_in_']

        # LH linear regression parameters
        lh_reg = LinearRegression()
        lh_reg.coef_ = weights['lh_reg_param']['coef_'][self.selected_lh_vertices]
        lh_reg.intercept_ = weights['lh_reg_param']['intercept_'][self.selected_lh_vertices]
        lh_reg.n_features_in_ = weights['lh_reg_param']['n_features_in_']

        # RH linear regression parameters
        rh_reg = LinearRegression()
        rh_reg.coef_ = weights['rh_reg_param']['coef_'][self.selected_rh_vertices]
        rh_reg.intercept_ = weights['rh_reg_param']['intercept_'][self.selected_rh_vertices]
        rh_reg.n_features_in_ = weights['rh_reg_param']['n_features_in_']

        return scaler, pca, lh_reg, rh_reg


    def generate_response(
            self, 
            stimulus: np.ndarray,
            show_progress: bool = True) -> np.ndarray:
        """
        Generate in silico fMRI responses for a batch of images.

        Parameters
        ----------
        stimulus : np.ndarray
            Images for which the in silico neural responses are generated. Must be
            a 4-D numpy array of shape (Batch size x 3 RGB Channels x Width x
            Height) consisting of integer values in the range [0, 255].
            Furthermore, the images must be of square size (i.e., equal width and
            height).
        show_progress : bool, default=True
            Whether to display a progress bar during encoding.

        Returns
        -------
        (lh_insilico_fmri, rh_insilico_fmri) : tuple of np.ndarray
            LH and RH in silico fMRI response array, each with with shape
            (batch_size, n_vertices), where the number of vertices depends on
            The selection parameter.
        """

        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
            raise StimulusError(
                "Stimulus must be a 4D numpy array (batch, channels, height, width)"
            )

        # Preprocess the images
        images = self.transform(torch.from_numpy(stimulus))

        # Extract features and generate responses in batches
        batch_size = 100
        n_batches = int(np.ceil(len(images) / batch_size))

        if show_progress:
            progress_bar = tqdm(range(n_batches), desc='Encoding fMRI responses')
        else:
            progress_bar = range(n_batches)

        lh_insilico_fmri = None
        rh_insilico_fmri = None

        with torch.no_grad():
            for b in progress_bar:
                # Image batch indices
                idx_start = b * batch_size
                idx_end = idx_start + batch_size

                # Extract features
                img_batch = images[idx_start:idx_end].to(self.device)
                features = self.feature_extractor(img_batch)

                # Flatten features
                features = torch.hstack([torch.flatten(l, start_dim=1) for l in features.values()])
                features = features.detach().cpu().numpy()

                # Process features
                features = self.scaler.transform(features)
                features = self.pca.transform(features)
                features = features.astype(np.float32)

                # Generate the in silico fMRI responses
                lh_insilico_fmri_batch = self.lh_reg.predict(features).astype(np.float32)
                rh_insilico_fmri_batch = self.rh_reg.predict(features).astype(np.float32)

                # Combine with previous batches
                if lh_insilico_fmri is None:
                    lh_insilico_fmri = lh_insilico_fmri_batch
                    rh_insilico_fmri = rh_insilico_fmri_batch
                else:
                    lh_insilico_fmri = np.append(
                        lh_insilico_fmri,
                        lh_insilico_fmri_batch,
                        axis=0
                    )
                    rh_insilico_fmri = np.append(
                        rh_insilico_fmri,
                        rh_insilico_fmri_batch,
                        axis=0
                    )

                if show_progress and isinstance(progress_bar, tqdm):
                    encoded_images = min((b + 1) * batch_size, len(images))
                    progress_bar.set_postfix({
                        'Encoded images': encoded_images, 
                        'Total images': len(images)
                    })

        return (lh_insilico_fmri, rh_insilico_fmri)


    @classmethod
    def get_metadata(cls, nest_dir=None, subject=None, model_instance=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        nest_dir : str
            Path to NEST directory.
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
            nest_dir = model_instance.nest_dir
            subject = model_instance.subject

        # If this method is called on an instance (rather than the class)
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            nest_dir = cls.nest_dir
            subject = cls.subject

        # Validate required parameters
        missing_params = []
        if nest_dir is None: missing_params.append('nest_dir')
        if subject is None: missing_params.append('subject')

        if missing_params:
            raise InvalidParameterError(f"Required parameters missing: {', '.join(missing_params)}")

        # Validate parameters
        validate_subject(subject, cls.VALID_SUBJECTS)

        # Build metadata path
        file_name = os.path.join(nest_dir,
                            'encoding_models', 
                            'modality-fmri',
                            'train_dataset-nsd_fsaverage', 
                            'model-vit_b_32', 
                            'metadata',
                            f'metadata_subject-{subject:02d}.npy')

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