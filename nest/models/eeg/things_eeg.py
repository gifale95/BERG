import os
from copy import deepcopy
from typing import Any, Dict, Optional, List, Optional, Union
import numpy as np
import torch
import torchvision
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from nest.core.exceptions import (
    InvalidParameterError,
    ModelLoadError,
    StimulusError,
)
from nest.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_channels,
    validate_binary_array,
    get_selected_indices
)
from nest.core.model_registry import register_model
from nest.interfaces.base_model import BaseModelInterface


# Load model info from YAML
def load_model_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "eeg-things_eeg_2-vit_b_32.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="nest.models.eeg.things_eeg",
    class_name="EEGEncodingModel",
    modality=model_info.get("modality", "eeg"),
    dataset=model_info.get("dataset", "things_eeg_2"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "eeg-things_eeg_2-vit_b_32.yaml")
)


class EEGEncodingModel(BaseModelInterface):
    """
    EEG encoding model using a vision transformer backbone to generate
    in silico EEG responses for the THINGS-EEG-2 dataset.
    """
    
    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    VALID_CHANNELS = model_info["parameters"]["selection"]["properties"]["channels"]["valid_values"]
    TIMEPOINTS_LENGTH = 140
    
    def __init__(self, subject: int, device: str = "auto", selection: Optional[Dict] = None, nest_dir: Optional[str] = None):
        """
        Initialize the EEG encoding model.
        
        Parameters
        ----------
        subject : int
            Subject number from the THINGS-EEG-2 dataset.
            Must be one of the valid subject IDs (1-10).
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.    
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            Can include specific channels and/or timepoints.
            - channels: List of EEG channel names to include in the output
            - timepoints: Binary one-hot encoded vector indicating which timepoints to include
        nest_dir : str, optional
            Root path to the NEST directory containing model files and weights.
        """
        # Assign Parameters
        self.subject = subject
        self.nest_dir = nest_dir
        self.model = None
        
        # Parameters from selection
        self.selection = selection
        self.selected_channels = None
        self.selected_timepoints = None
        
        # Validate parameters
        self._validate_parameters()
        
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def _validate_parameters(self):
        """
        Validate user-provided parameters against supported model yaml.
        
        Ensures that subject IDs and other parameters match the expected
        values defined in the model's yaml.
        """

        # Validate subject
        validate_subject(self.subject, self.VALID_SUBJECTS)
        
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Individual validations
            if "channels" in self.selection:
                self.selected_channels = validate_channels(
                    self.selection["channels"], self.VALID_CHANNELS
                )

            if "timepoints" in self.selection:
                timepoints_array = validate_binary_array(
                    self.selection["timepoints"],
                    self.TIMEPOINTS_LENGTH,
                    "timepoints"
                )
                self.selected_timepoints = get_selected_indices(timepoints_array)

    def load_model(self) -> None:
        """
        Load model weights, preprocessing pipeline, and regression layers.
        
        Loads the vision transformer backbone, preprocessing components 
        (scaler, PCA), and trained regression weights for the specified
        subject. Sets up all necessary components for generating EEG
        responses.
        """
        try:
            # Get the EEG channels and time points dimensions
            metadata_dir = os.path.join(
                self.nest_dir, 'encoding_models', 'modality-eeg',
                'train_dataset-things_eeg_2', 'model-vit_b_32',
                'metadata', f'metadata_sub-{self.subject:02d}.npy'
            )
            metadata_dict = np.load(metadata_dir, allow_pickle=True).item()
            self.ch_names = metadata_dict['eeg']['ch_names']
            self.times = metadata_dict['eeg']['times']
            
            # If selected_channels is set, store the indices
            if self.selected_channels is not None:
                self.channel_indices = [self.ch_names.index(ch) for ch in self.selected_channels]
            else:
                # If no channels selected, use all channels
                self.channel_indices = range(len(self.ch_names))
            
            # If selected_timepoints is not set, use all timepoints
            if self.selected_timepoints is None:
                self.selected_timepoints = range(len(self.times))

            # Load the vision transformer
            self.feature_extractor = self._load_feature_extractor(self.device)
            
            # Load the scaler and PCA weights
            self.scaler, self.pca = self._load_scaler_and_pca()
            
            # Define the image preprocessing transform
            self.transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
            
            # Load the trained regression weights
            self.regression_weights = self._load_regression_weights()
            
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
    
    def _load_scaler_and_pca(self):
        """
        Load pretrained scaler and PCA transformation parameters.
        
        Loads and configures StandardScaler and PCA models with
        pre-computed parameters for feature normalization and
        dimensionality reduction.
        
        Returns
        -------
        tuple
            A tuple containing (scaler, pca) where:
            - scaler : StandardScaler - Fitted feature normalization object
            - pca : PCA - Fitted principal component analysis model
        """
        # Scaler
        weights_dir = os.path.join(
            self.nest_dir, 'encoding_models', 'modality-eeg',
            'train_dataset-things_eeg_2', 'model-vit_b_32',
            'encoding_models_weights', 'StandardScaler_param.npy'
        )
        scaler_weights = np.load(weights_dir, allow_pickle=True).item()
        scaler = StandardScaler()
        scaler.scale_ = scaler_weights['scale_']
        scaler.mean_ = scaler_weights['mean_']
        scaler.var_ = scaler_weights['var_']
        scaler.n_features_in_ = scaler_weights['n_features_in_']
        scaler.n_samples_seen_ = scaler_weights['n_samples_seen_']
        
        # PCA
        weights_dir = os.path.join(
            self.nest_dir, 'encoding_models', 'modality-eeg',
            'train_dataset-things_eeg_2', 'model-vit_b_32',
            'encoding_models_weights', 'pca_param.npy'
        )
        pca_weights = np.load(weights_dir, allow_pickle=True).item()
        pca = PCA(n_components=1000, random_state=20200220)
        pca.components_ = pca_weights['components_']
        pca.explained_variance_ = pca_weights['explained_variance_']
        pca.explained_variance_ratio_ = pca_weights['explained_variance_ratio_']
        pca.singular_values_ = pca_weights['singular_values_']
        pca.mean_ = pca_weights['mean_']
        pca.n_components_ = pca_weights['n_components_']
        pca.n_samples_ = pca_weights['n_samples_']
        pca.noise_variance_ = pca_weights['noise_variance_']
        pca.n_features_in_ = pca_weights['n_features_in_']
        
        return scaler, pca
    
    def _load_regression_weights(self):
        """
        Load trained linear regression models for each EEG repetition.
        
        Loads the weights for the linear mapping from visual features
        to EEG responses, with separate models for each repetition.
        
        Returns
        -------
        list
            List of scikit-learn LinearRegression models, one per repetition.
        """
        weights_dir = os.path.join(
            self.nest_dir, 'encoding_models', 'modality-eeg',
            'train_dataset-things_eeg_2', 'model-vit_b_32',
            'encoding_models_weights', f'LinearRegression_param_sub-{self.subject:02d}.npy'
        )
        reg_weights = np.load(weights_dir, allow_pickle=True).item()
        
        regression_weights = []
        for r in range(len(reg_weights)):
            reg = LinearRegression()
            reg.coef_ = reg_weights[f'rep-{r+1}']['coef_']
            reg.intercept_ = reg_weights[f'rep-{r+1}']['intercept_']
            reg.n_features_in_ = reg_weights[f'rep-{r+1}']['n_features_in_']
            regression_weights.append(deepcopy(reg))
        
        return regression_weights

    def generate_response(
            self, 
            stimulus: np.ndarray,
            show_progress: bool = True) -> np.ndarray:
        """
        Generate in silico EEG responses for a batch of visual stimuli.
        
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
        np.ndarray
            EEG response array with shape (batch_size, n_repetitions, n_selected_channels, n_selected_timepoints).
            The dimensions will be adapted based on the selection parameter:
            - If channels are selected, only those channels are included
            - If timepoints are selected, only those timepoints are included
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
            progress_bar = tqdm(range(n_batches), desc='Encoding EEG responses')
        else:
            progress_bar = range(n_batches)
            
        insilico_eeg_responses = None
        
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
                features = features[:, :250]  # Only use first 250 principal components
                features = features.astype(np.float32)
                
                # Generate responses for each repetition
                insilico_eeg_part = []
                for reg in self.regression_weights:
                    # Generate the in silico EEG responses for all channels and timepoints
                    insilico_eeg = reg.predict(features)
                    insilico_eeg = insilico_eeg.astype(np.float32)
                    
                    # Reshape to (Images x Channels x Time)
                    insilico_eeg = np.reshape(
                        insilico_eeg, 
                        (len(insilico_eeg), len(self.ch_names), len(self.times))
                    )
                    
                    # Extract only the selected channels and timepoints
                    insilico_eeg = insilico_eeg[:, self.channel_indices, :]
                    insilico_eeg = insilico_eeg[:, :, self.selected_timepoints]
                    
                    insilico_eeg_part.append(np.squeeze(insilico_eeg))
                
                # Reshape to (Images x Repeats x Channels x Time)
                batch_responses = np.swapaxes(np.asarray(insilico_eeg_part), 0, 1)
                batch_responses = batch_responses.astype(np.float32)
                
                # Combine with previous batches
                if insilico_eeg_responses is None:
                    insilico_eeg_responses = batch_responses
                else:
                    insilico_eeg_responses = np.append(
                        insilico_eeg_responses, 
                        batch_responses, 
                        axis=0
                    )
                
                if show_progress and isinstance(progress_bar, tqdm):
                    encoded_images = min((b + 1) * batch_size, len(images))
                    progress_bar.set_postfix({
                        'Encoded images': encoded_images, 
                        'Total images': len(images)
                    })
        
        return insilico_eeg_responses
    
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
                                'modality-eeg',
                                'train_dataset-things_eeg_2', 
                                'model-vit_b_32', 
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
        Release GPU memory and unload the feature extractor.
        
        Frees GPU memory by moving models to CPU and clearing CUDA cache
        if available, preventing memory leaks when working with multiple
        models.
        """
        if hasattr(self, 'feature_extractor'):
            # Free GPU memory if using CUDA
            if hasattr(self.feature_extractor, 'to'):
                self.feature_extractor.to('cpu')
            
            self.feature_extractor = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()