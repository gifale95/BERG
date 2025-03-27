import os
from copy import deepcopy
from typing import Any, Dict, Optional
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
from nest.core.model_registry import register_model
from nest.interfaces.base_model import BaseModelInterface


# Load model metadata from YAML
def load_model_metadata():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "eeg_things_eeg_2_vit_b_32.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load metadata once at the top
metadata = load_model_metadata()

# Register this model with the registry using metadata
register_model(
    model_id=metadata["model_id"],
    version=metadata["version"],
    module_path="nest.models.eeg.things_eeg",
    class_name="EEGEncodingModel",
    modality=metadata.get("modality", "eeg"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "eeg_things_eeg_2_vit_b_32.yaml")
)


class EEGEncodingModel(BaseModelInterface):
    """
    EEG encoding model using a vision transformer backbone to generate
    in silico EEG responses for the THINGS-EEG-2 dataset.
    """
    
    MODEL_ID = metadata["model_id"]
    VALID_SUBJECTS = metadata["supported_parameters"]["subject"]["valid_values"]
    
    def __init__(self, subject: int, nest_dir: Optional[str] = None):
        """
        Initialize the EEG encoding model.

        Args:
            subject (int): Subject number from the THINGS-EEG-2 dataset.
            nest_dir (Optional[str]): Root path to the NEST directory.
        """
        self.subject = subject
        self.nest_dir = nest_dir
        self.model = None
        self._validate_parameters()
        
    def _validate_parameters(self):
        """
        Validate user-provided parameters against supported metadata.
        """
        if self.subject not in self.VALID_SUBJECTS:
            raise InvalidParameterError(
                f"Subject must be one of {self.VALID_SUBJECTS}, got {self.subject}"
            )

    def load_model(self, device: str = "auto") -> None:
        """
        Load model weights, preprocessing pipeline, and regression layers.

        Args:
            device (str): Target device ("cpu", "cuda", or "auto").
        """
        try:
            # Select device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            # Get the EEG channels and time points dimensions
            metadata_dir = os.path.join(
                self.nest_dir, 'encoding_models', 'modality-eeg',
                'train_dataset-things_eeg_2', 'model-vit_b_32',
                'metadata', f'metadata_sub-{self.subject:02d}.npy'
            )
            metadata_dict = np.load(metadata_dir, allow_pickle=True).item()
            self.ch_names = metadata_dict['eeg']['ch_names']
            self.times = metadata_dict['eeg']['times']

            # Load the vision transformer
            self.feature_extractor = self._load_feature_extractor(device)
            
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

        Args:
            device (str): Computation device ("cpu" or "cuda").

        Returns:
            torch.nn.Module: Torch feature extractor model in eval mode.
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

        Returns:
            Tuple[StandardScaler, PCA]: Fitted scaler and PCA models.
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

        Returns:
            List[LinearRegression]: List of scikit-learn regression models.
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

        Args:
            stimulus (np.ndarray): Input stimulus of shape (B, C, H, W).
            show_progress (bool): Whether to display a progress bar.

        Returns:
            np.ndarray: EEG response array of shape (B, R, C, T), where:
                B = batch size,
                R = number of repetitions,
                C = number of EEG channels,
                T = number of time points.
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
                    # Generate the in silico EEG responses
                    insilico_eeg = reg.predict(features)
                    insilico_eeg = insilico_eeg.astype(np.float32)
                    
                    # Reshape to (Images x Channels x Time)
                    insilico_eeg = np.reshape(
                        insilico_eeg, 
                        (len(insilico_eeg), len(self.ch_names), len(self.times))
                    )
                    insilico_eeg_part.append(insilico_eeg)
                
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
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Load and return EEG metadata for the current subject.

        Returns:
            Dict[str, Any]: EEG metadata dictionary (e.g., channel names, time points).
        """

        file_name = os.path.join(self.nest_dir, 
                                 'encoding_models', 
                                 'modality-eeg',
                                 'train_dataset-things_eeg_2', 
                                 'model-vit_b_32', 
                                 'metadata',
                                 'metadata_sub-' + format(self.subject,'02') + '.npy')

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
        Release GPU memory and unload the feature extractor.
        """
        if hasattr(self, 'feature_extractor'):
            # Free GPU memory if using CUDA
            if hasattr(self.feature_extractor, 'to'):
                self.feature_extractor.to('cpu')
            
            self.feature_extractor = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()