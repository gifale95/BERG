import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import yaml
import torchextractor as tx
from tqdm import tqdm
from torchvision import transforms as trn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
)
from berg.core.model_registry import register_model
from berg.interfaces.base_model import BaseModelInterface


# Load model info from YAML
def load_model_info():
    yaml_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "model_cards", 
        "meg-things_meg_1-vit_b_32.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.meg.things_meg_1_vit_b_32",
    class_name="MEGEncodingModel",
    modality=model_info.get("modality", "MEG"),
    training_dataset=model_info.get("training_dataset", "things_meg_1"),
    yaml_path=os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "model_cards", 
        "meg-things_meg_1-vit_b_32.yaml"
    )
)


class MEGEncodingModel(BaseModelInterface):
    """
    MEG encoding model using vision transformer to generate
    in silico MEG responses for the THINGS meg_1 dataset.
    """
    
    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    VALID_REGIONS = model_info["parameters"]["selection"]["properties"]["region"]["valid_values"]
    VALID_SENSOR_PREFIXES = model_info["parameters"]["selection"]["properties"]["sensors"]["valid_values"]
    SENSORS_LENGTH = 271
    TIMEPOINTS_LENGTH = 281
    
    def __init__(
        self, 
        subject: str, 
        device: str = "auto", 
        selection: Optional[Dict] = None, 
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the MEG encoding model.
        
        Parameters
        ----------
        subject : str
            Subject ID from the THINGS MEG1 dataset. Must be "P1", "P2", "P3", or "P4".
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.    
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            Can include specific regions, sensor prefixes, sensor indices, and/or timepoints.
            - region: List of anatomical regions (Central, Frontal, Occipital, Parietal, Temporal)
            - sensors: List of sensor prefix codes (MLC, MLF, MLO, etc.)
            - sensor_index: Binary one-hot encoded vector for sensor selection
            - timepoints: Binary one-hot encoded vector for timepoint selection
        berg_dir : str, optional
            Root path to the BERG directory containing model files and weights.
        """
        # Assign Parameters
        self.subject = subject
        self.berg_dir = berg_dir
        self.model = None
        
        # Parameters from selection
        self.selection = selection
        self.selected_regions = None
        self.selected_sensor_prefixes = None
        self.selected_sensors = None
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
            
            # Validate regions
            if "region" in self.selection:
                region_list = self.selection["region"]
                if not isinstance(region_list, list):
                    raise InvalidParameterError("Region must be provided as a list")
                
                invalid_regions = [r for r in region_list if r not in self.VALID_REGIONS]
                if invalid_regions:
                    raise InvalidParameterError(
                        f"Invalid region(s): {invalid_regions}. "
                        f"Valid regions are: {self.VALID_REGIONS}"
                    )
                self.selected_regions = region_list
            
            # Validate sensor prefixes
            if "sensors" in self.selection:
                sensor_prefix_list = self.selection["sensors"]
                if not isinstance(sensor_prefix_list, list):
                    raise InvalidParameterError("Sensors must be provided as a list")
                
                invalid_prefixes = [s for s in sensor_prefix_list if s not in self.VALID_SENSOR_PREFIXES]
                if invalid_prefixes:
                    raise InvalidParameterError(
                        f"Invalid sensor prefix(es): {invalid_prefixes}. "
                        f"Valid sensor prefixes are: {self.VALID_SENSOR_PREFIXES}"
                    )
                self.selected_sensor_prefixes = sensor_prefix_list
            
            # Validate sensor indices
            if "sensor_index" in self.selection:
                sensor_array = validate_binary_array(
                    self.selection["sensor_index"],
                    self.SENSORS_LENGTH,
                    "sensor_index"
                )
                self.selected_sensors = get_selected_indices(sensor_array)
            
            # Validate timepoints
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
        (scalers, PCA), and trained regression weights for the specified
        subject. Only loads weights for selected sensors and timepoints
        to optimize memory usage.
        """
        try:
            # Load metadata
            metadata_dir = os.path.join(
                self.berg_dir, 
                'encoding_models',
                'modality-meg',
                'train_dataset-things_meg_1',
                'model-vit_b_32',
                'metadata',
                f'metadata_P{self.subject}.npy'
            )
            self.metadata = np.load(metadata_dir, allow_pickle=True).item()
            
            # Build sensor selection from multiple sources
            sensor_indices_set = set()
            
            # Add sensors from region selection
            if self.selected_regions is not None:
                for region in self.selected_regions:
                    # Find sensors matching this region
                    region_mask = self.metadata['sensors']['sensor_regions'] == region
                    region_sensors = np.where(region_mask)[0]
                    sensor_indices_set.update(region_sensors.tolist())
            
            # Add sensors from prefix selection
            if self.selected_sensor_prefixes is not None:
                for prefix in self.selected_sensor_prefixes:
                    # Find sensors matching this prefix
                    prefix_mask = self.metadata['sensors']['sensor_prefixes'] == prefix
                    prefix_sensors = np.where(prefix_mask)[0]
                    sensor_indices_set.update(prefix_sensors.tolist())
            
            # Add sensors from direct index selection
            if self.selected_sensors is not None:
                sensor_indices_set.update(self.selected_sensors)
            
            # If any selection was made, use the combined set
            if sensor_indices_set:
                self.selected_sensors = sorted(list(sensor_indices_set))
            else:
                # If no selection made, use all sensors
                self.selected_sensors = list(range(self.SENSORS_LENGTH))
            
            # If no timepoints selected, use all
            if self.selected_timepoints is None:
                self.selected_timepoints = list(range(self.TIMEPOINTS_LENGTH))
            
            # Load the vision transformer
            self.feature_extractor = self._load_feature_extractor(self.device)
            
            # Load the scalers, PCA, and trained regression weights (only for selection)
            self.scaler, self.pca, self.reg = self._load_encoding_weights()
            
            print(f"Model loaded on {self.device} for subject {self.subject}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def _load_feature_extractor(self, device):
        """
        Load the ViT feature extractor for all 12 transformer layers.
        
        Parameters
        ----------
        device : str
            Computation device ("cpu" or "cuda").
        
        Returns
        -------
        tx.Extractor
            Torchextractor wrapped model configured to extract 
            representations from all 12 transformer layers.
        """
        # Load ViT model
        weights = ViT_B_32_Weights.DEFAULT
        model = vit_b_32(weights=weights)
        model.to(device)
        
        if device == 'cuda':
            model = model.float()
        
        model.eval()

        
        # Define layer names for all 12 transformer blocks
        layer_names = ['encoder.layers.encoder_layer_0',
                            'encoder.layers.encoder_layer_1',
                            'encoder.layers.encoder_layer_2',
                            'encoder.layers.encoder_layer_3',
                            'encoder.layers.encoder_layer_4',
                            'encoder.layers.encoder_layer_5',
                            'encoder.layers.encoder_layer_6',
                            'encoder.layers.encoder_layer_7',
                            'encoder.layers.encoder_layer_8',
                            'encoder.layers.encoder_layer_9',
                            'encoder.layers.encoder_layer_10',
                            'encoder.layers.encoder_layer_11']
        
        # Wrap the visual encoder with torchextractor
        feature_extractor = tx.Extractor(model, layer_names)
        
        self.transform = weights.transforms()
        
        return feature_extractor
    
    def _load_encoding_weights(self):
        """
        Load pretrained scaler, PCA, and regression weights.
        Only loads regression weights for selected sensors and timepoints.
        
        Returns
        -------
        tuple
            A tuple containing (scaler, pca, reg) where:
            - scaler : StandardScaler - Pre-PCA feature normalization
            - pca : PCA - Fitted principal component analysis model
            - reg : LinearRegression - Model with only selected weights
        """
        # Load all weights
        weights_dir = os.path.join(
            self.berg_dir, 
            'encoding_models', 
            'modality-meg',
            'train_dataset-things_meg_1', 
            'model-vit_b_32',
            'encoding_models_weights',
            f'weights_P{self.subject}.npy'
        )
        weights = np.load(weights_dir, allow_pickle=True).item()
        
        # Reconstruct pre-PCA scaler
        scaler = StandardScaler()
        scaler.scale_ = weights['scaler_param']['scale_']
        scaler.mean_ = weights['scaler_param']['mean_']
        scaler.var_ = weights['scaler_param']['var_']
        scaler.n_features_in_ = weights['scaler_param']['n_features_in_']
        scaler.n_samples_seen_ = weights['scaler_param']['n_samples_seen_']
        
        # Reconstruct PCA
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
        
        # Load regression weights
        reg_coef_full = weights['reg_param']['coef_']  # Shape: (n_sensors * n_times, n_features)
        reg_intercept_full = weights['reg_param']['intercept_']  # Shape: (n_sensors * n_times,)
        
        # Calculate indices for selected outputs
        selected_indices = []
        for sensor_idx in self.selected_sensors:
            for time_idx in self.selected_timepoints:
                flat_idx = sensor_idx * self.TIMEPOINTS_LENGTH + time_idx
                selected_indices.append(flat_idx)
        
        # Extract only the weights for selected outputs
        coef_subset = reg_coef_full[selected_indices, :]  # Shape: (n_selected_outputs, n_features)
        intercept_subset = reg_intercept_full[selected_indices]  # Shape: (n_selected_outputs,)
        
        # Build regression model with only selected weights
        reg = LinearRegression()
        reg.coef_ = coef_subset
        reg.intercept_ = intercept_subset
        reg.n_features_in_ = weights['reg_param']['n_features_in_']
        
        return scaler, pca, reg
    
    def generate_response(
        self,
        stimulus: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate in silico MEG responses for a batch of images.
        
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
        insilico_meg_responses : np.ndarray
            In silico MEG response array of shape (batch_size, n_sensors, 
            n_timepoints), where the number of sensors and time points
            depends on the selection parameter.
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
            progress_bar = tqdm(range(n_batches), desc='Encoding MEG responses')
        else:
            progress_bar = range(n_batches)
        
        insilico_meg_responses = None
        
        with torch.no_grad():
            for b in progress_bar:
                # Image batch indices
                idx_start = b * batch_size
                idx_end = min(idx_start + batch_size, len(images))
                
                # Extract features
                img_batch = images[idx_start:idx_end].to(self.device)
                _, features = self.feature_extractor(img_batch)
                
                # Extract all tokens from each layer and concatenate
                batch_features = []
                for layer_name in features.keys():
                    layer_features = features[layer_name]
                    # ViT output: (batch_size, n_patches, hidden_dim)
                    # Flatten all tokens: (batch_size, n_patches * hidden_dim)
                    layer_flat = layer_features.flatten(1, 2)
                    batch_features.append(layer_flat)
                
                # Concatenate features from all layers
                # Shape: (batch_size, 12_layers * 50_patches * 768_dim)
                ft = torch.cat(batch_features, dim=-1)
                ft = ft.detach().cpu().numpy()
                
                # Process features through scaler and PCA
                ft = self.scaler.transform(ft)
                ft = self.pca.transform(ft)
                ft = ft.astype(np.float32)
                
                # Generate predictions with model
                batch_pred = self.reg.predict(ft)

                # Reshape to (batch_size, n_sensors, n_timepoints)
                batch_responses = batch_pred.reshape(
                    batch_pred.shape[0],
                    len(self.selected_sensors),
                    len(self.selected_timepoints)
                )
                
                batch_responses = batch_responses.astype(np.float32)
                
                # Combine with previous batches
                if insilico_meg_responses is None:
                    insilico_meg_responses = batch_responses
                else:
                    insilico_meg_responses = np.append(
                        insilico_meg_responses,
                        batch_responses,
                        axis=0
                    )
                
                if show_progress and isinstance(progress_bar, tqdm):
                    encoded_images = min((b + 1) * batch_size, len(images))
                    progress_bar.set_postfix({
                        'Encoded images': encoded_images,
                        'Total images': len(images)
                    })
        
        return insilico_meg_responses
    
    @classmethod
    def get_metadata(
        cls, 
        berg_dir=None, 
        subject=None, 
        model_instance=None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        berg_dir : str
            Path to BERG directory.
        subject : str
            Subject ID (1,2,3,4).
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
        if berg_dir is None:
            missing_params.append('berg_dir')
        if subject is None:
            missing_params.append('subject')
        
        if missing_params:
            raise InvalidParameterError(
                f"Required parameters missing: {', '.join(missing_params)}"
            )
        
        # Validate parameters
        validate_subject(subject, cls.VALID_SUBJECTS)
        
        # Build metadata path
        file_name = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-meg',
            'train_dataset-things_meg_1',
            'model-vit_b_32',
            'metadata',
            f'metadata_P{subject}.npy'
        )
        
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
            # The feature extractor is a torchextractor wrapper
            # Clear the underlying model
            if hasattr(self.feature_extractor, 'model'):
                if hasattr(self.feature_extractor.model, 'to'):
                    self.feature_extractor.model.to('cpu')
            
            self.feature_extractor = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()