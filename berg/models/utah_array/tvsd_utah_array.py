import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import yaml
import torchextractor as tx
from tqdm import tqdm
from torchvision import transforms as trn
from torchvision.models import vit_b_32, ViT_B_32_Weights
import timm
import joblib
import torchvision
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
    get_selected_indices
)
from berg.core.model_registry import register_model
from berg.interfaces.base_model import BaseModelInterface


# Load model info from YAML
def load_model_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "utah_array-tvsd-vit_b_32.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.utah_array.tvsd_utah_array",
    class_name="UtahArrayEncodingModel",
    modality=model_info.get("modality", "utah_array"),
    training_dataset=model_info.get("training_dataset", "tvsd"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "utah_array-tvsd-vit_b_32.yaml")
)


class UtahArrayEncodingModel(BaseModelInterface):
    """
    Utah Array encoding model using vision transformer to generate
    in silico spiking responses for the TVSD dataset.
    """
    
    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    ELECTRODES_LENGTH = 1024
    TIMEPOINTS_LENGTH = 300
    N_CHUNKS = 8
    ELECTRODES_PER_CHUNK = 128
    
    def __init__(
        self, 
        subject: str, 
        device: str = "auto", 
        selection: Optional[Dict] = None, 
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the Utah Array encoding model.
        
        Parameters
        ----------
        subject : str
            Monkey ID from the TVSD dataset. Must be "N" or "F".
        device : str, default="auto"
            Target device for computation. Options are "cpu", "cuda", or "auto".
            If "auto", will use GPU if available, otherwise CPU.    
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            Can include specific ROIs, electrodes, and/or timepoints.
            - roi: List of brain areas to include (V1, V4, IT)
            - electrodes: Binary one-hot encoded vector for electrode selection
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
        self.selected_rois = None
        self.selected_electrodes = None
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
            
            # Validate ROIs
            if "roi" in self.selection:
                roi_list = self.selection["roi"]
                if not isinstance(roi_list, list):
                    raise InvalidParameterError("ROI must be provided as a list")
                
                invalid_rois = [r for r in roi_list if r not in self.VALID_ROIS]
                if invalid_rois:
                    raise InvalidParameterError(
                        f"Invalid ROI(s): {invalid_rois}. "
                        f"Valid ROIs are: {self.VALID_ROIS}"
                    )
                self.selected_rois = roi_list
            
            # Validate electrodes
            if "electrodes" in self.selection:
                electrodes_array = validate_binary_array(
                    self.selection["electrodes"],
                    self.ELECTRODES_LENGTH,
                    "electrodes"
                )
                self.selected_electrodes = get_selected_indices(electrodes_array)
            
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
        (scaler, PCA), and trained regression weights for the specified
        subject. Sets up all necessary components for generating EEG
        responses.
        """

        try:
            # Get the EEG channels and time points dimensions
            metadata_dir = os.path.join(
                self.berg_dir, 'encoding_models', 'modality-utah_array',
                'train_dataset-tvsd', 'model-vit_b_32',
                'metadata', f'metadata_monkey{self.subject}.npy'
            )
            self.metadata = np.load(metadata_dir, allow_pickle=True).item()
            
            # Get electrode mapping if ROI selection is specified
            if self.selected_rois is not None:
                if self.selected_electrodes is None:
                    self.selected_electrodes = np.array([])
                electrode_indices = self.metadata['utah_array']['roi_assignments']
                for roi in self.selected_rois:

                    roi_index = list(self.metadata['utah_array']['roi_labels']).index(roi)
                    roi_electrodes = np.where(electrode_indices==roi_index)[0]
                    self.selected_electrodes = np.append(self.selected_electrodes,roi_electrodes).astype(int)
                self.selected_electrodes = sorted(list(set(self.selected_electrodes)))
            
            # If no electrodes selected, use all
            if self.selected_electrodes is None:
                self.selected_electrodes = list(range(self.ELECTRODES_LENGTH))
            
            # If no timepoints selected, use all
            if self.selected_timepoints is None:
                self.selected_timepoints = list(range(self.TIMEPOINTS_LENGTH))
            
            # Load the vision transformer
            self.feature_extractor = self._load_feature_extractor(self.device)        
            
            # Load the scaler, PCA, and trained regression weights
            self.scaler, self.pca, self.chunk_models = self._load_encoding_weights()
            
            print(f"Model loaded on {self.device} for monkey {self.subject}")
            
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
        Load pretrained scaler and PCA transformation parameters.
        Load trained chunked linear regression models.
        
        Returns
        -------
        tuple
            A tuple containing (scaler, pca, chunk_models) where:
            - scaler : StandardScaler - Fitted feature normalization object
            - pca : PCA - Fitted principal component analysis model
            - chunk_models : List of LinearRegression models for each chunk
        """
        # Load preprocessing parameters
        weights_dir = os.path.join(
            self.berg_dir, 
            'encoding_models', 
            'modality-utah_array',
            'train_dataset-tvsd', 
            'model-vit_b_32',
            'encoding_models_weights',
            f'preprocessing_linear_all_monkey{self.subject}.npy'
        )
        preprocessing = np.load(weights_dir, allow_pickle=True).item()
        
        # Reconstruct scaler
        scaler = StandardScaler()
        scaler.scale_ = preprocessing['scaler_param']['scale_']
        scaler.mean_ = preprocessing['scaler_param']['mean_']
        scaler.var_ = preprocessing['scaler_param']['var_']
        scaler.n_features_in_ = preprocessing['scaler_param']['n_features_in_']
        scaler.n_samples_seen_ = preprocessing['scaler_param']['n_samples_seen_']
        
        # Reconstruct PCA
        pca = PCA(n_components=250, random_state=20200220)
        pca.components_ = preprocessing['pca_param']['components_']
        pca.explained_variance_ = preprocessing['pca_param']['explained_variance_']
        pca.explained_variance_ratio_ = preprocessing['pca_param']['explained_variance_ratio_']
        pca.singular_values_ = preprocessing['pca_param']['singular_values_']
        pca.mean_ = preprocessing['pca_param']['mean_']
        pca.n_components_ = preprocessing['pca_param']['n_components_']
        pca.n_samples_ = preprocessing['pca_param']['n_samples_']
        pca.noise_variance_ = preprocessing['pca_param']['noise_variance_']
        pca.n_features_in_ = preprocessing['pca_param']['n_features_in_']
        
        # Load all chunk models
        chunk_models = []
        model_dir = os.path.join(
            self.berg_dir, 
            'encoding_models', 
            'modality-utah_array',
            'train_dataset-tvsd', 
            'model-vit_b_32',
            'encoding_models_weights'
        )
        
        for chunk_idx in range(self.N_CHUNKS):
            model_filename = f'linear_all_chunk_{chunk_idx}_monkey{self.subject}.pkl'
            chunk_model = joblib.load(os.path.join(model_dir, model_filename))
            chunk_models.append(chunk_model)
        
        return scaler, pca, chunk_models


    def generate_response(
        self,
        stimulus: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate in silico Utah Array spiking responses for a batch of images.
        
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
        insilico_spike_responses : np.ndarray
            In silico spiking response array of shape (batch_size, n_timepoints,
            n_electrodes), where the number of electrodes and time points
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
            progress_bar = tqdm(range(n_batches), desc='Encoding Utah Array responses')
        else:
            progress_bar = range(n_batches)
        
        insilico_spike_responses = None
        
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
                # Shape: (batch_size, 13_layers * 196_patches * 768_dim)
                ft = torch.cat(batch_features, dim=-1)
                ft = ft.detach().cpu().numpy()
                
                # Process features through scaler and PCA
                ft = self.scaler.transform(ft)
                ft = self.pca.transform(ft)
                ft = ft.astype(np.float32)
                
                # Generate predictions from all chunks
                chunk_predictions = []
                for chunk_model in self.chunk_models:
                    chunk_pred = chunk_model.predict(ft)
                    chunk_predictions.append(chunk_pred)
                
                # Reshape each chunk: (batch, n_times * electrodes_per_chunk) 
                # -> (batch, n_times, electrodes_per_chunk)
                reshaped_chunks = []
                for chunk_pred in chunk_predictions:
                    reshaped = chunk_pred.reshape(
                        chunk_pred.shape[0], 
                        self.TIMEPOINTS_LENGTH, 
                        self.ELECTRODES_PER_CHUNK
                    )
                    reshaped_chunks.append(reshaped)
                
                # Concatenate along electrode dimension
                # Shape: (batch, n_times, n_electrodes)
                batch_responses = np.concatenate(reshaped_chunks, axis=2)
                
                # Apply electrode selection
                batch_responses = batch_responses[:, :, self.selected_electrodes]
                
                # Apply timepoint selection
                batch_responses = batch_responses[:, self.selected_timepoints, :]
                
                batch_responses = batch_responses.astype(np.float32)
                
                # Combine with previous batches
                if insilico_spike_responses is None:
                    insilico_spike_responses = batch_responses
                else:
                    insilico_spike_responses = np.append(
                        insilico_spike_responses,
                        batch_responses,
                        axis=0
                    )
                
                if show_progress and isinstance(progress_bar, tqdm):
                    encoded_images = min((b + 1) * batch_size, len(images))
                    progress_bar.set_postfix({
                        'Encoded images': encoded_images,
                        'Total images': len(images)
                    })
        
        return insilico_spike_responses


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
            Monkey ID ("N" or "F").
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
            'modality-utah_array',
            'train_dataset-tvsd',
            'model-vit_b_32',
            'metadata',
            f'metadata_monkey{subject}.npy'
        )

        
        # Load metadata if file exists
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found for monkey {subject}")
    
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