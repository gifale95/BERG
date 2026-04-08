import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import yaml
from tqdm import tqdm

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
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards",
                             "ecog-podcast_ecog-gpt2_xl.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)

# Load model_info once at the top
model_info = load_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.ecog.podcast_ecog_gpt2_xl",
    class_name="PodcastECoGEncodingModel",
    modality=model_info.get("modality", "ecog"),
    training_dataset=model_info.get("training_dataset", "podcast_ecog"),
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards",
                           "ecog-podcast_ecog-gpt2_xl.yaml")
)


class PodcastECoGEncodingModel(BaseModelInterface):
    """
    ECoG encoding model using GPT-2 XL contextual word embeddings to generate
    in silico high-gamma responses for the Podcast ECoG dataset.

    The model extracts layer-24 embeddings (1,600-dim) from GPT-2 XL for each
    input word, using the preceding word context. These embeddings are then
    mapped to neural responses via a pre-trained ridge regression, producing
    time-resolved high-gamma predictions at each electrode and time lag
    relative to word onset.
    """

    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    ELECTRODE_COUNTS = model_info["electrode_counts"]
    N_LAGS = 129
    FEATURE_LAYER = 24
    FEATURE_DIM = 1600

    def __init__(
        self,
        subject: str,
        device: str = "auto",
        selection: Optional[Dict] = None,
        context_length: int = 1024,
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the Podcast ECoG encoding model.

        Parameters
        ----------
        subject : str
            Subject ID from the Podcast ECoG dataset. Must be one of
            "01" through "09".
        device : str, default="auto"
            Target device for GPT-2 XL computation. Options are "cpu",
            "cuda", or "auto". If "auto", will use GPU if available.
        selection : dict, optional
            Specifies which outputs to include in the model responses.
            - electrode_index: Binary array for electrode selection
              (length must match subject's electrode count)
            - timepoints: Binary array for time lag selection (length 129)
        context_length : int, default=1024
            Number of preceding tokens to use as context for GPT-2 XL
            feature extraction. Must be between 1 and 1024.
        berg_dir : str, optional
            Root path to the BERG directory containing model files and weights.
        """
        # Assign parameters
        self.subject = subject
        self.context_length = context_length
        self.berg_dir = berg_dir
        self.model = None

        # Parameters from selection
        self.selection = selection
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
        """
        # Validate subject
        validate_subject(self.subject, self.VALID_SUBJECTS)

        # Validate context_length
        if not isinstance(self.context_length, int) or \
                not 1 <= self.context_length <= 1024:
            raise InvalidParameterError(
                f"context_length must be an integer between 1 and 1024, "
                f"got {self.context_length}"
            )

        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Get subject's electrode count for validation
            subject_key = f"sub-{self.subject}"
            n_electrodes = self.ELECTRODE_COUNTS[subject_key]

            # Validate electrode_index
            if "electrode_index" in self.selection:
                electrode_array = validate_binary_array(
                    self.selection["electrode_index"],
                    n_electrodes,
                    "electrode_index"
                )
                self.selected_electrodes = get_selected_indices(electrode_array)

            # Validate timepoints
            if "timepoints" in self.selection:
                timepoints_array = validate_binary_array(
                    self.selection["timepoints"],
                    self.N_LAGS,
                    "timepoints"
                )
                self.selected_timepoints = get_selected_indices(timepoints_array)

    def load_model(self) -> None:
        """
        Load GPT-2 XL, preprocessing scalers, and ridge regression weights.

        Loads the GPT-2 XL language model and tokenizer for feature extraction,
        then loads the trained encoding weights (StandardScaler for X and Y,
        ridge regression coefficients). Only loads regression weights for
        selected electrodes and timepoints to optimize memory usage.
        """
      
        # Load metadata to get electrode/lag dimensions
        metadata_dir = os.path.join(
            self.berg_dir, 'encoding_models', 'modality-ecog',
            'train_dataset-podcast_ecog', 'model-gpt2_xl',
            'metadata', f'metadata_sub-{self.subject}.npy'
        )
        self.metadata = np.load(metadata_dir, allow_pickle=True).item()

        n_electrodes = self.metadata['ecog']['n_electrodes']
        n_lags = self.metadata['ecog']['n_lags']

        # If no electrodes selected, use all
        if self.selected_electrodes is None:
            self.selected_electrodes = list(range(n_electrodes))

        # If no timepoints selected, use all
        if self.selected_timepoints is None:
            self.selected_timepoints = list(range(n_lags))

        # Load GPT-2 XL tokenizer and model
        self._load_language_model()

        # Load encoding weights
        self.scaler_X, self.scaler_Y, self.ridge_coef, self.ridge_intercept = \
            self._load_encoding_weights(n_electrodes, n_lags)

        print(f"Model loaded on {self.device} for subject {self.subject} "
                f"({len(self.selected_electrodes)} electrodes, "
                f"{len(self.selected_timepoints)} lags)")



    def _load_language_model(self):
        """
        Load GPT-2 XL tokenizer and model from HuggingFace transformers.

        The model is loaded in evaluation mode with no gradient computation.
        Uses float16 on CUDA for memory efficiency.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading GPT-2 XL...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        self.language_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        self.language_model.eval()
        self.language_model.to(self.device)

    def _load_encoding_weights(self, n_electrodes, n_lags):
        """
        Load pretrained scaler and ridge regression weights.

        Applies electrode/timepoint selection to load only the required
        subset of regression weights.

        Parameters
        ----------
        n_electrodes : int
            Total number of electrodes for this subject.
        n_lags : int
            Total number of time lags.

        Returns
        -------
        tuple
            (scaler_X, scaler_Y, ridge_coef, ridge_intercept) where:
            - scaler_X : StandardScaler for input features
            - scaler_Y : StandardScaler for neural data (for inverse transform)
            - ridge_coef : ndarray, selected regression coefficients
            - ridge_intercept : ndarray, selected regression intercepts
        """
        weights_dir = os.path.join(
            self.berg_dir,
            'encoding_models',
            'modality-ecog',
            'train_dataset-podcast_ecog',
            'model-gpt2_xl',
            'encoding_models_weights',
            f'weights_sub-{self.subject}.npy'
        )
        weights = np.load(weights_dir, allow_pickle=True).item()

        # Reconstruct scaler for X (features)
        scaler_X = StandardScaler()
        scaler_X.scale_ = weights['scaler_X_param']['scale_']
        scaler_X.mean_ = weights['scaler_X_param']['mean_']
        scaler_X.var_ = weights['scaler_X_param']['var_']
        scaler_X.n_features_in_ = weights['scaler_X_param']['n_features_in_']
        scaler_X.n_samples_seen_ = weights['scaler_X_param']['n_samples_seen_']

        # Reconstruct scaler for Y (neural data, for inverse transform)
        scaler_Y = StandardScaler()
        scaler_Y.scale_ = weights['scaler_Y_param']['scale_']
        scaler_Y.mean_ = weights['scaler_Y_param']['mean_']
        scaler_Y.var_ = weights['scaler_Y_param']['var_']
        scaler_Y.n_features_in_ = weights['scaler_Y_param']['n_features_in_']
        scaler_Y.n_samples_seen_ = weights['scaler_Y_param']['n_samples_seen_']

        # Create masks for selection
        electrode_mask = np.zeros(n_electrodes, dtype=bool)
        electrode_mask[self.selected_electrodes] = True

        time_mask = np.zeros(n_lags, dtype=bool)
        time_mask[self.selected_timepoints] = True

        # Combined mask for flattened neural space (electrodes x lags)
        combined_mask = (electrode_mask[:, None] & time_mask[None, :]).flatten()

        # Slice regression weights to selected subset only
        ridge_coef = weights['ridge_param']['coef_'][:, combined_mask]
        ridge_intercept = weights['ridge_param']['intercept_'][combined_mask]

        # Also slice scaler_Y to match selection
        scaler_Y.scale_ = scaler_Y.scale_[combined_mask]
        scaler_Y.mean_ = scaler_Y.mean_[combined_mask]
        scaler_Y.var_ = scaler_Y.var_[combined_mask]
        scaler_Y.n_features_in_ = int(combined_mask.sum())

        return scaler_X, scaler_Y, ridge_coef, ridge_intercept

    def _extract_features(self, words, show_progress=True):
        """
        Extract GPT-2 XL layer-24 contextual word embeddings.

        Tokenizes the input words, runs them through GPT-2 XL with preceding
        context (up to context_length tokens), and extracts hidden state
        representations from layer 24. When a word is split into multiple
        sub-word tokens, the token embeddings are averaged to produce a single
        word-level embedding.

        Parameters
        ----------
        words : list[str]
            List of words to extract features for.
        show_progress : bool
            Whether to display a progress bar.

        Returns
        -------
        word_embeddings : np.ndarray
            Word-level embeddings of shape (n_words, 1600).
        """
        # Tokenize all words, keeping track of word-to-token mapping
        all_token_ids = []
        word_to_token_indices = []

        # Convention is that each words gets a leading space
        for word in words:
            tokens = self.tokenizer.encode(" " + word)
            start_idx = len(all_token_ids)
            all_token_ids.extend(tokens)
            end_idx = len(all_token_ids)
            word_to_token_indices.append((start_idx, end_idx))  # Records which token belongs to which word

        n_tokens = len(all_token_ids)

        # Build context windows for each token
        # For token i, the input is [max(0, i-context_length) : i+1]
        fill_value = self.tokenizer.pad_token_id or 0
        context_len = self.context_length

        # Extract embeddings for all tokens
        token_embeddings = np.zeros((n_tokens, self.FEATURE_DIM), dtype=np.float32)

        # Process in batches to manage memory
        batch_size = 32

        if show_progress:
            progress = tqdm(range(0, n_tokens, batch_size),
                            desc='Extracting GPT-2 XL features')
        else:
            progress = range(0, n_tokens, batch_size)

        with torch.no_grad():
            for batch_start in progress:
                batch_end = min(batch_start + batch_size, n_tokens)

                batch_inputs = []
                for i in range(batch_start, batch_end):
                    # Get context window: preceding tokens + current token
                    # Total length must not exceed context_len (max 1024 for GPT-2)
                    start = max(0, i - context_len + 1)
                    context = all_token_ids[start:i + 1]

                    # Pad if needed
                    if len(context) < context_len:
                        padding = [fill_value] * (context_len - len(context))
                        context = padding + context

                    batch_inputs.append(context)

                # Convert to tensor
                input_ids = torch.tensor(batch_inputs, dtype=torch.long,
                                         device=self.device)

                # Forward pass
                output = self.language_model(input_ids,
                                             output_hidden_states=True)

                # Extract layer 24 hidden states for the last token position
                # hidden_states[0] is the embedding layer, so layer 24 is index 24
                layer_states = output.hidden_states[self.FEATURE_LAYER]
                embeddings = layer_states[:, -1, :].cpu().numpy()

                token_embeddings[batch_start:batch_end] = embeddings

        # Average sub-word token embeddings to word level
        word_embeddings = np.zeros((len(words), self.FEATURE_DIM),
                                  dtype=np.float32)
        for w, (start, end) in enumerate(word_to_token_indices):
            word_embeddings[w] = token_embeddings[start:end].mean(axis=0)

        return word_embeddings

    def generate_response(
        self,
        stimulus: list,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate in silico ECoG high-gamma responses for a list of words.

        Parameters
        ----------
        stimulus : list[str]
            List of words for which to generate neural responses. Context is
            built from all preceding words in the list. Word order matters.
        show_progress : bool, default=True
            Whether to display a progress bar during encoding.

        Returns
        -------
        insilico_responses : np.ndarray
            In silico high-gamma response array of shape
            (n_words, n_selected_electrodes, n_selected_timepoints).
            Values are in original high-gamma power units.
        """
        # Validate stimulus
        if not isinstance(stimulus, list) or \
                not all(isinstance(w, str) for w in stimulus):
            raise StimulusError(
                "Stimulus must be a list of strings (words)"
            )

        if len(stimulus) == 0:
            raise StimulusError("Stimulus list must not be empty")

        # Extract GPT-2 XL features
        word_embeddings = self._extract_features(stimulus, show_progress)

        # Standardize features
        X = self.scaler_X.transform(word_embeddings)
        X = X.astype(np.float32)

        # Predict with ridge regression weights
        # Y_pred_scaled = X @ coef.T + intercept
        Y_pred_scaled = X @ self.ridge_coef + self.ridge_intercept

        # Inverse transform to original high-gamma units
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)

        # Reshape to (n_words, n_electrodes, n_lags)
        insilico_responses = Y_pred.reshape(
            len(stimulus),
            len(self.selected_electrodes),
            len(self.selected_timepoints)
        ).astype(np.float32)

        return insilico_responses

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
            Subject ID ("01" through "09").
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

        # If this method is called on an instance
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
            'modality-ecog',
            'train_dataset-podcast_ecog',
            'model-gpt2_xl',
            'metadata',
            f'metadata_sub-{subject}.npy'
        )

        # Load metadata if file exists
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(
                f"Metadata file not found for subject {subject}"
            )

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
        Release GPU memory and unload the language model.

        Frees GPU memory by moving the GPT-2 XL model to CPU and clearing
        CUDA cache if available.
        """
        if hasattr(self, 'language_model') and self.language_model is not None:
            self.language_model.to('cpu')
            self.language_model = None

        if hasattr(self, 'tokenizer'):
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()