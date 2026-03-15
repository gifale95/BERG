import os
import re
import numpy as np
import yaml
from typing import Dict, Any, Optional, List, Union

from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model
from berg.core.exceptions import (
    ModelLoadError,
    InvalidParameterError,
    StimulusError,
)
from berg.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_binary_array,
    get_selected_indices,
)


def load_model_info():
    """Load model information from YAML file."""
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "fmri-lebel-eng1000.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


model_info = load_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.eng1000_ridge",
    class_name="FMRITextEncodingModel",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "eng1000"),
    yaml_path=os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "fmri-lebel-eng1000.yaml"
    )
)


class FMRITextEncodingModel(BaseModelInterface):
    """
    fMRI encoding model predicting voxelwise BOLD responses from natural
    language input using English1000 semantic embeddings and ridge regression.

    Based on the LeBel et al. (2023) natural language fMRI dataset and the
    voxelwise encoding model framework from Huth et al. (2016).

    The model takes sentences as input and internally handles:
    1. Word tokenization and English1000 embedding lookup (985 dimensions)
    2. Lanczos downsampling from word-level to fMRI TR rate (2 seconds)
    3. FIR delay concatenation (4 delays at 2, 4, 6, 8 seconds)
    4. Ridge regression prediction using pre-trained weights
    5. Peak response extraction per sentence
    """

    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(
        model_info["parameters"]["selection"]["properties"].keys()
    )
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"][
        "valid_values"
    ]

    # Per-subject voxel counts (from preprocessed cortical surface data)
    VOXELS_PER_SUBJECT = {
        "UTS01":  81126,
        "UTS02":  94251,
        "UTS03":  95556,
        "UTS04": 109469,
        "UTS05":  99322,
        "UTS06":  92198,
        "UTS07":  94395,
        "UTS08":  97023,
    }

    # Temporal processing parameters (matching the training pipeline)
    TR = 2.0              # fMRI repetition time in seconds
    NDELAYS = 4           # Number of FIR delays (2, 4, 6, 8 seconds)
    WORD_RATE = 3.0       # Assumed words per second (~3.5 in training data)
    SENTENCE_GAP = 3.0    # Seconds of silence between sentences
    LANCZOS_WINDOW = 3    # Number of lobes in the Lanczos filter
    FEATURE_DIM = 985     # English1000 embedding dimensionality

    def __init__(
        self,
        subject: str,
        selection: Optional[Dict] = None,
        device: str = "cpu",
        berg_dir: Optional[str] = None,
    ):
        """
        Initialize the fMRI text encoding model.

        Parameters
        ----------
        subject : str
            Subject ID from the LeBel et al. (2023) dataset
            (e.g., 'UTS01', 'UTS02', ..., 'UTS08').
        selection : dict, optional
            Specifies which voxels to include in the output.
            - roi: List of ROI names (e.g., ['AC', 'Broca'])
            - voxel_index: Binary array indicating which voxels to include
        device : str, default="cpu"
            Device parameter (included for API consistency; this model
            uses NumPy and always runs on CPU).
        berg_dir : str, optional
            Path to the BERG directory containing model weights and metadata.
        """
        self.subject = subject
        self.berg_dir = berg_dir
        self.metadata = None
        self.weights = None
        self.eng1000 = None

        # Parameters from selection
        self.selection = selection
        self.roi_list = None
        self.selected_voxels = None

        # Validate parameters
        self._validate_parameters()

        # This model always runs on CPU
        self.device = "cpu"
        if device not in ("cpu", "auto"):
            print(f"Note: This model runs on CPU only. "
                  f"Ignoring device='{device}'.")

    def _validate_parameters(self):
        """Validate user-provided parameters against the model YAML."""
        # Validate subject
        validate_subject(self.subject, self.VALID_SUBJECTS)

        if self.selection is not None:
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Store ROI list for validation at load time (when metadata
            # is available to check subject-specific ROI availability)
            if "roi" in self.selection:
                roi_input = self.selection["roi"]
                if not isinstance(roi_input, list):
                    roi_input = [roi_input]
                self.roi_list = roi_input

            # Store voxel index for validation at load time (when we
            # know the subject's voxel count)
            if "voxel_index" in self.selection:
                self.selected_voxels = self.selection["voxel_index"]

    def load_model(self) -> None:
        """
        Load pre-trained ridge weights, English1000 embeddings, and metadata.

        Loads the following components:
        - English1000 semantic model (985-dim word embeddings)
        - Ridge regression weights of shape (3940, n_voxels), where
          3940 = 985 features × 4 FIR delays
        - Metadata including ROI masks, prediction correlations, and
          noise ceiling estimates
        """
        if self.berg_dir is None:
            raise InvalidParameterError(
                "berg_dir must be provided to load model weights."
            )

        try:
            # ----------------------------------------------------------------
            # Load metadata (includes ROIs, correlations, noise ceiling)
            # ----------------------------------------------------------------
            metadata_path = os.path.join(
                self.berg_dir,
                'encoding_models',
                'modality-fmri',
                'train_dataset-eng1000',
                'model-ridge',
                'metadata',
                f'sub-{self.subject}.npy'
            )
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"Metadata not found: {metadata_path}"
                )
            self.metadata = np.load(metadata_path, allow_pickle=True).item()
            n_voxels = self.metadata['fmri']['n_voxels']

            # Verify voxel count matches expected value
            expected = self.VOXELS_PER_SUBJECT[self.subject]
            if n_voxels != expected:
                print(f"Warning: Expected {expected} voxels for "
                      f"{self.subject}, found {n_voxels} in metadata.")

            # ----------------------------------------------------------------
            # Validate ROI selection against subject-specific availability
            # ----------------------------------------------------------------
            roi_dict = self.metadata.get('roi', {})
            available_rois = sorted(roi_dict.keys())

            if self.roi_list is not None:
                invalid_rois = [r for r in self.roi_list
                                if r not in roi_dict]
                if invalid_rois:
                    raise InvalidParameterError(
                        f"ROI(s) {invalid_rois} not available for subject "
                        f"{self.subject}. Available ROIs for this subject: "
                        f"{available_rois}"
                    )

            # ----------------------------------------------------------------
            # Build voxel selection from ROI + voxel_index
            # ----------------------------------------------------------------
            selected = set()

            if self.roi_list is not None:
                for roi in self.roi_list:
                    roi_mask = roi_dict[roi]
                    selected.update(np.where(roi_mask)[0].tolist())

            if self.selected_voxels is not None:
                voxel_array = validate_binary_array(
                    self.selected_voxels, n_voxels, "voxel_index"
                )
                selected.update(get_selected_indices(voxel_array).tolist())

            if selected:
                self.selected_voxels = np.array(sorted(selected))
            else:
                self.selected_voxels = np.arange(n_voxels)

            # ----------------------------------------------------------------
            # Load ridge regression weights
            # ----------------------------------------------------------------
            weights_path = os.path.join(
                self.berg_dir,
                'encoding_models',
                'modality-fmri',
                'train_dataset-eng1000',
                'model-ridge',
                'encoding_models_weights',
                f'sub-{self.subject}',
                'weights.npz'
            )
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights not found: {weights_path}"
                )
            weights_data = np.load(weights_path, allow_pickle=True)
            all_weights = weights_data[weights_data.files[0]]
            # Shape: (n_features * n_delays, n_voxels) = (3940, n_voxels)
            # Select only the voxels we need.
            self.weights = all_weights[:, self.selected_voxels]

            # ----------------------------------------------------------------
            # Load English1000 semantic model
            # ----------------------------------------------------------------
            eng1000_path = os.path.join(
                self.berg_dir,
                'encoding_models',
                'modality-fmri',
                'train_dataset-eng1000',
                'model-ridge',
                'encoding_models_weights',
                'english1000sm.hf5'
            )
            if not os.path.exists(eng1000_path):
                raise FileNotFoundError(
                    f"English1000 embeddings not found: {eng1000_path}. "
                    f"Copy english1000sm.hf5 from the deep-fMRI-dataset "
                    f"em_data/ directory."
                )
            self.eng1000 = self._load_english1000(eng1000_path)

            print(f"Model loaded for subject {self.subject} "
                  f"({len(self.selected_voxels)} voxels selected)")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def _load_english1000(self, path: str) -> Dict[str, np.ndarray]:
        """
        Load the English1000 word embedding matrix from HDF5.

        The HDF5 file stores the embedding matrix as (985, 10470) where
        rows are features and columns are words. Each word maps to a
        985-dim vector (a column of the data matrix).

        Parameters
        ----------
        path : str
            Path to the english1000sm.hf5 file.

        Returns
        -------
        dict
            Dictionary mapping lowercase words to 985-dim numpy vectors.
        """
        import h5py
        with h5py.File(path, 'r') as hf:
            vocab = [w.decode('utf-8') if isinstance(w, bytes) else w
                     for w in hf['vocab'][:]]
            # Data shape is (985, 10470): features × words
            data = hf['data'][:]
        # Build dict mapping word -> 985-dim vector (column of data matrix)
        return {word: data[:, i].astype(np.float32)
                for i, word in enumerate(vocab)}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a sentence into words, matching the preprocessing used
        in the training pipeline.

        Converts to lowercase, strips punctuation, and splits on whitespace.

        Parameters
        ----------
        text : str
            Input sentence.

        Returns
        -------
        list of str
            List of lowercase word tokens.
        """
        text = text.lower().strip()
        # Remove punctuation but keep apostrophes within words
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        return words

    def _words_to_features(self, words: List[str]) -> np.ndarray:
        """
        Look up English1000 embedding vectors for a list of words.

        Words not found in the vocabulary are assigned zero vectors.

        Parameters
        ----------
        words : list of str
            Lowercase word tokens.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_words, 985).
        """
        features = np.zeros((len(words), self.FEATURE_DIM), dtype=np.float32)
        for i, word in enumerate(words):
            if word in self.eng1000:
                features[i] = self.eng1000[word]
        return features

    def _lanczos_downsample(
        self,
        features: np.ndarray,
        word_times: np.ndarray,
        tr_times: np.ndarray,
    ) -> np.ndarray:
        """
        Downsample word-level features to TR rate using Lanczos interpolation.

        Each word is treated as a Dirac delta impulse at its onset time,
        scaled by the feature vector. These impulses are low-pass filtered
        (antialiased) and resampled at the fMRI TR times.

        This is equivalent to the Lanczos interpolation used in the training
        pipeline (LeBel et al., 2023; Huth et al., 2016).

        Parameters
        ----------
        features : np.ndarray, shape (n_words, 985)
            Word-level feature vectors.
        word_times : np.ndarray, shape (n_words,)
            Onset time of each word in seconds.
        tr_times : np.ndarray, shape (n_TRs,)
            Time of each fMRI TR in seconds.

        Returns
        -------
        np.ndarray, shape (n_TRs, 985)
            Downsampled feature matrix at TR resolution.
        """
        n_trs = len(tr_times)
        n_features = features.shape[1]
        downsampled = np.zeros((n_trs, n_features), dtype=np.float32)

        # Lanczos kernel with specified number of lobes
        a = self.LANCZOS_WINDOW
        cutoff = 0.5 / self.TR  # Nyquist frequency

        for t_idx, t in enumerate(tr_times):
            for w_idx, wt in enumerate(word_times):
                # Normalized distance
                delta = (t - wt) * 2 * cutoff
                if abs(delta) < a:
                    # Lanczos kernel: sinc(delta) * sinc(delta / a)
                    if abs(delta) < 1e-10:
                        kernel = 1.0
                    else:
                        kernel = (
                            np.sin(np.pi * delta) / (np.pi * delta) *
                            np.sin(np.pi * delta / a) / (np.pi * delta / a)
                        )
                    downsampled[t_idx] += features[w_idx] * kernel

        return downsampled

    def _make_delayed(
        self, features: np.ndarray, delays: List[int]
    ) -> np.ndarray:
        """
        Create a delayed feature matrix for the FIR hemodynamic model.

        For each delay d, the feature matrix is shifted forward by d TRs
        (so that features at time t appear at time t+d). The beginning is
        zero-padded. All delayed copies are concatenated along the feature axis.

        Parameters
        ----------
        features : np.ndarray, shape (n_TRs, n_features)
            Downsampled feature matrix.
        delays : list of int
            Delays in TRs (e.g., [1, 2, 3, 4] for 2, 4, 6, 8 seconds).

        Returns
        -------
        np.ndarray, shape (n_TRs, n_features * n_delays)
            Delayed feature matrix.
        """
        n_trs, n_features = features.shape
        delayed = np.zeros(
            (n_trs, n_features * len(delays)), dtype=np.float32
        )
        for i, d in enumerate(delays):
            if d < n_trs:
                delayed[d:, i * n_features:(i + 1) * n_features] = \
                    features[:n_trs - d]
        return delayed

    def generate_response(
        self,
        stimulus: Union[List[str], np.ndarray],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate in silico fMRI responses for text stimuli.

        Processing pipeline:
        1. Tokenize each sentence into words
        2. Look up 985-dim English1000 embedding per word
        3. Assign word onset times at a fixed rate (3 words/sec)
        4. Lanczos-downsample word features to TR rate (2 seconds)
        5. Apply FIR delays (1, 2, 3, 4 TRs → 2, 4, 6, 8 seconds)
        6. Z-score the delayed features across time
        7. Multiply by pre-trained ridge weights
        8. Extract peak predicted BOLD per sentence

        Parameters
        ----------
        stimulus : list of str or np.ndarray
            Natural language sentences to encode.
        show_progress : bool, default=True
            Whether to show progress (included for API consistency).

        Returns
        -------
        np.ndarray, shape (n_sentences, n_selected_voxels)
            Predicted BOLD responses for each sentence.
        """
        # ----------------------------------------------------------------
        # Validate stimulus
        # ----------------------------------------------------------------
        if isinstance(stimulus, np.ndarray):
            if stimulus.ndim != 1:
                raise StimulusError(
                    f"Stimulus array must be 1D, got shape {stimulus.shape}"
                )
            stimulus = stimulus.tolist()
        if not isinstance(stimulus, list):
            raise StimulusError(
                "Stimulus must be a list of strings or 1D numpy array."
            )
        if len(stimulus) == 0:
            raise StimulusError("Stimulus list cannot be empty.")
        if not all(isinstance(s, str) for s in stimulus):
            raise StimulusError("All stimulus elements must be strings.")

        # ----------------------------------------------------------------
        # Step 1–2: Tokenize sentences and look up embeddings
        # ----------------------------------------------------------------
        all_words = []
        all_word_times = []
        sentence_tr_ranges = []  # (start_tr, end_tr) for each sentence
        current_time = 0.0

        for sent in stimulus:
            words = self._tokenize(sent)
            if len(words) == 0:
                # Empty sentence after tokenization — assign zero response
                all_words.append([])
                sentence_tr_ranges.append(None)
                current_time += self.SENTENCE_GAP
                continue

            # Step 3: Assign word onset times at fixed rate
            word_times = [
                current_time + i / self.WORD_RATE for i in range(len(words))
            ]
            sentence_start = current_time
            sentence_end = word_times[-1] + 1.0 / self.WORD_RATE

            all_words.extend(words)
            all_word_times.extend(word_times)

            # Record which TRs belong to this sentence (accounting for
            # hemodynamic delay: peak response occurs ~4–6s after stimulus).
            # We look for the peak in a window from sentence_start + 4s
            # to sentence_end + 8s (covering the full HRF).
            hrf_start = sentence_start + self.NDELAYS * self.TR / 2
            hrf_end = sentence_end + self.NDELAYS * self.TR
            sentence_tr_ranges.append((hrf_start, hrf_end))

            # Advance time past this sentence plus the inter-sentence gap
            current_time = sentence_end + self.SENTENCE_GAP

        # Total duration determines the number of TRs
        total_duration = current_time + self.NDELAYS * self.TR
        n_trs = int(np.ceil(total_duration / self.TR))
        tr_times = np.arange(n_trs) * self.TR

        # Look up word embeddings
        word_features = self._words_to_features(all_words)
        word_times_arr = np.array(all_word_times, dtype=np.float32)

        # ----------------------------------------------------------------
        # Step 4: Lanczos downsample to TR rate
        # ----------------------------------------------------------------
        downsampled = self._lanczos_downsample(
            word_features, word_times_arr, tr_times
        )

        # ----------------------------------------------------------------
        # Step 5: Apply FIR delays
        # ----------------------------------------------------------------
        delays = list(range(1, self.NDELAYS + 1))
        delayed = self._make_delayed(downsampled, delays)

        # ----------------------------------------------------------------
        # Step 6: Z-score features across time
        # ----------------------------------------------------------------
        mean = delayed.mean(axis=0, keepdims=True)
        std = delayed.std(axis=0, keepdims=True)
        std[std < 1e-10] = 1.0  # Avoid division by zero
        delayed_z = (delayed - mean) / std

        # ----------------------------------------------------------------
        # Step 7: Predict BOLD responses
        # ----------------------------------------------------------------
        predicted_bold = delayed_z @ self.weights

        # ----------------------------------------------------------------
        # Step 8: Extract peak response per sentence
        # ----------------------------------------------------------------
        n_sentences = len(stimulus)
        n_selected = self.weights.shape[1]
        responses = np.zeros((n_sentences, n_selected), dtype=np.float32)

        for i, tr_range in enumerate(sentence_tr_ranges):
            if tr_range is None:
                # Empty sentence — leave as zeros
                continue
            hrf_start, hrf_end = tr_range
            # Find TRs within this sentence's HRF window
            tr_mask = (tr_times >= hrf_start) & (tr_times <= hrf_end)
            if tr_mask.any():
                # Take the peak response (max absolute value, preserving sign)
                sentence_preds = predicted_bold[tr_mask]
                peak_idx = np.argmax(
                    np.abs(sentence_preds).mean(axis=1)
                )
                responses[i] = sentence_preds[peak_idx]

        return responses

    @classmethod
    def get_metadata(
        cls,
        berg_dir: Optional[str] = None,
        subject: Optional[str] = None,
        model_instance: Optional[BaseModelInterface] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.

        Parameters
        ----------
        berg_dir : str, optional
            Path to BERG directory.
        subject : str, optional
            Subject ID (e.g., 'UTS01').
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.

        Returns
        -------
        dict
            Metadata dictionary with 'fmri', 'encoding_models', and 'roi' keys.
        """
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
            subject = model_instance.subject
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir
            subject = cls.subject

        missing = []
        if berg_dir is None:
            missing.append('berg_dir')
        if subject is None:
            missing.append('subject')
        if missing:
            raise InvalidParameterError(
                f"Required parameters missing: {', '.join(missing)}"
            )

        validate_subject(subject, cls.VALID_SUBJECTS)

        metadata_path = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-fmri',
            'train_dataset-eng1000',
            'model-ridge',
            'metadata',
            f'sub-{subject}.npy'
        )

        if os.path.exists(metadata_path):
            return np.load(metadata_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(
                f"Metadata file not found for subject {subject}: "
                f"{metadata_path}"
            )

    @classmethod
    def get_model_id(cls) -> str:
        """Return the model's unique string identifier."""
        return cls.MODEL_ID

    def cleanup(self) -> None:
        """Release memory and resources associated with the model."""
        if hasattr(self, 'weights') and self.weights is not None:
            self.weights = None
        if hasattr(self, 'eng1000') and self.eng1000 is not None:
            self.eng1000 = None
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata = None