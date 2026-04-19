import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
import torch
from tqdm import tqdm

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
        "fmri-lebel2023-opt_1_3b.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


model_info = load_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.lebel2023_opt_1_3b_ridge",
    class_name="FMRIOpt13BEncodingModel",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "lebel2023"),
    yaml_path=os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "fmri-lebel2023-opt_1_3b.yaml"
    )
)


class FMRIOpt13BEncodingModel(BaseModelInterface):
    """
    fMRI encoding model predicting voxelwise BOLD responses from natural
    language input using OPT-1.3B contextual embeddings and ridge regression.

    Based on the scaling-laws approach of Antonello, Vaidya & Huth (NeurIPS
    2023), trained on the LeBel et al. (2023) natural language fMRI dataset.

    The model takes a dictionary of words and their onset times as input and
    internally handles:
    1. OPT-1.3B feature extraction with dynamic context windowing (layer 18)
    2. Lanczos downsampling from word-level to fMRI TR rate (2 seconds)
    3. Z-scoring across time
    4. FIR delay concatenation (4 delays at 2, 4, 6, 8 seconds)
    5. Ridge regression prediction using pre-trained weights
    """

    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    SELECTION_KEYS = list(
        model_info["parameters"]["selection"]["properties"].keys()
    )
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"][
        "valid_values"
    ]

    # Per-subject voxel counts
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

    # Temporal processing
    TR = 2.0
    NDELAYS = 4
    LANCZOS_WINDOW = 3

    # OPT-1.3B
    MODEL_NAME = "facebook/opt-1.3b"
    LAYER = 18
    HIDDEN_DIM = 2048
    CONTEXT_MIN_WORDS = 256
    CONTEXT_MAX_WORDS = 512

    def __init__(
        self,
        subject: str,
        selection: Optional[Dict] = None,
        device: str = "auto",
        berg_dir: Optional[str] = None,
    ):
        """
        Initialize the fMRI OPT-1.3B encoding model.

        Parameters
        ----------
        subject : str
            Subject ID from the LeBel et al. (2023) dataset
            (e.g., 'UTS01' through 'UTS08').
        selection : dict, optional
            Specifies which voxels to include in the output.
            - roi: List of ROI names (e.g., ['AC', 'Broca'])
            - voxel_index: Binary array indicating which voxels to include
        device : str, default="auto"
            Device for OPT-1.3B inference. Options: "cpu", "cuda", "auto".
            OPT-1.3B requires ~3 GB VRAM (fp16) or ~5 GB RAM (fp32).
        berg_dir : str, optional
            Path to BERG directory with model weights and metadata.
        """
        self.subject = subject
        self.berg_dir = berg_dir
        self.metadata = None
        self.weights = None
        self.opt_model = None
        self.tokenizer = None

        # Parameters from selection
        self.selection = selection
        self.roi_list = None
        self.selected_voxels = None

        # Validate parameters
        self._validate_parameters()

        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def _validate_parameters(self):
        """Validate user-provided parameters against the model YAML."""
        validate_subject(self.subject, self.VALID_SUBJECTS)

        if self.selection is not None:
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            if "roi" in self.selection:
                roi_input = self.selection["roi"]
                if not isinstance(roi_input, list):
                    roi_input = [roi_input]
                self.roi_list = roi_input

            if "voxel_index" in self.selection:
                self.selected_voxels = self.selection["voxel_index"]

    def load_model(self) -> None:
        """
        Load OPT-1.3B, pre-trained ridge weights, and metadata.

        Loads the following components:
        - OPT-1.3B language model and tokenizer from HuggingFace
        - Ridge regression weights of shape (8192, n_voxels), where
          8192 = 2048 hidden_dim × 4 FIR delays
        - Metadata including ROI masks, prediction correlations, and
          noise ceiling estimates
        """
        if self.berg_dir is None:
            raise InvalidParameterError(
                "berg_dir must be provided to load model weights."
            )

        try:
            # ----------------------------------------------------------------
            # Load metadata
            # ----------------------------------------------------------------
            metadata_path = os.path.join(
                self.berg_dir,
                'encoding_models',
                'modality-fmri',
                'train_dataset-lebel2023',
                'model-opt_1_3b_ridge',
                'metadata',
                f'metadata_{self.subject}.npy'
            )
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"Metadata not found: {metadata_path}"
                )
            self.metadata = np.load(metadata_path, allow_pickle=True).item()
            n_voxels = self.metadata['fmri']['n_voxels']

            expected = self.VOXELS_PER_SUBJECT[self.subject]
            if n_voxels != expected:
                print(f"Warning: Expected {expected} voxels for "
                      f"{self.subject}, found {n_voxels} in metadata.")

            # ----------------------------------------------------------------
            # Validate ROI selection
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
            # Build voxel selection
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
                'train_dataset-lebel2023',
                'model-opt_1_3b_ridge',
                'encoding_models_weights',
                f'weights_{self.subject}.npy'
            )
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights not found: {weights_path}"
                )
            weights_data = np.load(weights_path, allow_pickle=True).item()
            all_weights = weights_data['ridge_weights']
            self.weights = all_weights[:, self.selected_voxels].astype(
                np.float32)

            # ----------------------------------------------------------------
            # Load OPT-1.3B model and tokenizer
            # ----------------------------------------------------------------
            print(f"Loading {self.MODEL_NAME} ...")
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.opt_model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=(torch.float16 if self.device == "cuda"
                             else torch.float32),
            )
            self.opt_model.eval()
            self.opt_model.to(self.device)

            print(f"Model loaded for subject {self.subject} "
                  f"({len(self.selected_voxels)} voxels selected, "
                  f"device={self.device})")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    # ====================================================================
    # OPT-1.3B feature extraction
    # ====================================================================

    def _tokenize_story(self, words: List[str]):
        """Tokenize words and build word-to-token mapping.

        Each real word (non-empty, not bare possessive) is tokenized
        individually with a leading space (OPT BPE convention).

        Parameters
        ----------
        words : list of str
            Raw word list.

        Returns
        -------
        all_tokens : list of int
            Token IDs including BOS.
        real_word_indices : list of int
            Indices into *words* for real words.
        word_first_tok : list of int
            First token index per real word.
        word_last_tok : list of int
            Last token index per real word.
        """
        bos_id = self.tokenizer.bos_token_id
        all_tokens = [bos_id]
        real_word_indices = []
        word_first_tok = []
        word_last_tok = []

        is_first = True
        for i, w in enumerate(words):
            if w.strip() == '' or w == "'s":
                continue
            real_word_indices.append(i)
            prefix = '' if is_first else ' '
            toks = self.tokenizer.encode(prefix + w,
                                         add_special_tokens=False)
            word_first_tok.append(len(all_tokens))
            all_tokens.extend(toks)
            word_last_tok.append(len(all_tokens) - 1)
            is_first = False

        return all_tokens, real_word_indices, word_first_tok, word_last_tok

    def _extract_features(self, words: List[str]) -> np.ndarray:
        """Extract OPT-1.3B hidden states using dynamic context windows.

        Following Antonello et al. (2023): context grows word-by-word until
        CONTEXT_MAX_WORDS, then a forward pass is run and the context resets
        to CONTEXT_MIN_WORDS. For each word, the hidden state at its last
        BPE token from the specified layer is used.

        Non-real words receive a copy of the most recent real word's vector.

        Parameters
        ----------
        words : list of str
            Input words.

        Returns
        -------
        features : ndarray, shape (len(words), HIDDEN_DIM), float32
        """
        all_tokens, real_word_indices, word_first_tok, word_last_tok = \
            self._tokenize_story(words)
        n_real = len(real_word_indices)

        if n_real == 0:
            return np.zeros((len(words), self.HIDDEN_DIM), dtype=np.float32)

        real_features = np.zeros((n_real, self.HIDDEN_DIM), dtype=np.float32)
        phase_start = 0
        next_to_assign = 0

        for rw in range(n_real):
            words_in_ctx = rw - phase_start + 1
            at_end = (rw == n_real - 1)

            if words_in_ctx < self.CONTEXT_MAX_WORDS and not at_end:
                continue

            # Build token context
            if phase_start == 0:
                ctx_tokens = all_tokens[:word_last_tok[rw] + 1]
            else:
                tok_start = word_first_tok[phase_start]
                tok_end = word_last_tok[rw] + 1
                ctx_tokens = [all_tokens[0]] + all_tokens[tok_start:tok_end]

            input_ids = torch.tensor(
                [ctx_tokens], dtype=torch.long, device=self.device)

            with torch.no_grad():
                hidden = (
                    self.opt_model(input_ids, output_hidden_states=True)
                    .hidden_states[self.LAYER][0]
                    .cpu().float().numpy()
                )

            # Read off hidden state at each word's last token
            for w in range(next_to_assign, rw + 1):
                if phase_start == 0:
                    rel = word_last_tok[w]
                else:
                    rel = (word_last_tok[w]
                           - word_first_tok[phase_start] + 1)
                real_features[w] = hidden[rel]

            next_to_assign = rw + 1
            if not at_end:
                phase_start = max(0, rw - self.CONTEXT_MIN_WORDS + 1)

        # Map back to full word list
        features = np.zeros((len(words), self.HIDDEN_DIM), dtype=np.float32)
        last_feat = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        rp = 0
        for i in range(len(words)):
            if rp < n_real and real_word_indices[rp] == i:
                features[i] = real_features[rp]
                last_feat = real_features[rp]
                rp += 1
            else:
                features[i] = last_feat

        return features

    # ====================================================================
    # Temporal processing
    # ====================================================================

    def _lanczos_downsample(
        self,
        features: np.ndarray,
        word_times: np.ndarray,
        tr_times: np.ndarray,
    ) -> np.ndarray:
        """Downsample word-level features to TR rate using Lanczos
        interpolation.

        Each word is treated as a Dirac delta impulse at its onset time,
        scaled by the feature vector. These impulses are low-pass filtered
        and resampled at the fMRI TR times.

        Parameters
        ----------
        features : ndarray, shape (n_words, hidden_dim)
        word_times : ndarray, shape (n_words,)
            Word onset times in seconds.
        tr_times : ndarray, shape (n_TRs,)
            fMRI TR times in seconds.

        Returns
        -------
        ndarray, shape (n_TRs, hidden_dim)
        """
        n_trs = len(tr_times)
        n_features = features.shape[1]
        downsampled = np.zeros((n_trs, n_features), dtype=np.float32)

        a = self.LANCZOS_WINDOW
        cutoff = 0.5 / self.TR  # Nyquist frequency

        for t_idx, t in enumerate(tr_times):
            for w_idx, wt in enumerate(word_times):
                delta = (t - wt) * 2 * cutoff
                if abs(delta) < a:
                    if abs(delta) < 1e-10:
                        kernel = 1.0
                    else:
                        kernel = (
                            np.sin(np.pi * delta) / (np.pi * delta) *
                            np.sin(np.pi * delta / a)
                            / (np.pi * delta / a)
                        )
                    downsampled[t_idx] += features[w_idx] * kernel

        return downsampled

    def _make_delayed(
        self, features: np.ndarray, delays: List[int]
    ) -> np.ndarray:
        """Create delayed feature matrix for the FIR hemodynamic model.

        For each delay d, the feature matrix is shifted forward by d TRs.
        The beginning is zero-padded. All delayed copies are concatenated
        along the feature axis.

        Parameters
        ----------
        features : ndarray, shape (n_TRs, n_features)
        delays : list of int
            Delays in TRs (e.g., [1, 2, 3, 4]).

        Returns
        -------
        ndarray, shape (n_TRs, n_features * len(delays))
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

    # ====================================================================
    # Main inference
    # ====================================================================

    def generate_response(
        self,
        stimulus: Dict[str, Any],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate in silico fMRI responses for a word sequence with
        onset times.

        Processing pipeline:
        1. Extract OPT-1.3B contextual embeddings per word (layer 18)
        2. Lanczos-downsample to TR rate (2s)
        3. Z-score features across time
        4. Apply FIR delays (2, 4, 6, 8 seconds)
        5. Multiply by pre-trained ridge weights

        Parameters
        ----------
        stimulus : dict
            Dictionary with two required keys:
            - "words": list of str — words in presentation order
            - "word_onsets": list of float — onset time of each word (seconds)
            Both must have the same length.
        show_progress : bool, default=True
            Whether to show a progress indicator.

        Returns
        -------
        ndarray, shape (n_TRs, n_selected_voxels)
            Predicted z-scored BOLD responses at each TR.
        """
        # ----------------------------------------------------------------
        # Validate stimulus
        # ----------------------------------------------------------------
        if not isinstance(stimulus, dict):
            raise StimulusError(
                "Stimulus must be a dictionary with 'words' and "
                "'word_onsets' keys."
            )
        if "words" not in stimulus or "word_onsets" not in stimulus:
            raise StimulusError(
                "Stimulus must contain 'words' and 'word_onsets' keys."
            )

        words = stimulus["words"]
        word_onsets = np.asarray(stimulus["word_onsets"], dtype=np.float64)

        if len(words) == 0:
            raise StimulusError("Word list cannot be empty.")
        if len(words) != len(word_onsets):
            raise StimulusError(
                f"'words' (length {len(words)}) and 'word_onsets' "
                f"(length {len(word_onsets)}) must have the same length."
            )

        # ----------------------------------------------------------------
        # Extract OPT-1.3B features
        # ----------------------------------------------------------------
        if show_progress:
            print("Extracting OPT-1.3B features ...")
        word_features = self._extract_features(words)

        # ----------------------------------------------------------------
        # Lanczos downsample to TR rate
        # ----------------------------------------------------------------
        # Generate TR times from the word onsets
        t_start = word_onsets[0]
        t_end = word_onsets[-1] + self.NDELAYS * self.TR
        n_trs = int(np.ceil((t_end - t_start) / self.TR))
        tr_times = t_start + np.arange(n_trs) * self.TR

        downsampled = self._lanczos_downsample(
            word_features, word_onsets, tr_times
        )

        # ----------------------------------------------------------------
        # Z-score features
        # ----------------------------------------------------------------
        mean = downsampled.mean(axis=0, keepdims=True)
        std = downsampled.std(axis=0, keepdims=True)
        std[std < 1e-10] = 1.0
        downsampled_z = (downsampled - mean) / std

        # ----------------------------------------------------------------
        # FIR delays
        # ----------------------------------------------------------------
        delays = list(range(1, self.NDELAYS + 1))
        delayed = self._make_delayed(downsampled_z, delays)

        # ----------------------------------------------------------------
        # Predict BOLD
        # ----------------------------------------------------------------
        predicted_bold = delayed @ self.weights

        return predicted_bold.astype(np.float32)

    # ====================================================================
    # Class methods
    # ====================================================================

    @classmethod
    def get_metadata(
        cls,
        berg_dir: Optional[str] = None,
        subject: Optional[str] = None,
        model_instance: Optional[BaseModelInterface] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Retrieve subject metadata (fmri info, ROI masks, encoding stats)."""
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
            'train_dataset-lebel2023',
            'model-opt_1_3b_ridge',
            'metadata',
            f'metadata_{subject}.npy'
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
        """Release GPU memory and unload all model components."""
        if hasattr(self, 'opt_model') and self.opt_model is not None:
            if hasattr(self.opt_model, 'to'):
                self.opt_model.to('cpu')
            self.opt_model = None

        if hasattr(self, 'tokenizer'):
            self.tokenizer = None

        if hasattr(self, 'weights'):
            self.weights = None

        if hasattr(self, 'metadata'):
            self.metadata = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()