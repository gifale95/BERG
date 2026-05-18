"""TRIBE v2 fMRI encoding model for BERG.

Wraps the TRIBE v2 multimodal brain encoding model (d'Ascoli et al., 2026)
to generate in silico fMRI responses on the fsaverage5 cortical surface
(20,484 vertices) from video, audio, or text stimuli.

TRIBE v2 uses frozen pretrained feature extractors (V-JEPA2-Giant for video,
Wav2Vec-BERT-2.0 for audio, LLaMA-3.2-3B for text) fed into a trainable
Transformer that predicts whole-brain fMRI at 1 Hz temporal resolution.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
import importlib.util

from berg.core.exceptions import (
    InvalidParameterError,
    ModelLoadError,
    StimulusError,
)
from berg.core.model_registry import register_model
from berg.core.parameter_validator import (
    validate_binary_array,
    validate_selection_keys,
    get_selected_indices,
)
from berg.interfaces.base_model import BaseModelInterface


# =============================================================================
# Load model info from YAML
# =============================================================================


def load_model_info():
    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "model_cards", "fmri-dascoli_2026-tribe_v2.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


model_info = load_model_info()


if importlib.util.find_spec("tribev2") is not None:
    register_model(
        model_id=model_info["model_id"],
        module_path="berg.models.fmri.dascoli_2026_tribe_v2",
        class_name="TribeV2EncodingModel",
        modality=model_info.get("modality", "fmri"),
        training_dataset=model_info.get("training_dataset", "dascoli_2026"),
        yaml_path=os.path.join(
            os.path.dirname(__file__),
            "..",
            "model_cards",
            "fmri-dascoli_2026-tribe_v2.yaml",
        ),
    )

# =============================================================================
# Supported stimulus file extensions
# =============================================================================

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
VALID_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
VALID_TEXT_EXTENSIONS = {".txt"}
ALL_VALID_EXTENSIONS = VALID_VIDEO_EXTENSIONS | VALID_AUDIO_EXTENSIONS | VALID_TEXT_EXTENSIONS


# =============================================================================
# Model implementation
# =============================================================================


class TribeV2EncodingModel(BaseModelInterface):
    """TRIBE v2 encoding model for generating in silico fMRI responses.

    Wraps the TRIBE v2 model to predict cortical fMRI activity on the
    fsaverage5 surface (20,484 vertices) from video, audio, or text stimuli.
    The model operates in 'unseen subject' mode, producing group-average-like
    predictions without requiring subject-specific training data.

    Parameters
    ----------
    device : str
        Device for computation ('cpu', 'cuda', or 'auto').
    selection : dict, optional
        Vertex selection via ROI names and/or binary vertex mask.
    berg_dir : str, optional
        Root path to the BERG directory containing metadata files.
    """

    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    N_VERTICES = 20484
    N_VERTICES_PER_HEMI = 10242

    def __init__(
        self,
        device: str = "auto",
        selection: Optional[Dict] = None,
        berg_dir: Optional[str] = None,
    ):
        """Initialize the TRIBE v2 encoding model.

        Parameters
        ----------
        device : str, default='auto'
            Target device for computation. 'auto' selects CUDA if available.
        selection : dict, optional
            Specifies which cortical vertices to include in the output.
            Keys can include:
            - 'roi': list of Glasser HCP-MMP1.0 ROI names (e.g., ['V1', 'V2'])
            - 'vertices': binary array of length 20,484 for vertex selection
            If both are provided, their union is used.
        berg_dir : str, optional
            Root path to the BERG directory.
        """
        self.subject = "average"
        self.berg_dir = berg_dir
        self.selection = selection
        self.tribe_model = None

        # Selection state
        self.selected_rois = None
        self.selected_vertices = None

        # Validate parameters
        self._validate_parameters()

        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def _validate_parameters(self):
        """Validate user-provided parameters against the model YAML.

        Checks that selection keys, ROI names, and vertex arrays all
        conform to the expected formats and values.
        """
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Validate ROIs (names are checked against metadata during load_model)
            if "roi" in self.selection:
                roi_list = self.selection["roi"]
                if not isinstance(roi_list, list):
                    raise InvalidParameterError("ROI must be provided as a list")
                self.selected_rois = roi_list

            # Validate vertex binary array
            if "vertices" in self.selection:
                vertices_array = validate_binary_array(
                    self.selection["vertices"],
                    self.N_VERTICES,
                    "vertices",
                )
                self.selected_vertices = get_selected_indices(vertices_array)

    def load_model(self) -> None:
        """Load TRIBE v2 model weights and metadata.

        Downloads the model checkpoint from HuggingFace (cached after first
        download), loads metadata for ROI selection, and initializes the
        feature extraction + Transformer inference pipeline.

        The first call triggers downloads of:
        - TRIBE v2 checkpoint (~1 GB)
        - V-JEPA2-Giant, Wav2Vec-BERT-2.0, LLaMA-3.2-3B backbone weights

        Raises
        ------
        ModelLoadError
            If the model fails to load (e.g., missing HuggingFace auth,
            insufficient GPU memory).
        """
        try:
            # Load metadata for ROI selection
            metadata_path = os.path.join(
                self.berg_dir,
                "encoding_models",
                "modality-fmri",
                "train_dataset-dascoli_2026",
                "model-tribe_v2",
                "metadata",
                "metadata.npy",
            )
            self.metadata = np.load(metadata_path, allow_pickle=True).item()

            # Resolve ROI selection to vertex indices
            if self.selected_rois is not None:
                # Validate ROI names against YAML
                invalid_rois = [r for r in self.selected_rois if r not in self.VALID_ROIS]
                if invalid_rois:
                    raise InvalidParameterError(
                        f"Invalid ROI(s): {invalid_rois}. "
                        f"Valid ROIs: {self.VALID_ROIS}"
                    )

                # Get vertex indices for selected ROIs using roi_index mapping
                roi_assignments = self.metadata["roi"]["roi_assignments"]
                roi_index = self.metadata["roi"]["roi_index"]
                roi_vertex_indices = np.array([], dtype=int)
                for roi_name in self.selected_rois:
                    roi_vertices = np.where(roi_assignments == roi_index[roi_name])[0]
                    roi_vertex_indices = np.append(roi_vertex_indices, roi_vertices)

                # Combine with explicit vertex selection (logical OR)
                if self.selected_vertices is not None:
                    combined = set(self.selected_vertices) | set(roi_vertex_indices)
                    self.selected_vertices = sorted(list(combined))
                else:
                    self.selected_vertices = sorted(list(set(roi_vertex_indices)))

            # If no selection at all, use all vertices
            if self.selected_vertices is None:
                self.selected_vertices = list(range(self.N_VERTICES))

            # Cache directory for TRIBE v2 features
            cache_folder = os.path.join(self.berg_dir, "cache", "tribe_v2")
            os.makedirs(cache_folder, exist_ok=True)

            # Load TRIBE v2 model from HuggingFace
            print("Loading TRIBE v2 model from HuggingFace...")
            print(
                "  (First run downloads ~1 GB checkpoint + backbone model weights. "
                "This may take several minutes.)"
            )
            from tribev2.demo_utils import TribeModel

            # Ensure 'spawn' multiprocessing start method to avoid broken pipe
            import torch.multiprocessing
            try:
                torch.multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError:
                pass  # already set

            # Whisperx sometimes has compute errors if CPU selected. This patches it.
            if self.device != "cuda":
                try:
                    import subprocess
                    import functools
                    import tribev2.eventstransforms as _et

                    _orig_subprocess_run = subprocess.run

                    @functools.wraps(_orig_subprocess_run)
                    def _patched_subprocess_run(cmd, *args, **kwargs):
                        if isinstance(cmd, list):
                            cmd = ["int8" if c == "float16" else c for c in cmd]
                        return _orig_subprocess_run(cmd, *args, **kwargs)

                    subprocess.run = _patched_subprocess_run
                    print("  Patched whisperx compute_type to int8 for CPU compatibility.")
                except Exception as patch_err:
                    print(f"  Warning: could not patch whisperx compute_type: {patch_err}")

            # Build config overrides to ensure feature extractors use the
            # correct device (they default to CUDA in the shipped config).
            config_update = {}
            if self.device != "cuda":
                config_update["data.text_feature.device"] = self.device
                config_update["data.audio_feature.device"] = self.device
                config_update["data.image_feature.image.device"] = self.device
                config_update["data.video_feature.image.device"] = self.device
                config_update["data.num_workers"] = 0

            self.tribe_model = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder=cache_folder,
                device=self.device,
                config_update=config_update,
            )

            print(
                f"Model loaded on {self.device} "
                f"({len(self.selected_vertices)} / {self.N_VERTICES} vertices selected)"
            )

        except Exception as e:
            raise ModelLoadError(f"Failed to load TRIBE v2 model: {e}")

    def generate_response(
        self,
        stimulus: str,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate in silico fMRI responses for a video, audio, or text stimulus.

        Takes a file path to a stimulus, runs the full TRIBE v2 pipeline
        (feature extraction → Transformer → cortical prediction), and returns
        the predicted fMRI activity for the selected cortical vertices.

        Parameters
        ----------
        stimulus : str
            File path to the stimulus. Supported formats:
            - Video: .mp4, .avi, .mkv, .mov, .webm
            - Audio: .wav, .mp3, .flac, .ogg
            - Text: .txt (converted to speech via gTTS, then processed)
        show_progress : bool, default=True
            Whether to display a progress bar during inference.

        Returns
        -------
        np.ndarray
            Predicted fMRI activity, shape (n_timesteps, n_selected_vertices).
            n_timesteps depends on stimulus duration (1 per second).
            n_selected_vertices depends on the selection parameter.

        Raises
        ------
        StimulusError
            If the file does not exist or has an unsupported extension.
        RuntimeError
            If the model has not been loaded via load_model().
        """
        if self.tribe_model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before generate_response()."
            )

        # Validate stimulus path
        if not isinstance(stimulus, (str, Path)):
            raise StimulusError(
                "Stimulus must be a file path (str or Path) to a video, audio, "
                "or text file."
            )

        path = Path(stimulus)
        if not path.is_file():
            raise StimulusError(f"Stimulus file does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix not in ALL_VALID_EXTENSIONS:
            raise StimulusError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported formats: video {sorted(VALID_VIDEO_EXTENSIONS)}, "
                f"audio {sorted(VALID_AUDIO_EXTENSIONS)}, "
                f"text {sorted(VALID_TEXT_EXTENSIONS)}"
            )

        # Build events dataframe based on file type
        if suffix in VALID_VIDEO_EXTENSIONS:
            df = self.tribe_model.get_events_dataframe(video_path=str(path))
        elif suffix in VALID_AUDIO_EXTENSIONS:
            df = self.tribe_model.get_events_dataframe(audio_path=str(path))
        elif suffix in VALID_TEXT_EXTENSIONS:
            df = self.tribe_model.get_events_dataframe(text_path=str(path))

        # Run prediction
        preds, segments = self.tribe_model.predict(
            events=df, verbose=show_progress
        )
        # preds shape: (n_timesteps, 20484)

        # Apply vertex selection
        preds = preds[:, self.selected_vertices]

        return preds

    @classmethod
    def get_metadata(
        cls,
        berg_dir=None,
        model_instance=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Retrieve metadata for the TRIBE v2 model.

        Parameters
        ----------
        berg_dir : str
            Path to BERG directory.
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        dict
            Metadata dictionary with keys 'fmri', 'roi'.

        Raises
        ------
        InvalidParameterError
            If required parameters are missing.
        FileNotFoundError
            If the metadata file does not exist.
        """
        # Extract parameters from model instance if provided
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir

        # Validate required parameters
        if berg_dir is None:
            raise InvalidParameterError("Required parameter missing: berg_dir")

        # Build metadata path
        metadata_path = os.path.join(
            berg_dir,
            "encoding_models",
            "modality-fmri",
            "train_dataset-dascoli_2026",
            "model-tribe_v2",
            "metadata",
            "metadata.npy",
        )

        if os.path.exists(metadata_path):
            return np.load(metadata_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                "Run prepare_tribe_v2.py to generate metadata."
            )

    @classmethod
    def get_model_id(cls) -> str:
        """Return the model's unique string identifier.

        Returns
        -------
        str
            Model ID string: 'fmri-dascoli_2026-tribe_v2'
        """
        return cls.MODEL_ID

    def cleanup(self) -> None:
        """Release GPU memory and unload the model.

        Frees GPU memory by deleting the TRIBE v2 model and all loaded
        backbone models (V-JEPA2, LLaMA, Wav2Vec-BERT), then clears
        the CUDA cache.
        """
        if self.tribe_model is not None:
            # The TribeModel holds _model (FmriEncoderModel) on GPU
            if hasattr(self.tribe_model, "_model") and self.tribe_model._model is not None:
                self.tribe_model._model.to("cpu")
                self.tribe_model._model = None

            self.tribe_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()