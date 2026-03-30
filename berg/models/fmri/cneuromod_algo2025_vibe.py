import os
import shutil
import tempfile
import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import yaml
from nilearn.datasets import fetch_atlas_schaefer_2018

from berg.core.exceptions import InvalidParameterError
from berg.core.model_registry import register_model
from berg.core.parameter_validator import (
    validate_binary_array,
    validate_roi,
    validate_selection_keys,
    validate_subjects,
)
from berg.interfaces.base_model import BaseModelInterface
from berg.models.fmri.vibe_utils.audio_feature_extractor import AudioFeatureExtractor
from berg.models.fmri.vibe_utils.config import VIBEConfig, get_pretrained_model_configs
from berg.models.fmri.vibe_utils.model import VIBEModel
from berg.models.fmri.vibe_utils.text_feature_extractor import TextFeatureExtractor
from berg.models.fmri.vibe_utils.video_feature_extractor import VideoFeatureExtractor

# Load model info from YAML


def load_model_info():
    yaml_path = os.path.join(os.path.dirname(
        __file__), "..", "model_cards", "fmri-cneuromod_algo2025-vibe.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.cneuromod_algo2025_vibe",
    class_name="VIBE",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "cneuromod_algo2025"),
    yaml_path=os.path.join(os.path.dirname(__file__),
                           "..", "model_cards", "fmri-cneuromod_algo2025-vibe.yaml")
)

# Algonauts Mapping
SUBJECT_MAPPING = {1: 0,
                   2: 1,
                   3: 2,
                   5: 3,}


class VIBE(BaseModelInterface):
    """
    BERG interface for the VIBE multimodal fMRI encoding model.

    VIBE combines text, audio, and video features aligned to TRs and predicts
    parcel-wise fMRI responses.

    Attributes:
        text_feature_extractor (TextFeatureExtractor): Transcript encoder.
        audio_feature_extractor (AudioFeatureExtractor): Audio encoder.
        video_feature_extractor (VideoFeatureExtractor): Video encoder.
        model (VIBEModel): Mapping model from fused multimodal features to brain space.
        config (VIBEConfig): Hyperparameters for the model architecture.
        selection (dict): User selection criteria (e.g., specific ROIs).
    """

    MODEL_ID = model_info["model_id"]
    # Extract any validation info from model_info
    _collection_slug = "ShreyDixit/vibe"
    SELECTION_KEYS = list(
        model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    N_PARCELS_MODEL = 1000  # Model output size (1000 ROIs from Schaefer 2018 atlas)

    # Setting it to None so that no calls to hugging face are made during the import
    pretrained_configs_dict = None

    def __init__(self,
                 berg_dir: str = None,
                 selection: dict = None,
                 subject: int = None,
                 config: VIBEConfig = VIBEConfig(),
                 device: str = "auto",
                 low_mem_use: bool = True,
                 model_variant: str = None):
        """
        Initialize the VIBE interface.

        Args:
            berg_dir (str, optional): Path to the BERG cache directory.
            selection (dict, optional): Dictionary specifying output filters.
                To filter by brain region, include the key "roi" with a value corresponding 
                to a network name from the Schaefer 2018 atlas (7-network parcellation).
                Valid values include: 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 
                'Limbic', 'Cont', or 'Default'.
            subject (int): The subject ID (index) to generate predictions for.
            config (VIBEConfig): Configuration object for the model architecture.
                Ignored if `model_variant` is provided.
            device (str): Computation device ('cpu', 'cuda', 'auto').
            low_mem_use : bool
                If True, feature extractors/model are unloaded after use to reduce
                memory usage at the cost of extra load time.
            model_variant (str, optional): HuggingFace repo ID of a specific pretrained
                variant to load.
                If provided, the config tied to this variant is used and the `config` argument is ignored.
                If None, uses the default configuration.
                Use model.get_pretrained_variants() to see all available options.
        """
        self.model = None
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.berg_dir = berg_dir
        self.selection = selection
        self.roi = self.selection.get(
            "roi", None) if self.selection is not None else None
        self.subject = subject
        self.model_variant = model_variant
        self.subject_idx = None
        self.parcel_index = None

        if VIBE.pretrained_configs_dict is None:
            VIBE.get_pretrained_configs()

        if self.model_variant is not None:
            variant_to_config = {
                repo_id: cfg for cfg, repo_id in self.pretrained_configs_dict.items()
            }
            if self.model_variant not in variant_to_config:
                valid_variants = list(self.pretrained_configs_dict.values())
                raise InvalidParameterError(
                    f"Invalid model_variant: '{self.model_variant}'. "
                    f"Available variants: {valid_variants}. "
                    "Use model.get_pretrained_variants() to see all options."
                )
            config = variant_to_config[self.model_variant]

        self.config: VIBEConfig = config
        self.text_feature_extractor = TextFeatureExtractor(
            config=self.config, device=device, low_mem_usage=low_mem_use)
        self.audio_feature_extractor = AudioFeatureExtractor(
            config=self.config, device=device, low_mem_usage=low_mem_use)
        self.video_feature_extractor = VideoFeatureExtractor(
            config=self.config, device=device, low_mem_usage=low_mem_use)

        self._validate_parameters()
        self.roi_labels = self._extract_network_names()
        self.low_mem_use = low_mem_use

    def _validate_parameters(self):
        """
        Validate the input parameters against the model specs.
        """
        if self.config not in self.pretrained_configs_dict:
            logging.warning(
                f"Config {self.config} not found in pretrained registry. "
                "Model will be initialized with RANDOM weights. "
                "To use a pretrained model, ensure parameters match a valid config."
                f"Pretrained configs: {self.pretrained_configs_dict.keys()}"
            )

        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Individual validations
            if "roi" in self.selection:
                self.roi = validate_roi(
                    self.selection["roi"], self.VALID_ROIS
                )

            if "parcel_index" in self.selection:
                parcel_index = self.selection["parcel_index"]
                validated_array = validate_binary_array(
                    parcel_index,
                    expected_length=self.N_PARCELS_MODEL,
                    parameter_name="parcel_index"
                )
                self.parcel_index = validated_array.astype(bool)

        if self.subject is not None:
            self.subject_idx = self._normalize_subject(self.subject)

        # Validate model_variant if provided
        if self.model_variant is not None:
            valid_variants = list(self.pretrained_configs_dict.values())
            if self.model_variant not in valid_variants:
                raise InvalidParameterError(
                    f"Invalid model_variant: '{self.model_variant}'. "
                    f"Available variants: {valid_variants}. "
                    "Use model.get_pretrained_variants() to see all options."
                )

    @classmethod
    def get_pretrained_configs(cls) -> list[VIBEConfig]:
        """
        A list of all pretrained configs available on HuggingFace. Choose from this list to get a pretrained model.

        Returns
        -------
        List[VIBEConfig]:
            A list of all pretrained configs available on HuggingFace.

        Notes
        -----
        This property is a lazy loader, meaning it only loads the registry when first accessed.
        """
        if cls.pretrained_configs_dict is None:
            cls.pretrained_configs_dict = get_pretrained_model_configs(
                cls._collection_slug)
        return list(cls.pretrained_configs_dict.keys())

    @classmethod
    def get_pretrained_variants(cls) -> list[str]:
        """
        Returns the HuggingFace repo IDs of all available pretrained model variants.

        Returns
        -------
        list[str]
            A list of HuggingFace repo IDs.
        """
        if cls.pretrained_configs_dict is None:
            cls.get_pretrained_configs()
        return list(cls.pretrained_configs_dict.values())

    def load_model(self, load_feature_extractors: bool = False):
        """
        Load VIBE weights and optionally preload modality feature extractors.

        Args:
            load_feature_extractors (bool): If True, preloads all modality feature extractors.
        """
        if load_feature_extractors:
            self.text_feature_extractor.load_model()
            self.audio_feature_extractor.load_model()
            self.video_feature_extractor.load_model()

        self.model = VIBEModel(config=self.config, device=self.device, berg_dir=self.berg_dir)

        if self.config in self.pretrained_configs_dict:
            repo_id = self.model_variant if self.model_variant is not None else self.pretrained_configs_dict[self.config]
            self.model.load_model_from_hub(repo_id)
        self.model.eval()

    def _normalize_subject(self, subject: int) -> int:
        validated_subject = validate_subjects(
            subject, self.VALID_SUBJECTS
        )[0]
        return SUBJECT_MAPPING[validated_subject]

    def _get_selected_parcel_indices(self):
        if self.roi is None and self.parcel_index is None:
            return None

        mask = np.zeros(self.N_PARCELS_MODEL, dtype=bool)
        if self.roi is not None:
            mask |= np.isin(self.roi_labels, self.roi)
        if self.parcel_index is not None:
            mask |= self.parcel_index
        return np.where(mask)[0]

    @contextmanager
    def _model_session(self):
        if self.model is None:
            self.load_model()
        try:
            yield
        finally:
            if self.low_mem_use:
                self._cleanup_model()

    @torch.no_grad()
    def generate_response(self,
                          transcripts: list[str],
                          video_path: str) -> torch.Tensor:
        """
        Generate predicted fMRI responses for transcript + video input.

        Args:
            transcripts (list[str]): Transcript string per TR.
            video_path (str): Path to the video file used for audio/video features.

        Returns:
            torch.Tensor: Predicted brain activity. 
                Shape: [num_timepoints, num_parcels] (or subset of parcels if selected).
        """
        if self.subject_idx is None:
            raise InvalidParameterError(
                "Subject must be set during model initialization."
            )

        if len(transcripts) == 0:
            raise InvalidParameterError("transcripts cannot be empty.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        text_features = self.text_feature_extractor.extract_features(transcripts)
        audio_features = self.audio_feature_extractor.extract_features(video_path)
        video_features = self.video_feature_extractor.extract_features(video_path)

        n_text = text_features.shape[0]
        n_audio = audio_features.shape[0]
        n_video = video_features.shape[0]

        if n_text != n_audio or n_text != n_video:
            raise InvalidParameterError(
                f"TR count mismatch: transcripts has {n_text} TRs, but "
                f"audio features have {n_audio} and video features have "
                f"{n_video}. The number of transcript strings must match "
                f"the number of TRs derived from the video "
                f"(floor(video_duration / {self.config.tr}))."
            )

        inputs_dict = {
            "text": text_features[None].float(),
            "audio": audio_features[None].float(),
            "video": video_features[None].float(),
        }
        selected_parcel_indices = self._get_selected_parcel_indices()

        with self._model_session():
            with torch.inference_mode():
                responses = self.model(
                    inputs_dict, torch.as_tensor([self.subject_idx], device=self.device)
                ).squeeze(0)

            if selected_parcel_indices is not None:
                responses = responses[:, selected_parcel_indices]

            return responses

    def generate_glass_brain_animation(
        self,
        responses: torch.Tensor,
        out_path: str,
        display_mode: str = "lyrz",
        cmap: str = "cold_hot",
        cleanup_frames: bool = True,
    ) -> str:
        """
        Save a glass-brain animation from model responses.

        Args:
            responses: Tensor of shape [n_trs, n_parcels] from generate_response.
            out_path: Output path ending with .gif.
            display_mode: Nilearn display mode (e.g. "lyrz").
            cmap: Colormap used for plotting.
            cleanup_frames: If True, temporary PNG frames are removed.

        Returns:
            str: Absolute path to saved animation.
        """
        from nilearn import image, plotting
        from nilearn.maskers import NiftiLabelsMasker
        from PIL import Image

        r = responses.detach().cpu().numpy().astype(np.float32)
        if r.ndim != 2:
            raise ValueError(f"`responses` must be 2D [time, parcels], got shape {r.shape}.")

        selected_idx = self._get_selected_parcel_indices()
        if selected_idx is not None:
            full_r = np.zeros((r.shape[0], self.N_PARCELS_MODEL), dtype=np.float32)
            full_r[:, selected_idx] = r
            r = full_r

        atlas = fetch_atlas_schaefer_2018(
            n_rois=self.N_PARCELS_MODEL, yeo_networks=7, verbose=0
        )
        masker = NiftiLabelsMasker(labels_img=atlas["maps"], standardize=False)
        masker.fit()
        nii = masker.inverse_transform(r)

        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()
        if ext != ".gif":
            raise ValueError("`out_path` must end with .gif.")

        vmax = float(np.percentile(np.abs(r), 99))
        vmin = -vmax
        fps = float(self.config.tr)

        frame_dir = tempfile.mkdtemp(prefix="glass_brain_frames_", dir=os.path.dirname(out_path) or ".")
        frame_paths = []
        frames = []
        try:
            for i in range(r.shape[0]):
                disp = plotting.plot_glass_brain(
                    image.index_img(nii, i),
                    display_mode=display_mode,
                    cmap=cmap,
                    symmetric_cbar=True,
                    plot_abs=False,
                    colorbar=True,
                    vmin=vmin,
                    vmax=vmax,
                    title=f"TR {i + 1}/{r.shape[0]}",
                )
                frame_path = os.path.join(frame_dir, f"frame_{i:05d}.png")
                disp.savefig(frame_path, dpi=120)
                disp.close()
                frame_paths.append(frame_path)

            frames = [Image.open(p).convert("RGB") for p in frame_paths]
            duration_ms = int(round(1000.0 / fps))
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
        finally:
            for frame in frames:
                frame.close()
            if cleanup_frames:
                shutil.rmtree(frame_dir, ignore_errors=True)

        return out_path

    def cleanup(self):
        """Frees memory by unloading models and clearing CUDA cache."""
        self.text_feature_extractor.cleanup()
        self.audio_feature_extractor.cleanup()
        self.video_feature_extractor.cleanup()
        self._cleanup_model()

    def _cleanup_model(self):
        if self.model is not None:
            self.model.cpu()
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique identifier.

        Returns
        -------
        str
            Model ID string from the YAML config.
        """
        return cls.MODEL_ID

    @classmethod
    def get_metadata(
        cls,
        berg_dir=None,
        model_instance=None,
        **kwargs
    ) -> dict[str, Any]:
        # If model_instance is provided, extract parameters from it
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
        # If this method is called on an instance (rather than the class)
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir

        if berg_dir is None:
            return model_info

        file_name = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-fmri',
            'train_dataset-cneuromod_algo2025',
            'model-vibe',
            'metadata',
            'metadata.npy'
        )

        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata

        return model_info

    def _extract_network_names(self):
        """
        Parses the Schaefer 2018 Atlas labels to extract network names.

        Returns:
            np.ndarray: Array of network names corresponding to the 1000 ROIs.
        """
        labels = fetch_atlas_schaefer_2018(
            n_rois=1000, yeo_networks=7, verbose=0)['labels'][1:]
        nets = []
        for s in labels:
            if isinstance(s, bytes):  # just in case
                s = s.decode("utf-8", errors="ignore")
            parts = s.split("_")

            if parts[0] in {"7Networks", "17Networks"}:
                if len(parts) < 3:
                    raise ValueError(
                        f"Unexpected label format (too few parts): {s}")
                net = parts[2]
            else:
                # Fallback for labels without the '7Networks/17Networks' prefix
                if len(parts) < 3:
                    raise ValueError(
                        f"Unexpected label format (no network token): {s}")
                net = parts[1]

            nets.append(net)
        return np.array(nets, dtype=object)