import os
import shutil
import tempfile
import numpy as np
import yaml
import torch
import logging
from typing import Any
from nilearn.datasets import fetch_atlas_schaefer_2018
from berg.core.exceptions import InvalidParameterError
from berg.core.model_registry import register_model
from berg.core.parameter_validator import validate_roi, validate_selection_keys, validate_subjects, validate_binary_array
from berg.interfaces.base_model import BaseModelInterface
from berg.models.fmri.text2fmri_utils.config import Text2fMRIConfig, get_pretrained_model_configs
from berg.models.fmri.text2fmri_utils.feature_extractor import FeatureExtractor
from berg.models.fmri.text2fmri_utils.model import Text2fMRIModel


# Load model info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(
        __file__), "..", "model_cards", "fmri-cneuromod_algo2025-text2fmri.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.cneuromod_algo2025_text2fmri",
    class_name="Text2fMRI",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "cneuromod_algo2025"),
    yaml_path=os.path.join(os.path.dirname(__file__),
                           "..", "model_cards", "fmri-cneuromod_algo2025-text2fmri.yaml")
)

# Algonauts Mapping
SUBJECT_MAPPING = {1: 0,
                   2: 1,
                   3: 2,
                   5: 3,}


class Text2fMRI(BaseModelInterface):
    """
    An interface for the Text2fMRI model, predicting brain activity from natural language.

    This model takes transcribed speech (text aligned with fMRI TRs) and predicts 
    whole-brain fMRI activity using a LLM-based feature extractor followed by a 
    transformer encoder.

    Attributes:
        feature_extractor (FeatureExtractor): Handles text tokenization and LLM feature extraction.
        model (Text2fMRIModel): The mapping model from LLM features to brain space.
        config (Text2fMRIConfig): Hyperparameters for the model architecture.
        selection (dict): User selection criteria (e.g., specific ROIs).
        subject (int): The subject ID for predictions.
    """

    MODEL_ID = model_info["model_id"]
    # Extract any validation info from model_info
    _collection_slug = "ShreyDixit/text2fmri"
    SELECTION_KEYS = list(
        model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    N_PARCELS_MODEL = 1000  # Model output size (1000 ROIs from Schaefer 2018 atlas)

    # Setting it to None so that no calls to hugging face are made during the import
    pretrained_configs_dict = None

    def __init__(self, berg_dir: str = None, selection: dict = None, subject: int = None, config: Text2fMRIConfig = Text2fMRIConfig(), device: str = "auto", low_mem_use: bool = False, model_variant: str = None):
        """
        Initialize the Text2fMRI interface.

        Args:
            berg_dir (str, optional): Path to the BERG cache directory.
            selection (dict, optional): Dictionary specifying output filters.
                To filter by brain region, include the key "roi" with a value corresponding 
                to a network name from the Schaefer 2018 atlas (7-network parcellation).
                Valid values include: 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 
                'Limbic', 'Cont', or 'Default'.
            subject (int): The subject ID (index) to generate predictions for.
            config (Text2fMRIConfig): Configuration object for the model architecture.
            device (str): Computation device ('cpu', 'cuda', 'auto').
            low_mem_use : bool
                If True, use a low-memory usage. The model and the LLMFeatureExtractor will not be loaded at the same time.
                This will take less memory but it will take longer to generate the responses because it will load the models
                every time.
            model_variant (str, optional): HuggingFace repo ID of a specific pretrained
                variant to load (e.g. "ShreyDixit/Text2fMRI-Qwen-2.5-3B").
                If None, uses the default configuration (Qwen-2.5-0.5B).
                Use model.get_pretrained_variants() to see all available options.
        """
        self.model = None
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.feature_extractor: FeatureExtractor = FeatureExtractor(
            config=config, device=device, berg_dir=berg_dir)
        self.config: Text2fMRIConfig = config
        self.berg_dir = berg_dir
        self.selection = selection
        self.subject = subject
        self.model_variant = model_variant
        
        
        
        self.voxel_index = None
        if Text2fMRI.pretrained_configs_dict is None:
            Text2fMRI.get_pretrained_configs()
        self._validate_parameters()
        self.subject = SUBJECT_MAPPING[subject]
        self.roi = self.selection.get(
            "roi", None) if self.selection is not None else None
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

            if "roi" in self.selection:
                self.roi = validate_roi(self.selection["roi"], self.VALID_ROIS)

            if "voxel_index" in self.selection:
                parcel_index = self.selection["voxel_index"]
                # Validate as binary array
                validated_array = validate_binary_array(
                    parcel_index,
                    expected_length=self.N_PARCELS_MODEL,
                    parameter_name="voxel_index"
                )
                self.voxel_index = validated_array.astype(bool)

        # Validate subject
        self.subject = validate_subjects(
            self.subject, self.VALID_SUBJECTS
        )[0]

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
    def get_pretrained_configs(cls) -> list[Text2fMRIConfig]:
        """
        A list of all pretrained configs available on HuggingFace. Choose from this list to get a pretrained model.

        Returns
        -------
        List[Text2fMRIConfig]:
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

        Use this to discover which models are available, then pass the chosen
        repo ID as model_variant to get_encoding_model().

        Returns
        -------
        list[str]
            A list of HuggingFace repo IDs, e.g.:
            ["ShreyDixit/Text2fMRI-Qwen-2.5-0.5B", "ShreyDixit/Text2fMRI-Qwen-2.5-3B"]

        Examples
        --------
        >>> model = berg.get_encoding_model(model_id, subject=1)
        >>> variants = model.get_pretrained_variants()
        >>> model = berg.get_encoding_model(model_id, subject=1, model_variant=variants[1])
        """
        if cls.pretrained_configs_dict is None:
            cls.get_pretrained_configs()
        return list(cls.pretrained_configs_dict.values())

    def load_model(self, load_feature_extractor=True):
        """
        Loads the neural network weights and optionally the LLM backbone.

        Args:
            load_feature_extractor (bool): If True, loads the heavy LLM into memory.
        """
        if load_feature_extractor:
            self.feature_extractor.load_model()
        self.model = Text2fMRIModel(config=self.config, device=self.device, berg_dir=self.berg_dir)

        if self.config in self.pretrained_configs_dict:
            repo_id = self.model_variant if self.model_variant is not None else self.pretrained_configs_dict[self.config]
            self.model.load_model_from_hub(repo_id)
        self.model.eval()

    def _get_selected_parcel_indices(self):
        if self.roi is None and self.voxel_index is None:
            return None
        mask = np.zeros(self.N_PARCELS_MODEL, dtype=bool)
        if self.roi is not None:
            mask |= np.isin(self.roi_labels, self.roi)
        if self.voxel_index is not None:
            mask |= self.voxel_index
        return np.where(mask)[0]

    @torch.no_grad()
    def generate_response(self, stimulus: list[str]) -> torch.Tensor:
        """
        Generates in silico neural responses (fMRI TRs) for a given text stimulus.

        Args:
            stimulus (List[str]): A list of strings, where each string corresponds 
                to the transcript of one fMRI TR (Time Repetition).

        Returns:
            torch.Tensor: Predicted brain activity. 
                Shape: [num_timepoints, num_rois] (or subset of ROIs if selected).
        """
        features = self.feature_extractor.extract_features(stimulus)
        if self.low_mem_use:
            self.feature_extractor.cleanup()

        if self.model is None:
            self.load_model(load_feature_extractor=not self.low_mem_use)

        roi_indices = self._get_selected_parcel_indices()

        with torch.inference_mode():
            responses = self.model(
                features[None], 
                torch.as_tensor([self.subject], device=self.device),
                roi_indices=roi_indices
            ).squeeze()

        if self.low_mem_use:
            self.cleanup()

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
        self.feature_extractor.cleanup()
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
        """
        Retrieve metadata for the model.

        Parameters
        ----------
        berg_dir : str, optional
            Path to BERG directory.
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        **kwargs
            Additional parameters (subject is ignored as metadata is shared across subjects).

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """
        # If model_instance is provided, extract parameters from it
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
        # If this method is called on an instance (rather than the class)
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir

        # Validate required parameters
        if berg_dir is None:
            raise InvalidParameterError("Required parameter missing: berg_dir")

        # Build metadata path
        file_name = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-fmri',
            'train_dataset-cneuromod_algo2025',
            'model-text2fmri',
            'metadata',
            'metadata.npy'
        )

        # Load metadata if file exists
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found: {file_name}")

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
