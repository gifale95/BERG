import os
import numpy as np
import yaml
import torch
import logging
from typing import Any
from nilearn.datasets import fetch_atlas_schaefer_2018

from berg.core.exceptions import InvalidParameterError
from berg.core.model_registry import register_model
from berg.core.parameter_validator import validate_roi, validate_selection_keys
from berg.interfaces.base_model import BaseModelInterface
from berg.models.fmri.text2fmri_utils.config import Text2fMRIConfig, get_pretrained_model_configs
from berg.models.fmri.text2fmri_utils.feature_extractor import FeatureExtractor
from berg.models.fmri.text2fmri_utils.model import Text2fMRIModel


# Load model info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(
        __file__), "..", "model_cards", "text2fmri.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.text2fmri",
    class_name="Text2fMRI",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "CNeuroMods"),
    yaml_path=os.path.join(os.path.dirname(__file__),
                           "..", "model_cards", "text2fmri.yaml")
)


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
    """

    MODEL_ID = model_info["model_id"]
    # Extract any validation info from model_info
    _collection_slug = "ShreyDixit/text2fmri"
    SELECTION_KEYS = list(
        model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]

    # Setting it to None so that no calls to hugging face are made during the import
    pretrained_configs_dict = None

    def __init__(self, berg_dir: str = None, selection: dict = None, config: Text2fMRIConfig = Text2fMRIConfig(), device: str = "auto", low_mem_use: bool = False):
        """
        Initialize the Text2fMRI interface.

        Args:
            berg_dir (str, optional): Path to the BERG cache directory.
            selection (dict, optional): Dictionary specifying output filters.
                To filter by brain region, include the key "roi" with a value corresponding 
                to a network name from the Schaefer 2018 atlas (7-network parcellation).
                Valid values include: 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 
                'Limbic', 'Cont', or 'Default'.
            config (Text2fMRIConfig): Configuration object for the model architecture.
            device (str): Computation device ('cpu', 'cuda', 'auto').
            low_mem_use : bool
                If True, use a low-memory usage. The model and the LLMFeatureExtractor will not be loaded at the same time.
                This will take less memory but it will take longer to generate the responses because it will load the models
                every time.
        """
        self.model = None
        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.feature_extractor: FeatureExtractor = FeatureExtractor(
            config=config, device=device)
        self.config: Text2fMRIConfig = config
        self.berg_dir = berg_dir
        self.selection = selection
        if Text2fMRI.pretrained_configs_dict is None:
            Text2fMRI.get_pretrained_configs()
        self._validate_parameters()
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

            # Individual validations
            if "roi" in self.selection:
                self.roi = validate_roi(
                    self.selection["roi"], self.VALID_ROIS
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

    def load_model(self, load_feature_extractor=True):
        """
        Loads the neural network weights and optionally the LLM backbone.

        Args:
            load_feature_extractor (bool): If True, loads the heavy LLM into memory.
        """
        if load_feature_extractor:
            self.feature_extractor.load_model()
        self.model = Text2fMRIModel(config=self.config, device=self.device)

        if self.config in self.pretrained_configs_dict:
            self.model.load_model(self.pretrained_configs_dict)
        self.model.eval()

    @torch.no_grad()
    def generate_response(self, stimulus: list[str], subject: int = 1) -> torch.Tensor:
        """
        Generates in silico neural responses (fMRI TRs) for a given text stimulus.

        Args:
            stimulus (List[str]): A list of strings, where each string corresponds 
                to the transcript of one fMRI TR (Time Repetition).
            subject (int): The subject ID (index) to generate predictions for.

        Returns:
            torch.Tensor: Predicted brain activity. 
                Shape: [num_timepoints, num_rois] (or subset of ROIs if selected).
        """

        if subject < 0 or subject >= self.config.num_subjects:
            raise InvalidParameterError(
                f"Subject ID must be in the range [0, {self.config.num_subjects})")

        features = self.feature_extractor.extract_features(stimulus)
        if self.low_mem_use:
            self.feature_extractor.cleanup()

        if self.model is None:
            self.load_model(load_feature_extractor=not self.low_mem_use)

        with torch.inference_mode():
            responses = self.model(features[None], torch.as_tensor(
                [subject], device=self.device)).squeeze()

        if self.roi is not None:
            responses = responses[:, self.roi_labels == self.roi]

        if self.low_mem_use:
            self.cleanup()

        return responses

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
    def get_metadata(cls, berg_dir=None, model_instance=None, roi=None, **kwargs) -> dict[str, Any]:
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
