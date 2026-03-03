import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import yaml

from berg.core.exceptions import InvalidParameterError
from berg.core.model_registry import register_model
from berg.core.parameter_validator import validate_subject
from berg.interfaces.base_model import BaseModelInterface

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# Single benchmark for all language models
BENCHMARK_ID = "Pereira2018_384sentences"

BRAINSCORE_INSTALL_MSG = """
BrainScore is not installed. To use BrainScore language models:

    pip install berg[brainscore]

Note: BrainScore requires Python 3.11.
      You are currently running Python {major}.{minor}.

For more information, see: https://www.brain-score.org
""".strip()


def _check_brainscore_available():
    """
    Check if BrainScore language is importable and raise a clear error if not.
    Called lazily inside methods that need BrainScore, not at module level.
    """
    try:
        import brainscore_language  # noqa: F401
    except ImportError:
        raise ImportError(
            BRAINSCORE_INSTALL_MSG.format(
                major=sys.version_info.major,
                minor=sys.version_info.minor
            )
        )


def load_model_info():
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "brainscore_language.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at module level
model_info = load_model_info()

# Register the gateway
register_model(
    model_id="brainscore_language",
    module_path="berg.models.fmri.brainscore_language_models",
    class_name="BrainScoreLanguageGateway",
    modality="fMRI",
    training_dataset="BrainScore_Language",
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "brainscore_language.yaml")
)


class BrainScoreLanguageGateway(BaseModelInterface):
    """
    Gateway to BrainScore language models.

    Provides access to BrainScore's GPT-family language models trained against
    human fMRI recordings from the Pereira 2018 dataset (384 sentences,
    12,155 voxels across 9 subjects).

    Requires BrainScore to be installed: pip install berg[brainscore]
    BrainScore requires Python 3.11.

    Workflow
    --------
    1. Load language model (e.g., gpt2) via BrainScore API
    2. Run model on Pereira2018 benchmark sentences to get representations
    3. Train PLS regression: model representations → fMRI voxel responses
    4. Cache regression weights for fast subsequent predictions
    5. For new sentences: extract representations → predict voxel responses
    """

    MODEL_ID = model_info["model_id"]
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]

    def __init__(
        self,
        berg_dir: str,
        model_id: str,
        subject: str,
        device: str = "auto",
    ):
        """
        Initialize BrainScore language gateway.

        Parameters
        ----------
        berg_dir : str
            Path to BERG directory
        model_id : str
            Model identifier in format "brainscore_language-{model_name}"
        subject : str
            Subject ID (e.g., '018'). Required — filters to that subject's
            voxels (~1,350 voxels).
            Valid values: ['018', '199', '288', '289', '296', '343', '366', '407', '426']
        device : str
            Unused (BrainScore language models manage their own device).
            Kept for API consistency with other BERG models.
        """
        self.berg_dir = berg_dir
        self.model_id_full = model_id
        self.subject = subject
        self.device = device

        # Validate parameters
        self._validate_parameters()

        # Parse BrainScore model name from model_id
        prefix = "brainscore_language-"
        if model_id.startswith(prefix):
            self.brainscore_model_name = model_id[len(prefix):]
        else:
            self.brainscore_model_name = model_id

        self.subject_tag = self.subject

        # Set up paths
        self.model_dir = (
            Path(berg_dir)
            / "encoding_models"
            / "modality-fmri"
            / "train_dataset-brainscore"
            / f"model-{self.brainscore_model_name}"
        )
        self.weights_dir = self.model_dir / "encoding_models_weights"

        self.model = None
        self.regression = None

    def _validate_parameters(self):
        """Validate subject parameter. Subject is required."""
        validate_subject(self.subject, self.VALID_SUBJECTS)

    def _get_regression_cache_path(self) -> Path:
        """
        Get path to cached regression weights for the current model and subject.

        Examples
        --------
        gpt2, subject='018'  → gpt2_018_pereira384_regression.pkl
        gpt2, subject='199'  → gpt2_199_pereira384_regression.pkl
        """
        filename = f"{self.brainscore_model_name}_{self.subject_tag}_pereira384_regression.pkl"
        return self.weights_dir / filename

    def _get_benchmark_assembly(self):
        """
        Load the Pereira2018_384sentences benchmark filtered to the selected subject.

        Returns
        -------
        xarray.DataArray
            Neural assembly with shape (n_sentences, n_voxels) for the selected subject
            (~1,350 voxels).
        """
        from brainscore_language.benchmarks.pereira2018 import Pereira2018_384sentences
        benchmark = Pereira2018_384sentences()
        assembly = benchmark.data
        subject_mask = assembly['subject'] == self.subject
        return assembly.sel(neuroid=subject_mask)

    def _train_and_cache_regression(self):
        """
        Train PLS regression mapping model representations → fMRI voxels.

        Steps
        -----
        1. Load benchmark assembly (optionally filtered by subject)
        2. Run model on benchmark sentences to get neural representations
           - Shape: (n_sentences, n_features)
        3. Assign stimulus_id coordinate so PLS can align presentations
        4. Fit PLS regression: representations → voxel responses
        5. Cache the trained regression object to disk

        Returns
        -------
        PLSRegression
            Trained regression object.
        """
        from brainscore_vision.metrics.regression_correlation.metric import (
            pls_regression,
        )

        print(f"Training regression for {self.brainscore_model_name} "
              f"(subject={self.subject_tag}, benchmark={BENCHMARK_ID})...")
        print("This will take a few minutes (only done once, then cached)")

        assembly = self._get_benchmark_assembly()
        benchmark_sentences = assembly['sentence'].values

        # Configure model for fMRI recording
        self.model.start_neural_recording(
            recording_target=self.model.RecordingTarget.language_system,
            recording_type=self.model.RecordingType.fMRI
        )

        # Get model representations for benchmark sentences
        # Output: dict with key 'neural', value is xarray.DataArray
        #         shape (n_sentences, n_features)
        model_output = self.model.digest_text(benchmark_sentences)
        model_reps = model_output['neural']

        # Attach stimulus_id so PLS regression can align presentations
        # benchmark.data uses IDs like '384sentences.0', '384sentences.1', ...
        model_reps['stimulus_id'] = ('presentation', assembly['stimulus_id'].values)

        # Train PLS regression: model_reps → assembly (voxel responses)
        regression = pls_regression()
        regression.fit(model_reps, assembly)

        # Cache to disk
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_regression_cache_path()

        with open(cache_path, 'wb') as f:
            pickle.dump(regression, f)

        print(f"Regression trained and cached at: {cache_path}")
        return regression

    def _load_or_train_regression(self):
        """
        Load cached regression if available, otherwise train and cache.
        """
        cache_path = self._get_regression_cache_path()

        if cache_path.exists():
            print(f"Loading cached regression from: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.regression = pickle.load(f)
            print("Regression loaded from cache")
        else:
            print(f"No cached regression found for "
                  f"{self.brainscore_model_name} + subject={self.subject_tag}")
            self.regression = self._train_and_cache_regression()

    def load_model(self):
        """Load the BrainScore language model and regression weights."""
        _check_brainscore_available()

        from brainscore_language import load_model

        print(f"Loading BrainScore language model: {self.brainscore_model_name}")
        self.model = load_model(self.brainscore_model_name)
        print("Model loaded")

        self._load_or_train_regression()

    def generate_response(
        self,
        stimulus: Union[str, List[str]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate fMRI voxel responses for input sentences.

        Parameters
        ----------
        stimulus : str or list[str]
            One or more sentences to encode.
            - Single string: "The cat sat on the mat."
            - List of strings: ["Sentence one.", "Sentence two."]
        show_progress : bool
            Whether to print progress messages.

        Returns
        -------
        np.ndarray
            Predicted BOLD responses, shape (n_sentences, n_voxels).
            n_voxels ~1,350 for the selected subject.

        Notes
        -----
        Stimulus IDs assigned here ('pred_0', 'pred_1', ...) are arbitrary
        labels required by the regression API. They do not correspond to
        any benchmark stimulus IDs.
        """
        # Coerce single string to list
        sentences = [stimulus] if isinstance(stimulus, str) else list(stimulus)

        if show_progress:
            print(f"Encoding {len(sentences)} sentence(s)...")

        # Configure model for fMRI recording
        self.model.start_neural_recording(
            recording_target=self.model.RecordingTarget.language_system,
            recording_type=self.model.RecordingType.fMRI
        )

        # Extract model representations
        model_output = self.model.digest_text(sentences)
        model_reps = model_output['neural']

        # Assign stimulus_id (required by regression API)
        model_reps['stimulus_id'] = (
            'presentation',
            [f"pred_{i}" for i in range(len(sentences))]
        )

        # Predict voxel responses
        if show_progress:
            print("Predicting voxel responses...")

        predicted_responses = self.regression.predict(model_reps)

        if show_progress:
            print(f"Generated responses: {predicted_responses.shape}")

        return predicted_responses.values

    @classmethod
    def get_model_id(cls) -> str:
        """Return the model's unique identifier."""
        return cls.MODEL_ID

    @classmethod
    def get_metadata(
        cls,
        berg_dir: str = None,
        model_instance: 'BrainScoreLanguageGateway' = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get model metadata.
        """
        print("BrainScore does not provide metadata. Please check their website for more model information: https://www.brain-score.org/vision/leaderboard/")
        return {}

    def cleanup(self):
        """Release resources."""
        self.model = None
        self.regression = None


def discover_brainscore_language_models() -> List[str]:
    """
    Discover all available BrainScore language models via the model registry.

    Returns
    -------
    List[str]
        Sorted list of model names usable with 'brainscore_language-{name}'.

    Raises
    ------
    ImportError
        If BrainScore is not installed.
    """
    _check_brainscore_available()

    import brainscore_language.models.gpt  # trigger registrations

    return sorted(model_registry.keys())