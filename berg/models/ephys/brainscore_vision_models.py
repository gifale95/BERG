import os
import sys
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Union
import pickle
from PIL import Image
import shutil

from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model, MODEL_REGISTRY
from berg.core.parameter_validator import validate_selection_keys, validate_roi
from berg.core.exceptions import InvalidParameterError

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# Region to benchmark mapping
REGION_BENCHMARKS = {
    'V1': 'FreemanZiemba2013public.V1-pls',
    'V2': 'FreemanZiemba2013public.V2-pls',
    'V4': 'MajajHong2015public.V4-pls',
    'IT': 'MajajHong2015public.IT-pls'
}

BRAINSCORE_INSTALL_MSG = """
BrainScore is not installed. To use BrainScore vision models:

    pip install berg[brainscore]

Note: BrainScore requires Python 3.11.
      You are currently running Python {major}.{minor}.

For more information, see: https://www.brain-score.org
""".strip()


def _check_brainscore_available():
    """
    Check if BrainScore vision is importable and raise a clear error if not.
    Called lazily inside methods that need BrainScore, not at module level.
    """
    try:
        import brainscore_vision 
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
        "brainscore_vision.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register the gateway model
register_model(
    model_id="brainscore_vision",
    module_path="berg.models.ephys.brainscore_vision_models",
    class_name="BrainScoreGateway",
    modality="ephys",
    training_dataset="BrainScore_Vision",
    yaml_path="berg/models/model_cards/brainscore_vision.yaml"
)


class BrainScoreGateway(BaseModelInterface):
    """
    Gateway to BrainScore vision models.
    
    This class provides a unified interface to access BrainScore's collection
    of neural network models trained on biological neural recordings.
    """

    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]

    def __init__(
        self,
        berg_dir: str,
        model_id: str,
        device: str = "auto",
        selection: dict = None
    ):
        """
        Initialize BrainScore gateway.

        Parameters
        ----------
        berg_dir : str
            Path to BERG directory
        model_id : str
            Model identifier in format "brainscore_vision-{model_name}",
        device : str
            Device for computation ("cpu", "cuda", or "auto")
        selection : dict
            Selection parameters (roi is required):
                - roi: str (e.g., "V1", "V4", "IT")
        """
        self.berg_dir = berg_dir
        self.model_id_full = model_id
        self.device = device
        self.selection = selection

        # Validate parameters
        self._validate_parameters()
        
        # Parse BrainScore model name from model_id
        if model_id.startswith("brainscore_vision-"):
            self.brainscore_model_name = model_id.replace("brainscore_vision-", "")
        else:
            self.brainscore_model_name = model_id

        # Parse selection parameters (roi is guaranteed to exist after validation)
        self.roi = selection['roi']

        # Set up paths
        self.model_dir = (
            Path(berg_dir)
            / "encoding_models"
            / "modality-ephys"
            / "train_dataset-brainscore_vision"
            / f"model-{self.brainscore_model_name}"
        )
        self.weights_dir = self.model_dir / "encoding_models_weights"
        self.temp_dir = self.model_dir / "temp"

        # Initialize model and regression
        self.model = None
        self.regression = None
        self.time_bins = None
        self.temp_images_created = False

    def _validate_parameters(self):
        """
        Validate user-provided parameters.
        ROI selection is REQUIRED for BrainScore models.
        """
        # Check that selection exists and contains roi
        if self.selection is None or 'roi' not in self.selection:
            raise InvalidParameterError(
                f"ROI selection is required for BrainScore models. "
                f"Available ROIs: {self.VALID_ROIS}"
            )
        
        # Validate selection keys
        validate_selection_keys(self.selection, self.SELECTION_KEYS)
        
        # Validate ROI
        if "roi" in self.selection:
            self.roi = validate_roi(self.selection["roi"], self.VALID_ROIS)

    def _get_regression_cache_path(self) -> Path:
        """Get path to cached regression weights for current model and ROI."""
        return self.weights_dir / f"{self.brainscore_model_name}_{self.roi}_regression.pkl"

    def _train_and_cache_regression(self):
        """
        Train PLS regression on benchmark data and cache it.
        Also caches time_bins for later use.
        """
        from brainscore_vision import load_benchmark
        from brainscore_vision.metrics.regression_correlation.metric import pls_regression

        print(f"Training regression for {self.roi} region...")
        print("This will take ~3 minutes (only done once, then cached)")
        
        # Load benchmark
        benchmark_id = REGION_BENCHMARKS[self.roi]
        benchmark = load_benchmark(benchmark_id)
        
        # Use benchmark's time bins
        self.time_bins = benchmark.timebins
        
        # Configure model recording
        self.model.start_recording(benchmark.region, time_bins=self.time_bins)
        
        # Get model activations for benchmark stimuli
        benchmark_activations = self.model.look_at(benchmark._assembly.stimulus_set)
        
        # Squeeze time_bin if present and singleton
        if "time_bin" in benchmark_activations.dims and \
           benchmark_activations.sizes["time_bin"] == 1:
            benchmark_activations = benchmark_activations.squeeze("time_bin")
        
        # Train regression
        regression = pls_regression()
        regression.fit(benchmark_activations, benchmark._assembly)
        
        # Cache regression AND time_bins together
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_regression_cache_path()
        
        cache_data = {
            'regression': regression,
            'time_bins': self.time_bins
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Regression trained and cached at: {cache_path}")
        return regression

    def _load_or_train_regression(self):
        """Load cached regression if available, otherwise train and cache."""
        from brainscore_vision import load_benchmark

        cache_path = self._get_regression_cache_path()

        if cache_path.exists():
            print(f"Loading cached regression from: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            if isinstance(cache_data, dict):
                self.regression = cache_data['regression']
                self.time_bins = cache_data['time_bins']
            else:
                # Old cache format — just the regression
                self.regression = cache_data
                benchmark_id = REGION_BENCHMARKS[self.roi]
                benchmark = load_benchmark(benchmark_id)
                self.time_bins = benchmark.timebins

            print("Regression loaded from cache")
        else:
            print(f"No cached regression found for {self.brainscore_model_name} + {self.roi}")
            self.regression = self._train_and_cache_regression()

    def load_model(self):
        """Load the BrainScore model and regression weights."""
        _check_brainscore_available()

        from brainscore_vision import load_model

        print(f"Loading BrainScore model: {self.brainscore_model_name}")
        self.model = load_model(self.brainscore_model_name)
        print("Model loaded")

        self._load_or_train_regression()

    def _numpy_to_temp_paths(self, images: np.ndarray) -> List[str]:
        """
        Convert numpy array to temporary image files.

        Parameters
        ----------
        images : np.ndarray
            Images in shape (batch, 3, H, W) with values [0, 255]

        Returns
        -------
        List[str]
            List of paths to temporary image files
        """
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_images_created = True
        
        # Convert from (batch, 3, H, W) to (batch, H, W, 3)
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        
        # Ensure uint8
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)
        
        # Save each image
        image_paths = []
        for i, img in enumerate(images):
            img_path = self.temp_dir / f"image_{i:05d}.png"
            Image.fromarray(img).save(img_path)
            image_paths.append(str(img_path))

        return image_paths

    def _cleanup_temp_images(self):
        """Remove temporary image directory."""
        if self.temp_images_created and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_images_created = False

    def generate_response(
        self,
        stimulus: Union[str, List[str], np.ndarray],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate neural responses for given stimuli.

        Parameters
        ----------
        stimulus : str, list[str], or np.ndarray
            Either:
            - Path to image directory (str)
            - List of image file paths (list[str])
            - Numpy array of images (batch, 3, H, W) with values [0, 255]
        show_progress : bool
            Whether to show progress messages

        Returns
        -------
        np.ndarray
            Neural responses, shape (n_images, n_neurons)
        """
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

        if isinstance(stimulus, np.ndarray):
            # Convert numpy array to temporary image files
            if show_progress:
                print(f"Converting {len(stimulus)} numpy images to temporary files...")
            image_paths = self._numpy_to_temp_paths(stimulus)

        elif isinstance(stimulus, str):
            # Directory path - glob for images
            image_paths = sorted(Path(stimulus).glob("*.jpg")) + \
                         sorted(Path(stimulus).glob("*.png")) + \
                         sorted(Path(stimulus).glob("*.jpeg"))
            image_paths = [str(p) for p in image_paths]

        else:
            # List of paths
            image_paths = [str(Path(p)) for p in stimulus]
        
        # Create StimulusSet
        stimulus_ids = [Path(p).stem for p in image_paths]
        stimuli_df = pd.DataFrame({
            "stimulus_id": stimulus_ids,
            "filename": [Path(p).name for p in image_paths]
        })

        stimulus_set = StimulusSet(stimuli_df)
        stimulus_set.stimulus_paths = {
            sid: str(Path(path).absolute())
            for sid, path in zip(stimulus_ids, image_paths)
        }
        
        # Get model activations
        if show_progress:
            print(f"Extracting activations for {len(image_paths)} images from {self.roi} region...")
        
        # Configure model recording with the same time bins used during training
        self.model.start_recording(self.roi, time_bins=self.time_bins)
        
        activations = self.model.look_at(stimulus_set)
        
        # Squeeze time_bin if present
        if "time_bin" in activations.dims and activations.sizes["time_bin"] == 1:
            activations = activations.squeeze("time_bin")
        
        # Predict neural responses using cached regression
        if show_progress:
            print("Predicting neural responses...")
        
        predicted_responses = self.regression.predict(activations)
        
        # Cleanup temp images if we created them
        self._cleanup_temp_images()

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
        model_instance: 'BrainScoreGateway' = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get model metadata."""
        print("BrainScore does not provide metadata. Please check their website for more model information: https://www.brain-score.org/vision/leaderboard/")
        return {}

    def cleanup(self):
        """Release resources and cleanup temp files."""
        self._cleanup_temp_images()
        self.model = None
        self.regression = None


def discover_brainscore_models() -> List[str]:
    """
    Discover all available BrainScore vision models.
    Returns
    -------
    List[str]
        Sorted list of model names usable with 'brainscore_vision-{name}'.

    Raises
    ------
    ImportError
        If BrainScore is not installed.
    """
    _check_brainscore_available()

    import pkgutil
    import brainscore_vision.models as bs_models

    models = []
    for importer, modname, ispkg in pkgutil.iter_modules(bs_models.__path__):
        if not modname.startswith('_'):
            models.append(modname)

    return sorted(models)