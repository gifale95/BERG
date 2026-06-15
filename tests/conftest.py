"""Shared fixtures. Most tests here run without the S3 model weights; the ones
that need them are marked `weights`."""

import numpy as np
import pytest

import berg
from berg.core.model_registry import MODEL_REGISTRY


# Whatever registered with the currently installed deps (varies by install).
REGISTERED_MODEL_IDS = sorted(MODEL_REGISTRY.keys())

# Native models have a YAML card; the BrainScore gateway entries don't.
NATIVE_MODEL_IDS = sorted(
    mid for mid, info in MODEL_REGISTRY.items() if info.get("yaml_path")
)


@pytest.fixture(scope="session")
def registered_model_ids():
    return REGISTERED_MODEL_IDS


@pytest.fixture(scope="session")
def native_model_ids():
    return NATIVE_MODEL_IDS


@pytest.fixture
def dummy_images():
    """A tiny batch of valid image stimuli: (batch, 3, 224, 224) uint8."""
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, (2, 3, 224, 224)).astype(np.uint8)
