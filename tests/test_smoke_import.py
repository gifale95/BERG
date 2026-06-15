"""Smoke tests: the package imports and models register. Since models register
on import, most breakage shows up here without needing any weights."""

import berg
from berg.core.model_registry import MODEL_REGISTRY


def test_package_imports():
    assert berg.__version__


def test_registry_is_populated():
    # Even on a minimal install, the lightweight models must register.
    assert len(MODEL_REGISTRY) > 0


def test_core_lightweight_models_present():
    """These only need base deps, so they should always register. A missing one
    usually means an internal import broke."""
    expected = {
        "eeg-things_eeg_2-vit_b_32",
        "eeg-things_eeg_2-alexnet",
        "fmri-nsd_fsaverage-vit_b_32",
        "fmri-nsd_fsaverage-alexnet",
    }
    missing = expected - set(MODEL_REGISTRY)
    assert not missing, f"core models failed to register: {sorted(missing)}"


def test_registry_entries_well_formed():
    for model_id, info in MODEL_REGISTRY.items():
        assert info["module_path"], f"{model_id} has no module_path"
        assert info["class_name"], f"{model_id} has no class_name"
