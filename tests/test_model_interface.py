"""Interface checks over every registered model. Doesn't prove a model produces
correct responses (that needs weights), but confirms each class loads,
implements the interface, reports the right ID, and has a renderable card."""

import pytest

import berg
from berg.core.model_registry import MODEL_REGISTRY, get_model_class
from berg.interfaces.base_model import BaseModelInterface

# Every model that registered with the currently installed dependencies.
ALL_MODEL_IDS = sorted(MODEL_REGISTRY)

REQUIRED_METHODS = (
    "load_model",
    "generate_response",
    "get_metadata",
    "get_model_id",
    "cleanup",
)


@pytest.mark.parametrize("model_id", ALL_MODEL_IDS)
def test_class_loads_and_subclasses_interface(model_id):
    cls = get_model_class(model_id)
    assert issubclass(cls, BaseModelInterface)


@pytest.mark.parametrize("model_id", ALL_MODEL_IDS)
def test_class_is_concrete(model_id):
    """No leftover abstract methods, so the model can be instantiated."""
    cls = get_model_class(model_id)
    unimplemented = getattr(cls, "__abstractmethods__", set())
    assert not unimplemented, f"{model_id} missing implementations: {sorted(unimplemented)}"


@pytest.mark.parametrize("model_id", ALL_MODEL_IDS)
def test_required_methods_present(model_id):
    cls = get_model_class(model_id)
    for method in REQUIRED_METHODS:
        assert callable(getattr(cls, method, None)), f"{model_id} missing {method}()"


@pytest.mark.parametrize("model_id", ALL_MODEL_IDS)
def test_get_model_id_matches_registry(model_id):
    assert get_model_class(model_id).get_model_id() == model_id


@pytest.mark.parametrize("model_id", ALL_MODEL_IDS)
def test_describe_returns_dict_via_public_api(model_id, capsys):
    """BERG.describe() should return the info dict, not None. Goes through the
    public API on purpose — a missing `return` only shows up that way."""
    info = berg.BERG(berg_dir="/nonexistent").describe(model_id)
    assert isinstance(info, dict), "BERG.describe() returned None instead of the info dict"
    assert info["model_id"] == model_id
    assert "parameters" in info
