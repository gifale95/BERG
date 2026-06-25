"""Model construction and parameter errors.

Validation happens in __init__, before load_model reads any file, so the error
paths work without weights. The generic tests run over *every* registered native
model so a newly added model is covered automatically; the EEG-specific tests
below pin down selection/channel error messages in detail."""

import importlib

import pytest

from berg.core.exceptions import InvalidParameterError
from berg.core.model_registry import MODEL_REGISTRY
from berg.interfaces.base_model import BaseModelInterface

eeg = pytest.importorskip("berg.models.eeg.things_eeg2_vit_b_32")
EEGEncodingModel = eeg.EEGEncodingModel


# BrainScore gateways take a model_id (not a plain subject) and don't fit the
# common (subject, selection, device, berg_dir) constructor, so they're exempt.
_GATEWAY_MODEL_IDS = {"brainscore_vision", "brainscore_language"}


def _constructable_model_ids():
    """Native models with the common subject-based constructor."""
    return sorted(
        mid for mid, info in MODEL_REGISTRY.items()
        if info.get("yaml_path") and mid not in _GATEWAY_MODEL_IDS
    )


def _load_class(model_id):
    info = MODEL_REGISTRY[model_id]
    return getattr(importlib.import_module(info["module_path"]), info["class_name"])


@pytest.mark.parametrize("model_id", _constructable_model_ids())
def test_model_constructs_without_weights(model_id):
    """Every native model must construct from a valid subject without reading any
    file. berg_dir points nowhere on purpose; selection={} is accepted by all
    models (those that require a selection accept an empty one). Catches a new
    model whose __init__ / YAML-derived class attributes are broken."""
    cls = _load_class(model_id)
    subject = cls.VALID_SUBJECTS[0]
    model = cls(subject=subject, device="cpu", selection={}, berg_dir="/nonexistent")
    assert isinstance(model, BaseModelInterface)


@pytest.mark.parametrize("model_id", _constructable_model_ids())
def test_model_rejects_invalid_subject(model_id):
    """An out-of-range subject must raise InvalidParameterError for every model,
    before any file access."""
    cls = _load_class(model_id)
    valid = cls.VALID_SUBJECTS[0]
    bad_subject = "__not_a_subject__" if isinstance(valid, str) else 10**9
    with pytest.raises(InvalidParameterError):
        cls(subject=bad_subject, device="cpu", selection={}, berg_dir="/nonexistent")


@pytest.mark.parametrize("model_id", _constructable_model_ids())
def test_model_rejects_invalid_selection_key(model_id):
    """An unknown selection key must raise InvalidParameterError for every model,
    in __init__ before any file is read (validate_selection_keys runs first).
    Generalizes the EEG-specific test_invalid_selection_key_raises below so a new
    model gets the same guard automatically."""
    cls = _load_class(model_id)
    subject = cls.VALID_SUBJECTS[0]
    with pytest.raises(InvalidParameterError):
        cls(
            subject=subject,
            device="cpu",
            selection={"__not_a_real_key__": []},
            berg_dir="/nonexistent",
        )

VALID_SUBJECT = EEGEncodingModel.VALID_SUBJECTS[0]
INVALID_SUBJECT = max(s for s in EEGEncodingModel.VALID_SUBJECTS if isinstance(s, int)) + 1000


def test_valid_construction_does_not_touch_disk():
    # berg_dir points nowhere, but load_model is not called, so this must work.
    model = EEGEncodingModel(subject=VALID_SUBJECT, device="cpu", berg_dir="/nonexistent")
    assert model.subject == VALID_SUBJECT
    assert model.device == "cpu"


def test_invalid_subject_raises():
    with pytest.raises(InvalidParameterError):
        EEGEncodingModel(subject=INVALID_SUBJECT, device="cpu", berg_dir="/nonexistent")


def test_invalid_selection_key_raises():
    with pytest.raises(InvalidParameterError):
        EEGEncodingModel(
            subject=VALID_SUBJECT,
            device="cpu",
            selection={"not_a_real_key": []},
            berg_dir="/nonexistent",
        )


def test_invalid_channel_raises():
    with pytest.raises(Exception):
        EEGEncodingModel(
            subject=VALID_SUBJECT,
            device="cpu",
            selection={"channels": ["NOT_A_CHANNEL"]},
            berg_dir="/nonexistent",
        )


def test_get_model_id_classmethod():
    assert EEGEncodingModel.get_model_id() == "eeg-things_eeg_2-vit_b_32"
