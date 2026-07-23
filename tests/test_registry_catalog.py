"""Catalog / list / get_model_class. No weights needed."""

import pytest

import berg
from berg.core.model_registry import (
    get_available_models,
    get_model_class,
    MODEL_REGISTRY,
)


@pytest.fixture(scope="module")
def berg_instance():
    # berg_dir is never touched by catalog/list/describe — these read the
    # registry and YAML cards only.
    return berg.BERG(berg_dir="/nonexistent")


def test_get_available_models_matches_registry():
    assert set(get_available_models()) == set(MODEL_REGISTRY)


def test_get_model_catalog_groups_by_modality(berg_instance):
    catalog = berg_instance.get_model_catalog()
    assert isinstance(catalog, dict)
    # Keys are the human-readable modality labels from the YAML cards
    # (e.g. "fMRI", "EEG"); compare case-insensitively.
    modalities = {k.lower() for k in catalog}
    assert "fmri" in modalities
    assert "eeg" in modalities
    for datasets in catalog.values():
        assert isinstance(datasets, list)


def test_list_models_returns_sorted_ids(berg_instance):
    models = berg_instance.list_models()
    assert models == sorted(models)
    assert len(models) > 0


def test_get_model_class_roundtrips(native_model_ids):
    for model_id in native_model_ids:
        cls = get_model_class(model_id)
        assert cls.get_model_id() == model_id


def test_get_model_class_unknown_raises():
    with pytest.raises(ValueError):
        get_model_class("does-not-exist")
