"""Check the YAML model cards. The code reads keys out of these at import time
and `describe` renders them, so a malformed card should fail here. Runs over
every registered native model, so new ones are covered automatically."""

import os

import pytest
import yaml

from berg.core.model_registry import MODEL_REGISTRY

# Keys present in all current cards; treated as the required schema.
REQUIRED_TOP_LEVEL_KEYS = {
    "model_id",
    "modality",
    "training_dataset",
    "species",
    "stimuli",
    "model_type",
    "creator",
    "description",
    "input",
    "output",
    "parameters",
    "references",
}


def _native_cards():
    return [
        (mid, info["yaml_path"])
        for mid, info in MODEL_REGISTRY.items()
        if info.get("yaml_path")
    ]


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_exists_and_parses(model_id, yaml_path):
    assert os.path.exists(yaml_path), f"missing card for {model_id}: {yaml_path}"
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    assert isinstance(card, dict)


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_has_required_keys(model_id, yaml_path):
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    missing = REQUIRED_TOP_LEVEL_KEYS - card.keys()
    assert not missing, f"{model_id} card missing keys: {sorted(missing)}"


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_model_id_matches_registry(model_id, yaml_path):
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    assert card["model_id"] == model_id, (
        f"card model_id '{card['model_id']}' != registry key '{model_id}'"
    )


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_parameters_is_mapping(model_id, yaml_path):
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    assert isinstance(card["parameters"], dict) and card["parameters"]


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_subject_valid_values_well_formed(model_id, yaml_path):
    """The model classes read parameters.subject.valid_values to validate the
    subject; a card missing/malforming it would only blow up at runtime."""
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    params = card["parameters"]
    if "subject" not in params:
        pytest.skip(f"{model_id} has no subject parameter")
    subject = params["subject"]
    assert isinstance(subject, dict), f"{model_id}: parameters.subject must be a mapping"
    valid_values = subject.get("valid_values")
    assert isinstance(valid_values, list) and valid_values, (
        f"{model_id}: parameters.subject.valid_values must be a non-empty list"
    )


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_selection_properties_well_formed(model_id, yaml_path):
    """When a card declares a selection, the classes read selection.properties to
    derive the allowed selection keys, so it must be a mapping."""
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    selection = card["parameters"].get("selection")
    if selection is None:
        pytest.skip(f"{model_id} has no selection parameter")
    assert isinstance(selection, dict), f"{model_id}: parameters.selection must be a mapping"
    assert isinstance(selection.get("properties"), dict), (
        f"{model_id}: parameters.selection.properties must be a mapping"
    )


@pytest.mark.parametrize("model_id,yaml_path", _native_cards())
def test_card_roi_valid_values_well_formed(model_id, yaml_path):
    """When a card's selection declares an 'roi' property, its valid_values is the
    single source of truth: the model classes load it into VALID_ROIS to validate
    ROI selections, and the weights-tier ROI test reads the same list. An empty or
    malformed list would silently break ROI validation and describe(), without any
    weight-free test noticing — so pin it here."""
    with open(yaml_path) as f:
        card = yaml.safe_load(f)
    selection = card["parameters"].get("selection")
    if selection is None:
        pytest.skip(f"{model_id} has no selection parameter")
    properties = selection.get("properties") or {}
    if "roi" not in properties:
        pytest.skip(f"{model_id} has no roi selection property")
    valid_values = properties["roi"].get("valid_values")
    assert isinstance(valid_values, list) and valid_values, (
        f"{model_id}: parameters.selection.properties.roi.valid_values must be a "
        "non-empty list"
    )
    assert all(isinstance(v, (str, int)) for v in valid_values), (
        f"{model_id}: roi.valid_values must contain only strings/ints"
    )
