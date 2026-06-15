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
