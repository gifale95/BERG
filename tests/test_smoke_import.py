"""Smoke tests: the package imports and models register. Since models register
on import, most breakage shows up here without needing any weights."""

import glob
import os

import yaml

import berg
from berg.core.model_registry import MODEL_REGISTRY

_CARDS_DIR = os.path.join(os.path.dirname(berg.__file__), "models", "model_cards")

# Cards that intentionally do NOT register in a base/test install because they
# need optional dependencies (safe_import skips them with a warning). Keep the
# reason next to each. A model card that is absent from the registry but NOT
# listed here makes test_all_nonoptional_cards_register fail on purpose — that
# is the signal that a newly added model silently failed to import (a real bug,
# or a dependency wrongly assumed to be present). Either fix the import or, if
# the dep really is optional, add the model here with its reason.
OPTIONAL_MODELS = {
    "calcium_2p-wang_2025-3DCNN": "fnn",
    "fmri-cneuromod_algo2025-text2fmri": "nilearn",
    "fmri-cneuromod_algo2025-vibe": "nilearn",
    "fmri-dascoli_2026-tribe_v2": "tribe optional deps",
    "fmri-mosaic-CNN8_multihead_subAll_verticesVisual": "mosaic-dataset",
    "fmri-mosaic-CNN8_multihead_subNSD_verticesAll": "mosaic-dataset",
    "fmri-nsd_fsaverage-huze": "yacs / huze optional deps",
    "fmri-tuckute_2024-GPT2_XL": "tuckute optional deps",
}

# The example/template card ships a placeholder model_id and has no real model.
TEMPLATE_MODEL_IDS = {"modality-dataset-model_type"}


def _card_model_ids():
    """model_id of every YAML card under model_cards/."""
    ids = {}
    for path in glob.glob(os.path.join(_CARDS_DIR, "*.yaml")):
        card = yaml.safe_load(open(path))
        ids[card["model_id"]] = os.path.basename(path)
    return ids


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


def test_all_nonoptional_cards_register():
    """Every model card that isn't a template or a declared optional-dep model
    must actually be in the registry. Catches the silent-breakage path where a
    newly added model fails to import (safe_import swallows non-berg ImportErrors
    as 'optional') and nothing else notices."""
    cards = _card_model_ids()
    expected = set(cards) - TEMPLATE_MODEL_IDS - set(OPTIONAL_MODELS)
    missing = sorted(mid for mid in expected if mid not in MODEL_REGISTRY)
    assert not missing, (
        "model cards present but not registered (broken import, or a dependency "
        "wrongly treated as optional). Fix the import, or add to OPTIONAL_MODELS "
        f"with a reason: {missing}"
    )


def test_optional_models_list_is_accurate():
    """Guard the guard: every model_id in OPTIONAL_MODELS / TEMPLATE_MODEL_IDS
    must correspond to a real card, so the lists don't rot as models are renamed
    or removed."""
    cards = set(_card_model_ids())
    stale = sorted((set(OPTIONAL_MODELS) | TEMPLATE_MODEL_IDS) - cards)
    assert not stale, f"OPTIONAL_MODELS/TEMPLATE list references nonexistent cards: {stale}"
