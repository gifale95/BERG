"""End-to-end tests with real weights — the only tier that actually runs a
model and checks its output.

Table-driven via INTEGRATION_SPECS (one row per model: load kwargs + a stimulus
factory). Skipped unless BERG_DIR points at a weight download, and each case
also skips if that model's files aren't there — so a partial download is fine.

    BERG_DIR=/path/to/brain-encoding-response-generator pytest -m weights
"""

import os

import numpy as np
import pytest

import berg
from berg.core.exceptions import ModelLoadError
from berg.core.model_registry import MODEL_REGISTRY

pytestmark = pytest.mark.weights

BERG_DIR = os.environ.get("BERG_DIR")

requires_berg_dir = pytest.mark.skipif(
    not BERG_DIR or not os.path.isdir(BERG_DIR),
    reason="BERG_DIR not set to a directory with real weights",
)


def _images(n=2):
    """A batch of valid image stimuli: (n, 3, 224, 224) uint8."""
    return np.random.RandomState(0).randint(0, 256, (n, 3, 224, 224)).astype(np.uint8)


# One row per model. Fields:
#   kwargs    : args for get_encoding_model (subject, ...)
#   stimulus  : callable returning a valid stimulus batch
#   sel_axes  : {selection_key: axis_from_end} for binary one-hot selections
#   roi       : optional {axis, slicer} for ROI selection (see tests below)
INTEGRATION_SPECS = {
    # subject is an int for EEG/MEG/THINGS-fMRI; the model builds the
    # "P1" / "sub-01" filename itself. TVSD uses letter subjects.
    "eeg-things_eeg_2-vit_b_32": {
        "kwargs": {"subject": 1}, "stimulus": _images,
        "sel_axes": {"timepoints": -1},                       # (B, reps, ch, time)
    },
    "meg-things_meg_1-vit_b_32": {
        "kwargs": {"subject": 1}, "stimulus": _images,
        "sel_axes": {"sensor_index": -2, "timepoints": -1},   # (B, ch, time)
    },
    "utah_array-tvsd-vit_b_32": {
        "kwargs": {"subject": "F"}, "stimulus": _images,
        "sel_axes": {"electrodes": -2, "timepoints": -1},     # (B, elec, time)
        # ROI maps to electrodes via per-electrode label assignments.
        "roi": {
            "axis": -2,
            "slicer": lambda meta, roi: np.where(
                np.asarray(meta["roi"]["roi_assignments"])
                == list(meta["roi"]["roi_labels"]).index(roi)
            )[0],
        },
    },
    "fmri-things_fmri_1-vit_b_32": {
        "kwargs": {"subject": 1}, "stimulus": _images,
        "sel_axes": {"voxel_index": -1},                      # (B, voxels)
        # ROI maps directly to a voxel-index list in metadata.
        "roi": {
            "axis": -1,
            "slicer": lambda meta, roi: np.asarray(meta["roi"][roi]),
        },
    },
    # TODO (extend coverage): nsd_fsaverage-{vit_b_32,alexnet}, nsd-fwrf,
    # bmd-s3d (video), text models (lebel/tuckute/zada, brainscore_language).
    # Each needs its subject/selection kwargs and stimulus type filled in.
}


def _selection_axes_cases():
    """Flatten specs into (model_id, selection_key, axis_from_end) cases."""
    cases = []
    for mid, spec in INTEGRATION_SPECS.items():
        if mid not in MODEL_REGISTRY:
            continue
        for key, axis in spec.get("sel_axes", {}).items():
            cases.append((mid, key, axis))
    return cases


def _spec_ids():
    # Only parametrize over specs whose model actually registered in this env.
    return [mid for mid in INTEGRATION_SPECS if mid in MODEL_REGISTRY]


# Number of ROIs to check per model (each adds one encode). 2 catches
# label-mapping / off-by-one bugs that a single contiguous ROI might mask.
ROI_SAMPLE = 2


def _roi_values(model_id):
    """Read a model's valid ROI names from its YAML card — the same single
    source of truth the model classes read their valid values from."""
    import yaml
    with open(MODEL_REGISTRY[model_id]["yaml_path"]) as f:
        card = yaml.safe_load(f)
    return card["parameters"]["selection"]["properties"]["roi"]["valid_values"]


def _roi_spec_ids():
    return [
        mid for mid in INTEGRATION_SPECS
        if mid in MODEL_REGISTRY and "roi" in INTEGRATION_SPECS[mid]
    ]


@requires_berg_dir
@pytest.mark.parametrize("model_id", _spec_ids())
def test_model_runs_end_to_end(model_id):
    spec = INTEGRATION_SPECS[model_id]
    b = berg.BERG(berg_dir=BERG_DIR)
    try:
        model = b.get_encoding_model(model_id, **spec["kwargs"])
    except (FileNotFoundError, ModelLoadError) as exc:
        pytest.skip(f"weights for {model_id} not present in BERG_DIR ({exc})")

    stimulus = spec["stimulus"]()
    responses = b.encode(model, stimulus, return_metadata=False)

    assert responses.shape[0] == len(stimulus)
    assert np.isfinite(responses).all()


@requires_berg_dir
@pytest.mark.parametrize("model_id,sel_key,axis", _selection_axes_cases())
def test_selection_matches_sliced_full_output(model_id, sel_key, axis):
    """Same check the debug notebooks do by hand: selecting the first K elements
    of an axis must give the same numbers as slicing the full output to its
    first K. K comes from the full output, so nothing is hard-coded."""
    spec = INTEGRATION_SPECS[model_id]
    b = berg.BERG(berg_dir=BERG_DIR)
    try:
        full_model = b.get_encoding_model(model_id, **spec["kwargs"])
    except (FileNotFoundError, ModelLoadError) as exc:
        pytest.skip(f"weights for {model_id} not present in BERG_DIR ({exc})")

    stimulus = spec["stimulus"]()
    full = b.encode(full_model, stimulus, return_metadata=False)

    length = full.shape[axis]
    k = max(1, length // 2)
    onehot = np.zeros(length, dtype=int)
    onehot[:k] = 1

    subset_model = b.get_encoding_model(model_id, selection={sel_key: onehot}, **spec["kwargs"])
    subset = b.encode(subset_model, stimulus, return_metadata=False)

    # Slice the full output to its first k elements along `axis`.
    sl = [slice(None)] * full.ndim
    sl[axis] = slice(0, k)
    expected = full[tuple(sl)]

    assert subset.shape == expected.shape, (
        f"{model_id} selection '{sel_key}': shape {subset.shape} != sliced full {expected.shape}"
    )
    assert np.array_equal(subset, expected), (
        f"{model_id} selection '{sel_key}': values differ from sliced full output"
    )


@requires_berg_dir
@pytest.mark.parametrize("model_id", _roi_spec_ids())
def test_roi_selection_matches_metadata_slice(model_id):
    """Same idea for ROIs. ROI names come from the YAML card; each ROI's subset
    output must match the full output restricted to that ROI's units via the
    model's metadata mapping."""
    spec = INTEGRATION_SPECS[model_id]
    roi_cfg = spec["roi"]
    axis = roi_cfg["axis"]
    b = berg.BERG(berg_dir=BERG_DIR)
    try:
        full_model = b.get_encoding_model(model_id, **spec["kwargs"])
        meta = b.get_model_metadata(model_id, **spec["kwargs"])
    except (FileNotFoundError, ModelLoadError) as exc:
        pytest.skip(f"weights for {model_id} not present in BERG_DIR ({exc})")

    stimulus = spec["stimulus"]()
    full = b.encode(full_model, stimulus, return_metadata=False)

    rois = _roi_values(model_id)[:ROI_SAMPLE]
    assert rois, f"no ROI valid_values in card for {model_id}"

    for roi in rois:
        idx = np.asarray(roi_cfg["slicer"](meta, roi))
        if idx.size == 0:
            continue  # ROI not present for this subject
        subset_model = b.get_encoding_model(model_id, selection={"roi": [roi]}, **spec["kwargs"])
        subset = b.encode(subset_model, stimulus, return_metadata=False)
        manual = np.take(full, idx, axis=axis)

        assert subset.shape == manual.shape, (
            f"{model_id} roi '{roi}': shape {subset.shape} != metadata slice {manual.shape}"
        )
        assert np.array_equal(subset, manual), (
            f"{model_id} roi '{roi}': values differ from metadata slice"
        )


@requires_berg_dir
@pytest.mark.parametrize("model_id", _spec_ids())
def test_metadata_roundtrip(model_id):
    spec = INTEGRATION_SPECS[model_id]
    b = berg.BERG(berg_dir=BERG_DIR)
    try:
        meta = b.get_model_metadata(model_id, **spec["kwargs"])
    except (FileNotFoundError, ModelLoadError) as exc:
        pytest.skip(f"metadata for {model_id} not present in BERG_DIR ({exc})")
    assert isinstance(meta, dict)


def test_no_registered_model_lacks_a_spec():
    """Lists registered models that don't have an integration spec yet, so
    coverage gaps stay visible. BrainScore gateways are exempt."""
    exempt = {"brainscore_vision", "brainscore_language"}
    uncovered = sorted(
        mid for mid in MODEL_REGISTRY
        if mid not in INTEGRATION_SPECS and mid not in exempt
    )
    if uncovered:
        pytest.skip(
            "Models registered but not covered by an end-to-end integration "
            f"spec ({len(uncovered)}): {uncovered}"
        )
