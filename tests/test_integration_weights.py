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


# Skip a case only when the weights are genuinely missing. A ModelLoadError for
# any other reason (bad format, wrong key, ...) is a real bug and must fail, not
# silently skip.
_MISSING_HINTS = ("no such file", "not found", "cannot find", "does not exist")


def _load_or_skip(fn, model_id):
    try:
        return fn()
    except FileNotFoundError as exc:
        pytest.skip(f"weights for {model_id} not present in BERG_DIR ({exc})")
    except ModelLoadError as exc:
        if any(h in str(exc).lower() for h in _MISSING_HINTS):
            pytest.skip(f"weights for {model_id} not present in BERG_DIR ({exc})")
        raise


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
        "rep_axis": 1,   # 4 EEG repetitions must not be identical (per-rep PCAs)
    },
    # The two alexnet EEG models share the per-rep-PCA pipeline that hit the
    # stale-weights bug, so they get the same per-rep distinctness guard.
    "eeg-things_eeg_2-alexnet": {
        "kwargs": {"subject": 1}, "stimulus": _images,
        "sel_axes": {"timepoints": -1},
        "rep_axis": 1,
    },
    "eeg-things_eeg_2-alexnet_untrained": {
        "kwargs": {"subject": 1}, "stimulus": _images,
        "sel_axes": {"timepoints": -1},
        "rep_axis": 1,
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
    # Minimal coverage: load + encode + invariants + metadata. Its lh/rh-vertex
    # selection layout is more involved, so selection/ROI slice-equality is left
    # out here (no sel_axes/roi) rather than encoded incorrectly.
    "fmri-nsd_fsaverage-vit_b_32": {
        "kwargs": {"subject": 1}, "stimulus": _images,
    },
    "fmri-nsd_fsaverage-alexnet": {
        "kwargs": {"subject": 1}, "stimulus": _images,
    },
    "fmri-nsd_fsaverage-alexnet_untrained": {
        "kwargs": {"subject": 1}, "stimulus": _images,
    },
    # TODO (extend coverage): nsd_fsaverage selection (lh/rh vertices, roi),
    # nsd-fwrf, bmd-s3d (video), text models (lebel/tuckute/zada,
    # brainscore_language). Each needs its subject/selection kwargs and stimulus
    # type filled in.
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
    model = _load_or_skip(lambda: b.get_encoding_model(model_id, **spec["kwargs"]), model_id)

    stimulus = spec["stimulus"]()
    responses = b.encode(model, stimulus, return_metadata=False)

    # Some models (e.g. nsd_fsaverage) return a tuple of arrays — one per
    # hemisphere. Normalise to a list so the invariants apply uniformly.
    arrays = list(responses) if isinstance(responses, tuple) else [responses]

    for i, arr in enumerate(arrays):
        tag = f"{model_id}[{i}]" if len(arrays) > 1 else model_id
        # Shape, dtype, finiteness.
        assert arr.shape[0] == len(stimulus), f"{tag}: batch dim != n stimuli"
        assert np.issubdtype(arr.dtype, np.floating), (
            f"{tag}: expected floating-point responses, got {arr.dtype}"
        )
        assert np.isfinite(arr).all(), f"{tag}: responses contain NaN/Inf"

    # Determinism: a stateless encoder must return identical output for the same
    # input. A model that accidentally introduces nondeterminism (dropout left
    # on, uninitialised buffers, ...) fails here.
    again = b.encode(model, stimulus, return_metadata=False)
    again_arrays = list(again) if isinstance(again, tuple) else [again]
    for arr, arr2 in zip(arrays, again_arrays):
        assert np.array_equal(arr, arr2), f"{model_id}: encode is nondeterministic"

    # Multi-repetition models must not collapse to identical repeats. This is the
    # direct guard for the stale/flattened-PCA class of bug: a single shared PCA
    # (or copied weights) would make every repetition equal.
    rep_axis = spec.get("rep_axis")
    if rep_axis is not None:
        assert len(arrays) == 1, f"{model_id}: rep_axis set on a multi-array output"
        resp = arrays[0]
        n_reps = resp.shape[rep_axis]
        assert n_reps > 1, f"{model_id}: rep_axis {rep_axis} has only {n_reps} rep"
        first = np.take(resp, 0, axis=rep_axis)
        assert any(
            not np.array_equal(first, np.take(resp, r, axis=rep_axis))
            for r in range(1, n_reps)
        ), f"{model_id}: all {n_reps} repetitions are identical (PCA/weights bug?)"


@requires_berg_dir
@pytest.mark.parametrize("model_id,sel_key,axis", _selection_axes_cases())
def test_selection_matches_sliced_full_output(model_id, sel_key, axis):
    """Same check the debug notebooks do by hand: selecting the first K elements
    of an axis must give the same numbers as slicing the full output to its
    first K. K comes from the full output, so nothing is hard-coded."""
    spec = INTEGRATION_SPECS[model_id]
    b = berg.BERG(berg_dir=BERG_DIR)
    full_model = _load_or_skip(lambda: b.get_encoding_model(model_id, **spec["kwargs"]), model_id)

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
    full_model = _load_or_skip(lambda: b.get_encoding_model(model_id, **spec["kwargs"]), model_id)
    meta = _load_or_skip(lambda: b.get_model_metadata(model_id, **spec["kwargs"]), model_id)

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
    meta = _load_or_skip(lambda: b.get_model_metadata(model_id, **spec["kwargs"]), model_id)
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
