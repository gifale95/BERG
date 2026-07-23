"""Unit tests for the parameter validators. Pure logic, no weights."""

import numpy as np
import pytest

from berg.core.exceptions import InvalidParameterError
from berg.core.parameter_validator import (
    ValidationError,
    validate_subject,
    validate_subjects,
    validate_selection_keys,
    validate_channels,
    validate_binary_array,
    get_selected_indices,
    validate_roi,
)


# --- validate_subject -------------------------------------------------------

def test_validate_subject_accepts_valid():
    validate_subject(2, [1, 2, 3])  # no raise


def test_validate_subject_rejects_invalid():
    with pytest.raises(InvalidParameterError):
        validate_subject(99, [1, 2, 3])


# --- validate_subjects ------------------------------------------------------

def test_validate_subjects_all_keyword():
    assert validate_subjects("all", [1, 2, 3]) == [1, 2, 3]


def test_validate_subjects_single_int_normalized_to_list():
    assert validate_subjects(2, [1, 2, 3]) == [2]


def test_validate_subjects_empty_list_raises():
    with pytest.raises(InvalidParameterError):
        validate_subjects([], [1, 2, 3])


def test_validate_subjects_none_raises():
    with pytest.raises(InvalidParameterError):
        validate_subjects(None, [1, 2, 3])


# --- validate_selection_keys ------------------------------------------------

def test_validate_selection_keys_ok():
    validate_selection_keys({"channels": []}, ["channels", "timepoints"])


def test_validate_selection_keys_bad_key():
    with pytest.raises(InvalidParameterError):
        validate_selection_keys({"nope": 1}, ["channels", "timepoints"])


# --- validate_channels ------------------------------------------------------

def test_validate_channels_ok():
    assert validate_channels(["Oz"], ["Oz", "Pz"]) == ["Oz"]


def test_validate_channels_non_list_raises():
    with pytest.raises(ValidationError):
        validate_channels("Oz", ["Oz", "Pz"])


def test_validate_channels_unknown_raises():
    with pytest.raises(ValidationError):
        validate_channels(["XX"], ["Oz", "Pz"])


# --- validate_binary_array --------------------------------------------------

def test_validate_binary_array_ok():
    arr = validate_binary_array([1, 0, 1], 3, "timepoints")
    assert np.array_equal(arr, np.array([1, 0, 1]))


def test_validate_binary_array_wrong_length():
    with pytest.raises(ValidationError):
        validate_binary_array([1, 0], 3, "timepoints")


def test_validate_binary_array_non_binary():
    with pytest.raises(ValidationError):
        validate_binary_array([2, 0, 1], 3, "timepoints")


def test_validate_binary_array_all_zero():
    with pytest.raises(ValidationError):
        validate_binary_array([0, 0, 0], 3, "timepoints")


def test_get_selected_indices():
    idx = get_selected_indices(np.array([0, 1, 0, 1]))
    assert np.array_equal(idx, np.array([1, 3]))


# --- validate_roi -----------------------------------------------------------

def test_validate_roi_string_normalized():
    assert validate_roi("V1", ["V1", "V2"]) == ["V1"]


def test_validate_roi_unknown_raises():
    with pytest.raises(ValidationError):
        validate_roi("ZZ", ["V1", "V2"])
