"""
Parameter validation utilities for BERG models.

This module provides reusable validation functions for model parameters
across different modalities.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

from berg.core.exceptions import InvalidParameterError


class ValidationError(Exception):
    """Internal exception for validation errors that will be converted to appropriate BERG exceptions."""
    pass


def validate_subject(subject: int, valid_subjects: List[int]) -> None:
    """
    Validate that a subject ID is in the list of valid subjects.
    
    Parameters
    ----------
    subject : int
        The subject ID to validate
    valid_subjects : list
        List of valid subject IDs
        
    Raises
    ------
    InvalidParameterError
        If the subject is not in the list of valid subjects
    """
    if subject not in valid_subjects:
        raise InvalidParameterError(
            f"Subject must be one of {valid_subjects}, got {subject}"
        )


def validate_selection_keys(selection: Dict[str, Any], valid_keys: List[str]) -> None:
    """
    Ensure that all keys in the selection dictionary are valid.

    Parameters
    ----------
    selection : dict
        The user-provided selection dictionary.
    valid_keys : list
        List of allowed keys (e.g. ["channels", "timepoints"]).

    Raises
    ------
    InvalidParameterError
        If any key in the selection is not valid.
    """
    for key in selection:
        if key not in valid_keys:
            raise InvalidParameterError(
                f"Invalid selection key: '{key}'. Valid keys are: {sorted(valid_keys)}"
            )



def validate_channels(
    channels: Any, 
    valid_channels: List[str]
) -> List[str]:
    """
    Validate channel selections.
    
    Parameters
    ----------
    channels : any
        The channels parameter to validate
    valid_channels : list
        List of valid channel names
        
    Returns
    -------
    list
        List of validated channel names
        
    Raises
    ------
    ValidationError
        If the channels parameter is invalid
    """
    if not isinstance(channels, list):
        raise ValidationError(
            f"channels parameter must be a list, got {type(channels)}"
        )
    
    # Verify all channels are valid
    invalid_channels = [ch for ch in channels if ch not in valid_channels]
    if invalid_channels:
        raise ValidationError(
            f"Invalid channel(s): {invalid_channels}. Valid channels are: {valid_channels}"
        )
    
    # Return the validated channels
    return channels


def validate_binary_array(
    array: Any, 
    expected_length: int,
    parameter_name: str = "binary array"
) -> np.ndarray:
    """
    Validate a binary (one-hot encoded) array.

    Parameters
    ----------
    array : any
        The array to validate
    expected_length : int
        The expected length of the array
    parameter_name : str, optional
        Name of the parameter for error messages

    Returns
    -------
    np.ndarray
        Validated binary array

    Raises
    ------
    ValidationError
        If the array is invalid
    """

    # Check if array is a list or numpy array
    if not isinstance(array, (list, np.ndarray)):
        raise ValidationError(
            f"{parameter_name} must be a list or numpy array, got {type(array)}"
        )
    
    # Convert to numpy array if it's a list
    if isinstance(array, list):
        array = np.array(array)
    
    # Check length
    if len(array) != expected_length:
        raise ValidationError(
            f"{parameter_name} must have exactly {expected_length} elements, got {len(array)}"
        )
    
    # Validate binary encoding (only 0s and 1s)
    if not np.all(np.isin(array, [0, 1])):
        raise ValidationError(
            f"{parameter_name} must be binary (contain only 0s and 1s)"
        )
    
    # Check if at least one element is selected
    if np.sum(array) == 0:
        raise ValidationError(
            f"At least one element in {parameter_name} must be selected (contain at least one 1)"
        )
    
    # Return the validated array and selected indices
    return array


def get_selected_indices(binary_array: np.ndarray) -> np.ndarray:
    """
    Get the indices of selected elements in a binary array.
    
    Parameters
    ----------
    binary_array : np.ndarray
        Binary array with 0s and 1s
        
    Returns
    -------
    np.ndarray
        Array of indices where the value is 1
    """

    return np.where(binary_array == 1)[0]


def validate_roi(roi: Any, valid_rois: List[str]) -> str:
    """
    Validate a single ROI (Region of Interest) selection.

    Parameters
    ----------
    roi : any
        The ROI value to validate
    valid_rois : list
        List of valid ROI names

    Returns
    -------
    str
        The validated ROI string

    Raises
    ------
    ValidationError
        If the ROI is not a string or not in the list of valid ROIs
    """
    if not isinstance(roi, str):
        raise ValidationError(
            f"ROI parameter must be a string, got {type(roi)}"
        )

    if roi not in valid_rois:
        raise ValidationError(
            f"Invalid ROI: '{roi}'. Valid ROIs are: {valid_rois}"
        )

    return roi
