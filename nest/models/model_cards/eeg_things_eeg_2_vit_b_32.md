# eeg_things_eeg_2_vit_b_32

## Model Summary

| Key | Value |
|-----|-------|
| `modality` | `eeg` |
| `dataset` | `things_eeg_2` |
| `features` | `vision transformer (ViT-B/32)` |
| `repeats` | `multi` |
| `subject_level` | `True` |

## Description

This model generates in silico EEG responses to visual stimuli using a vision transformer model.
It was trained on the THINGS-EEG-2 dataset, which contains EEG recordings from subjects viewing
images of everyday objects. The model extracts visual features using a pre-trained ViT-B/32
transformer, applies dimensionality reduction, and then predicts EEG responses across all channels
and time points.

The model takes as input a batch of RGB images in the shape [batch_size, 3, height, width], with pixel values ranging from 0 to 255 and square dimensions (e.g., 224×224).

## Input

**Type**: `numpy.ndarray`  
**Shape**: `['batch_size', 3, 'height', 'width']`  
**Description**: The input should be a batch of RGB images.  
**Constraints:**
- Image values should be integers in range [0, 255]
- Image dimensions (height, width) should be equal (square)
- Minimum recommended image size: 224x224 pixels

## Output

**Type**: `numpy.ndarray`  
**Shape**: `['batch_size', 'n_repetitions', 'n_channels', 'n_timepoints']`  
**Description**:  
The output is a 4D array containing predicted EEG responses.  

**Dimensions:**

| Name | Description |
|------|-------------|
| `batch_size` | Number of stimuli in the batch |
| `n_repetitions` | Number of simulated repetitions of the same stimulus (typically 4) |
| `n_channels` | Number of EEG channels (typically 64) |
| `n_timepoints` | Number of time points in the EEG epoch (typically 140) |

## Parameters

### Parameters used in `get_encoding_model`

| Name | Type | Required | Description | Example | Valid Values |
|------|------|----------|-------------|---------|---------------|
| `subject` | `int` | `True` | Subject ID from the THINGS-EEG-2 dataset (1-4) | `1` | 1, 2, 3, 4 |
| `nest_dir` | `str` | `False` | Root directory of the NEST repository (optional if default paths are set) | `./` | - |

### Parameters used in `encode`

| Name | Type | Required | Description | Example | Valid Values |
|------|------|----------|-------------|---------|---------------|
| `stimulus` | `numpy.ndarray` | `True` | A batch of RGB images to be encoded. Images should be in integer format with values in the range [0, 255], and square dimensions (e.g. 224x224). | `An array of shape [100, 3, 224, 224] representing 100 RGB images.` | - |
| `device` | `str` | `False` | Device to run the model on. 'auto' will use CUDA if available, otherwise CPU. | `auto` | cpu, cuda, auto |
| `show_progress` | `bool` | `False` | Whether to show a progress bar during encoding (for large batches) | `True` | - |

## Performance

**Accuracy Plots:**
- `neural_encoding_simulation_toolkit/encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32/encoding_models_accuracy`

## References

- x
