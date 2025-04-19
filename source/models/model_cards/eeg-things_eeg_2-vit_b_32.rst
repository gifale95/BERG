=========================
eeg-things_eeg_2-vit_b_32
=========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - eeg
   * - Training Dataset
     - things_eeg_2
   * - Model Architecture
     - vision transformer (ViT-B/32)
   * - Creator
     - Alessandro Gifford

Description
----------

This model generates in silico EEG responses to visual stimuli using a vision transformer model.
It was trained on the THINGS-EEG-2 dataset, which contains EEG recordings from subjects viewing
images of everyday objects. The model extracts visual features using a pre-trained ViT-B/32
transformer, applies dimensionality reduction, and then predicts EEG responses across all channels
and time points.

The model takes as input a batch of RGB images in the shape [batch_size, 3, height, width], with pixel values ranging from 0 to 255 and square dimensions (e.g., 224×224).

Input
-----

**Type**: ``numpy.ndarray``  
**Shape**: ``['batch_size', 3, 'height', 'width']``  
**Description**: The input should be a batch of RGB images.

**Constraints:**

* Image values should be integers in range [0, 255]
* Image dimensions (height, width) should be equal (square)
* Minimum recommended image size: 224x224 pixels

Output
------

**Type**: ``numpy.ndarray``  
**Shape**: ``['batch_size', 'n_repetitions', 'n_channels', 'n_timepoints']``  
**Description**:  
The output is a 4D array containing predicted EEG responses.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch
   * - n_repetitions
     - Number of simulated repetitions of the same stimulus (typically 4)
   * - n_channels
     - Number of EEG channels (typically 63)
   * - n_timepoints
     - Number of time points in the EEG epoch (typically 140)

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** Subject ID from the THINGS-EEG-2 dataset (1-10)
       | **Valid Values:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
       | **Example:** 1
   * - **nest_dir**
     - | **Type:** str
       | **Required:** No
       | **Description:** Root directory of the NEST repository (optional if default paths are set)
       | **Example:** ./
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific channels and/or timepoints.
       | 
       | **Properties:**
       | 
       | **channels**
       |     **Type:** list[str]
       |     **Description:** List of EEG channel names to include in the output
       |     **Valid values:** *63 options available* - e.g., "Fp1", "F3", ...
       |     **Example:** ['Oz', 'Cz', 'Fp1']
       | 
       | **timepoints**
       |     **Type:** binary_vector
       |     **Description:** Binary one-hot encoded vector indicating which timepoints to include.
       |     Must have exactly the same length as the number of available timepoints (140).
       |     Each position set to 1 indicates that timepoint should be included.
       |     **Example:** [0, 0, 0, 1, 1, 1, 0, 0]

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **stimulus**
     - | **Type:** numpy.ndarray
       | **Required:** Yes
       | **Description:** A batch of RGB images to be encoded. Images should be in integer format with values in the range [0, 255], and square dimensions (e.g. 224x224).
       | **Example:** An array of shape [100, 3, 224, 224] representing 100 RGB images.
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
       | **Valid Values:** cpu, cuda, auto
       | **Example:** auto
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding (for large batches)
       | **Example:** True

Performance
----------

**Accuracy Plots:**

* ``neural_encoding_simulation_toolkit/encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from nest import NEST
    
    # Initialize NEST
    nest = NEST(nest_dir="path/to/nest")
    
    # Load the model
    model = nest.get_encoding_model("eeg-things_eeg_2-vit_b_32", subject=1, selection={"channels": ['Oz', 'Cz', 'Fp1'], "timepoints": [0, 0, 0, ...]})
    
    # Prepare your stimuli
    # stimulus shape should be ['batch_size', 3, 'height', 'width']
    
    # Generate responses
    responses = nest.encode(model, stimulus, device="auto", show_progress=True)
    
    # responses shape will be ['batch_size', 'n_repetitions', 'n_channels', 'n_timepoints']
    # where:
    # - n_repetitions is Number of simulated repetitions of the same stimulus (typically 4)
    # - n_channels is Number of EEG channels (typically 63)
    # - n_timepoints is Number of time points in the EEG epoch (typically 140)
    
    # Get responses with metadata
    responses, metadata = nest.encode(model, stimulus, return_metadata=True)
    
    # Access channel names and time information
    channel_names = metadata['eeg']['ch_names']
    time_points = metadata['eeg']['times']  # in seconds

References
---------

* https://doi.org/10.1038/s41593-021-00962-x