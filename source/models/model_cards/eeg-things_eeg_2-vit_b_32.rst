=========================
eeg-things_eeg_2-vit_b_32
=========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - EEG
   * - Training Dataset
     - THINGS EEG2
   * - Species
     - Human
   * - Stimuli
     - Images
   * - Model Type
     - vision transformer (ViT-B/32)
   * - Creator
     - Alessandro Gifford

Description
----------

These encoding models consist in a linear mapping (through linear regression) of vision transformer
(Dosovitskiy et al., 2020) image features onto EEG responses. Prior to mapping onto EEG responses, the
image features have been downsampled to 250 principal components using principal component analysis.

The encoding models were trained on THINGS EEG2 (Gifford et al., 2022), 63-channel EEG responses of 10 subjects to
over 16,740 images from the THINGS initiative (Hebart et al., 2019).

**Preprocessing**. During preprocessing the 63-channel raw EEG data was filtered between 0.03 Hz and 100 Hz; epoched
from -100 ms to +600 ms with respect to stimulus onset; transformed using current source density transform;
downsampled to 200 Hz resulting in 140 times points per epoch (one every 5 ms); baseline corrected at each channel
using the mean of the pre-stimulus interval.

**Model training partition.** EEG responses for 16,540 unique images, each repeated 4 times (i.e., the official
training partition of the THINGS EEG2 dataset).

**Model testing partition.** EEG responses for 200 unique images, each repeated 80 times (i.e., the official testing
partition of the THINGS EEG2 dataset).

Independent encoding models were trained for each of the 4 training data repeats, and as a result the trained encoding models 
generate 4 instances (i.e., repeats) of in silico EEG responses.  Indepedent encoding models were trained for each subject,
channel, and time point.

Input
-----

**Type**: ``numpy.ndarray``  
**Shape**: ``['batch_size', 3, 'height', 'width']``  
**Description**: The input should be a batch of RGB images.

**Constraints:**

* Image values should be integers in range [0, 255].
* Image dimensions (height, width) should be equal (square).
* Minimum recommended image size: 224×224 pixels.

Output
------

**Type**: ``numpy.ndarray``  
**Shape**: ``['batch_size', 'n_repetitions', 'n_channels', 'n_timepoints']``  
**Description**:  
The output is a 4D array containing in silico EEG responses.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch.
   * - n_repetitions
     - Number of simulated repetitions of the same stimulus (always 4).
   * - n_channels
     - Number of EEG channels (up to 63, based on the number of channels selected).
   * - n_timepoints
     - Number of time points in the EEG epoch (up to 140, based on the number of time points selected).

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** Subject ID from the THINGS EEG2 dataset (1-10).
       | **Valid Values:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
       | **Example:** 1
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific channels and/or timepoints. If not provided, EEG responses
       | are generated for all EEG channels and time points.
       | 
       | **Properties:**
       | 
       | **channels**
       |     **Type:** list[str]
       |     **Description:** List of EEG channel names to include in the output
       |     **Valid values:** "Fp1", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "FCz", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2", "AF4", "AF8"
       |     **Example:** ["Oz", "Cz", "Fp1"]
       | 
       | **timepoints**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which timepoints to include.
       |     Must have exactly the same length as the number of available timepoints (140).
       |     Each position set to 1 indicates that timepoint should be included.
       |     **Example:** [0, 0, ..., 1, 1, 0]

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **stimulus**
     - | **Type:** numpy.ndarray
       | **Required:** Yes
       | **Description:** A batch of RGB images to be encoded. Images should be in integer format with values in the range [0, 255], and square dimensions (e.g. 224×224).
       | **Example:** An array of shape [100, 3, 224, 224] representing 100 RGB images.
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "auto"
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding (for large batches).
       | **Example:** True

Performance
----------

**Accuracy Plots:**

* ``neural-encoding-simulation-toolkit/encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from nest import NEST
    
    # Initialize NEST
    nest = NEST(nest_dir="path/to/neural-encoding-simulation-toolkit")
    
    # Load the encoding model
    model = nest.get_encoding_model("eeg-things_eeg_2-vit_b_32", subject=1, selection={"channels": ['Oz', 'Cz', 'Fp1'], "timepoints": [0, 1, ..., 1]})
    
    # Prepare the stimulus images
    # Image shape should be [batch_size, 3 RGB channels, height, width]
    images = np.random.randint(0, 255, (100, 3, 256, 256))
    
    # Generates the in silico neural responses to images using the encoding model previously loaded
    responses = nest.encode(model, images, device="auto", show_progress=True)
    
    # responses shape will be [batch_size, n_repetitions, n_channels, n_timepoints]
    # where:
    # - n_repetitions is Number of simulated repetitions of the same stimulus (always 4).
    # - n_channels is Number of EEG channels (up to 63, based on the number of channels selected).
    # - n_timepoints is Number of time points in the EEG epoch (up to 140, based on the number of time points selected).
    
    # Generate in silico neural responses with metadata
    responses, metadata = nest.encode(model, images, return_metadata=True)
    
    # Access EEG channel names and time information
    channel_names = metadata["eeg"]["ch_names"]
    time_points = metadata["eeg"]["times"]  # in seconds

References
---------

* {'Model building code': 'https://github.com/gifale95/NEST/tree/main/nest_creation_code'}
* {'THINGS EEG2 (Gifford et al., 2022)': 'https://doi.org/10.1016/j.neuroimage.2022.119754'}
* {'THINGS initiative (Hebart et al., 2019)': 'https://things-initiative.org/'}
* {'ViT-B/32 (Dosovitskiy et al., 2020)': 'https://arxiv.org/abs/2010.11929'}
