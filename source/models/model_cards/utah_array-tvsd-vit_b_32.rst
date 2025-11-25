========================
utah_array-tvsd-vit_b_32
========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - utah_array
   * - Training Dataset
     - tvsd
   * - Species
     - Macaque
   * - Stimuli
     - Images
   * - Model Type
     - Vision transformer (ViT-B/32)
   * - Creator
     - Domenic Bersch

Description
----------

This encoding model consists of a linear mapping through linear regression of a vision transformer
(Dosovitskiy et al., 2020) image features onto intracortical spiking activity. The ViT-B/32 model extracts
features from all 12 transformer layers, using all 50 patch tokens per layer. Prior to
mapping onto neural responses, the image features have been downsampled to 250 principal components using
principal component analysis. The encoding models were trained on the THINGS Ventral Stream Spiking Dataset
(TVSD) (Papale et al., Neuron 2025), simultaneous intracortical recordings from 1,024 electrodes across
macaque ventral stream areas (V1, V4, IT) in response to natural images from the THINGS database
(Hebart et al., 2019).

**Neural data**. Raw broadband signals (30 kHz) were band-pass filtered to extract high-frequency spiking activity, 
and multi-unit activity (MUA) was obtained using threshold-based spike detection and smoothing, following the official TVSD pipeline. Responses were baseline-corrected and normalized per session, with area-specific time windows aligned to peak latencies (V1: 25–125 ms, V4: 50–150 ms, IT: 75–175 ms).
The data were epoched from -100 ms to +199 ms relative to stimulus onset, resulting in 300 time points.
More detailed preprocessing steps are described in the TVSD paper.

**Model training partition.** Spiking responses to 22,248 unique images from the THINGS database, each
presented once during passive fixation.

**Model testing partition.** Spiking responses to 100 unique images, each repeated 30 times. The test
predictions are generated for the averaged responses across repetitions.

**Training procedure.** Independent encoding models were trained for each monkey (monkeyN and monkeyF). For computational efficiency, models were trained in 8 electrode chunks (128 electrodes each), with predictions concatenated to generate full responses across all 1,024 electrodes and 300 time points simultaneously.

**Noise ceiling.** The noise ceiling was computed from the 30 repeated presentations of each test image 
to quantify the maximum explainable variance given measurement noise. We calculated the signal-to-noise 
ratio (ncsnr) by separating within-image variance (noise) from between-image variance (signal), then 
converted to r² units: noise_ceiling = 100 × (ncsnr² / (ncsnr² + 1/30)). This represents the theoretical 
upper bound of prediction accuracy for each electrode and time point.

**Output.** Each trained model predicts time-resolved spike responses for all 1024 electrodes (or user-specified
subsets) across 300 time points for each input image.

Metadata
--------

**'utah-array'**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Shape/Type
     - Description
   * - train_img_ids
     - ``(22248,)``
     - Training stimulus IDs
   * - train_img_files
     - ``(22248,)``
     - Training image filenames
   * - train_img_concepts
     - ``(22248,)``
     - Training object categories
   * - train_days
     - ``(22248,)``
     - Recording days for training
   * - train_sequence_pos
     - ``(22248,)``
     - Position in 4-image sequence
   * - test_img_ids
     - ``(3000,)``
     - Test stimulus IDs (individual trials)
   * - test_img_files
     - ``(3000,)``
     - Test image filenames (individual)
   * - test_img_concepts
     - ``(3000,)``
     - Test object categories (individual)
   * - test_days
     - ``(3000,)``
     - Recording days for test
   * - test_sequence_pos
     - ``(3000,)``
     - Position in sequence for test
   * - test_avg_img_ids
     - ``(100,)``
     - Unique test stimulus IDs
   * - test_avg_img_files
     - ``(100,)``
     - Test image filenames (averaged)
   * - test_avg_img_concepts
     - ``(100,)``
     - Test object categories (averaged)
   * - times
     - ``(300,)``
     - Time points (-100 to 199ms)
   * - electrode_order
     - ``(1024,)``
     - Electrode mapping order (0-based)
   * - roi_assignments
     - ``(1024,)``
     - ROI assignment per electrode (0=V1, 1=V4, 2=IT)
   * - roi_labels
     - ``(3,)``
     - ROI label names ['V1', 'V4', 'IT']
   * - monkey_id
     - ``str``
     - Monkey identifier
   * - n_electrodes
     - ``int``
     - Number of electrodes (1024)

**'encoding_model'**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Shape/Type
     - Description
   * - SNR
     - ``(4, 1024)``
     - Signal-to-noise ratio per day per electrode
   * - SNR_max
     - ``(1024,)``
     - Best SNR across all days per electrode
   * - oracle
     - ``(1024,)``
     - Noise ceiling estimate per electrode from paper
   * - ncsnr
     - ``(300, 1024)``
     - Calculated neural signal-to-noise ratio
   * - noise_ceiling
     - ``(300, 1024)``
     - Calculated noise Ceiling per timepoint for all electrodes


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
**Shape**: ``['batch_size', 'n_electrodes', 300]``  
**Description**:  
The output is a 3D array containing in silico utah-array responses.
The second dimension (n_electrodes) corresponds to the number of electrodes in the selected ROI,
which varies by ROI and monkey.
The third dimension corresponds to the timepoints (300).

Monkey N electrode count:
* V1: 512
* V4: 256
* IT: 256

Monkey N electrode count:
* V1: 512
* V4: 192
* IT: 320

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch
   * - n_electrodes
     - Number of electrodes in the selection
   * - timepoints
     - Timepoints of recording

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **subject**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Monkey ID
       | **Valid Values:** N, F
       | **Example:** N
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific electrodes and/or timepoints. If not provided,
       | utah-array responses are generated for all electrodes and time points.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of ROIs to include in the output
       |     **Valid values:** "V1", "V4", "IT"
       |     **Example:** ['V1', 'IT']
       | 
       | **electrodes**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which electrodes to include.
       |     Must have exactly the same length as the number of available electrode (1024).
       |     Each position set to 1 indicates that timepoint should be included.
       |     **Example:** [0, 0, '...', 1, 1, 0]
       | 
       | **timepoints**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which timepoints to include.
       |     Must have exactly the same length as the number of available timepoints (140).
       |     Each position set to 1 indicates that timepoint should be included.
       |     **Example:** [0, 0, '...', 1, 1, 0]

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

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
       | **Description:** Whether to show a progress bar during encoding (for large batches).
       | **Example:** True

Performance
----------

**Accuracy Plots:**

* ``brain-encoding-response-generator/encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "utah_array-tvsd-vit_b_32",
        subject=N,
        selection={
            "roi": ["V1", "IT"]
            "electrodes": [0, 0, '...', 1, 1, 0]
            "timepoints": [0, 0, '...', 1, 1, 0]
        },
        device="auto"
    )
    
    # Prepare the stimulus images
    # Image shape should be [batch_size, 3 RGB channels, height, width]
    images = np.random.randint(0, 255, (100, 3, 256, 256))
    
    # Generates the in silico neural responses to images using the encoding model previously loaded
    responses = berg.encode(
        model,
        images,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # ['batch_size', 'n_electrodes', 300]
    # where:
    # - n_electrodes: Number of electrodes in the selection
    # - timepoints: Timepoints of recording
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        images,
        return_metadata=True
    )
    

References
---------

* Model building code: https://github.com/gifale95/BERG/tree/main/berg_creation_code
* TVSD Paper (Papale et al., 2025): https://www.sciencedirect.com/science/article/pii/S089662732400881X
* TVSD Data (Papale et al., 2025): https://gin.g-node.org/paolo_papale/TVSD
* ViT-B/32 (Dosovitskiy et al., 2020): https://arxiv.org/abs/2010.11929