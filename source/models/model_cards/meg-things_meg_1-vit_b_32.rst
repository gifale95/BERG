========================
meg-things_meg1-vit_b_32
========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - MEG
   * - Training Dataset
     - THINGS MEG1
   * - Species
     - Human
   * - Stimuli
     - Images
   * - Model Type
     - Vision transformer (ViT-B/32)
   * - Creator
     - Domenic Bersch

Description
----------

This encoding model consists of a linear mapping through linear regression of a vision transformer
(Dosovitskiy et al., 2020) image features onto whole-brain magnetoencephalography (MEG) responses from the THINGS-data MEG dataset (Hebart et al., eLife 2023). The model provides features from all 12 transformer layers, using the full
set of patch tokens per layer to represent each stimulus image. For each image stimulus, features are concatenated across all spatial tokens and reduced to 250 principal components via principal-component analysis (PCA). These reduced features serve as predictors for MEG responses.

**Neural data.** Encoding models were trained on the preprocessed data preparation provided in THINGS MEG1.
MEG data were recorded from four human participants (P1–P4) viewing 1,854 object categories
from the THINGS database (~22,000 naturalistic object images). Recordings were acquired with 271 sensors at
200 Hz, epoched from −100 ms to +1300 ms relative to stimulus onset. The preprocessing pipeline included
band-pass filtering (0.1–40 Hz), epoching, baseline correction (−100 to 0 ms), and exclusion of malfunctioning
sensors.

**Model training partition.** Single-trial responses to approximately 22,000 unique naturalistic images were
used for training.

**Model testing partition.** 200 test images, each repeated 12 times, were used for evaluation; the
target responses correspond to the average MEG activity across repetitions.

**Training procedure.** Independent linear regression models were fitted separately for each MEG sensor
and time point, predicting the sensor’s time-resolved response from the feature vectors. The resulting model weights provide a spatiotemporal mapping from visual features to MEG sensor activity.

**Noise ceiling.** The noise ceiling was computed from the 12 repeated presentations of each of the 200 test image,
  following the analytical procedure described in the Natural Scenes Dataset (NSD) paper (Allen et al., 2022).

**Output.** Each trained model predicts time-resolved MEG responses for all 271 sensors (or user-specified
subsets) across 281 time points (−100 to +1300 ms) for each input image.

Metadata
--------

**'meg'**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Shape/Type
     - Description
   * - times
     - ``(281,)``
     - Time points (e.g., -0.1 to 1.3s relative to stimulus onset)
   * - subject_id
     - ``str``
     - Subject identifier

**'sensors**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Shape/Type
     - Description
   * - sensor_names
     - ``(271,)``
     - MEG sensor name strings
   * - sensor_prefixes
     - ``(271,)``
     - Sensor prefixes (e.g., 'MLF', 'MRC', 'MZO')
   * - sensor_hemispheres
     - ``(271,)``
     - Hemisphere labels ('Left', 'Right', 'Midline')
   * - sensor_regions
     - ``(271,)``
     - Region labels ('Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital')
   * - n_sensors
     - ``int``
     - Number of MEG sensors (271)

**'encoding_model'**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Key
     - Shape/Type
     - Description
   * - train_img_ids
     - ``(22248,)``
     - THINGS image IDs for training trials
   * - train_concepts
     - ``(22248,)``
     - Object category IDs (1–1854) for training trials
   * - train_sessions
     - ``(22248,)``
     - Session numbers for training trials
   * - train_runs
     - ``(22248,)``
     - Run numbers within each training session
   * - train_img_files
     - ``(22248,)``
     - Full image paths on disk for training trials
   * - test_things_img_ids
     - ``(2400,)``
     - THINGS image IDs for test trials
   * - test_image_nr
     - ``(2400,)``
     - Test image numbers (1–200) for test trials
   * - test_concepts
     - ``(2400,)``
     - Object category IDs for test trials
   * - test_sessions
     - ``(2400,)``
     - Session numbers for test trials
   * - test_runs
     - ``(2400,)``
     - Run numbers within each test session
   * - test_img_files
     - ``(2400,)``
     - Full image paths on disk for test images
   * - ncsnr
     - ``(281, 271)``
     - Noise ceiling signal-to-noise ratio for each sensor/timepoint (computed on the test data)
   * - noise_ceiling
     - ``(281, 271)``
     - Noise ceiling for each sensor/timepoint (computed on the test data)
   * - correlation_results
     - ``(281, 271)``
     - Encoding model prediction accuracy (Pearson's r) for each sensor/timepoint (computed on the test data)


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
**Shape**: ``['batch_size', 'n_sensors', 'n_timepoints']``  
**Description**:  
The output is a 3D array containing in silico MEG responses.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch.
   * - n_sensors
     - Number of MEG sensors (up to 271, based on the number of sensors selected).
   * - n_timepoints
     - Number of time points in the MEG epoch (up to 281, based on the number of time points selected).

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
       | **Description:** Subject ID from the THINGS MEG1 dataset.
       | **Valid Values:** P1, P2, P3, P4
       | **Example:** P1
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific senors and/or timepoints. If not provided,
       | MEG responses are generated for all MEG sensors and time points.
       | 
       | **Properties:**
       | 
       | **region**
       |     **Type:** list[str]
       |     **Description:** List of anatomical sensor-region labels to include. Each region
       |     groups multiple MEG sensors located over a specific cortical area:
       |       • Central — midline motor/somatosensory sensors  
       |       • Frontal — sensors over frontal cortex  
       |       • Occipital — sensors over visual cortex  
       |       • Parietal — sensors over parietal cortex  
       |       • Temporal — sensors over temporal lobes
       |     If multiple regions are listed, their sensors are concatenated.
       |     **Valid values:** "Central", "Frontal", "Occipital", "Parietal", "Temporal"
       |     **Example:** ['Central', 'Frontal', 'Occipital']
       | 
       | **sensors**
       |     **Type:** list[str]
       |     **Description:** List of MEG sensor prefix codes to include. Each code identifies
       |     a cluster of planar gradiometers or magnetometers based on
       |     hemisphere (L = left, R = right, Z = midline) and cortical region:
       |       • MLC / MRC — Central (motor/somatosensory)
       |       • MLF / MRF — Frontal
       |       • MLO / MRO — Occipital (visual)
       |       • MLP / MRP — Parietal
       |       • MLT / MRT — Temporal
       |       • MZC / MZF / MZO / MZP — Midline (central, frontal, occipital, parietal)
       |     Sensors sharing a prefix (e.g., “MLF”) are typically grouped together
       |     for regional analyses or dimensionality reduction.
       |     **Valid values:** "MLC", "MLF", "MLO", "MLP", "MLT", "MRC", "MRF", "MRO", "MRP", "MRT", "MZC", "MZF", "MZO", "MZP"
       |     **Example:** ['MLC', 'MLF', 'MLO']
       | 
       | **sensor_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which sensors to include.
       |     Must have exactly the same length as the number of available sensors (271).
       |     Each position set to 1 indicates that sensor should be included.
       |     **Example:** [0, 0, '...', 1, 1, 0]
       | 
       | **timepoints**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which timepoints to include.
       |     Length must equal the number of time samples (281). Each 1 indicates
       |     a selected timepoint.
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

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-meg/train_dataset-things_meg_1/model-vit_b_32/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "meg-things_meg1-vit_b_32",
        subject=P1,
        selection={
            "region": ["Central", "Frontal", "Occipital"]
            "sensors": ["MLC", "MLF", "MLO"]
            "sensor_index": [0, 0, '...', 1, 1, 0]
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
    # ['batch_size', 'n_sensors', 281]
    # where:
    # - n_sensors: Number of MEG sensors (up to 271, based on the number of sensors selected).
    # - n_timepoints: Number of time points in the MEG epoch (up to 281, based on the number of time points selected).
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        images,
        return_metadata=True
    )
    

References
---------

* Model building code: https://github.com/gifale95/BERG/tree/main/berg_creation_code/02_train_encoding_models/train_dataset-things_meg_1/model-vit_b_32
* THINGS MEG & fMRI Paper (Hebart et al., 2023): https://doi.org/10.7554/eLife.82580
* THINGS MEG & fMRI Data (Hebart et al., 2023): https://plus.figshare.com/collections/_/6161151
* THINGS initiative (Hebart et al., 2019): https://things-initiative.org/
* ViT-B/32 (Dosovitskiy et al., 2020): https://arxiv.org/abs/2010.11929