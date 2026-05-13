===========================
fmri-things_fmri_1-vit_b_32
===========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - THINGS fMRI1
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
(Dosovitskiy et al., 2020) image features onto whole-brain functional magnetic resonance imaging (fMRI) responses from the THINGS-fMRI dataset (Hebart et al., eLife 2023). The model provides features from all 12 transformer layers, using the full
set of patch tokens per layer to represent each stimulus image. For each image stimulus, features are concatenated across all spatial tokens and reduced to 250 principal components via principal-component analysis (PCA). These reduced features serve as predictors for fMRI responses.

**Neural data.** Encoding models were trained on the preprocessed data preparation provided in THINGS fMRI1. fMRI data were recorded from three human participants (sub-01–sub-03) viewing 1,854 object categories
from the THINGS database (~8,740 naturalistic object images). Recordings were acquired at 1.6 mm isotropic resolution,
preprocessed with standard fMRI pipelines including motion correction, slice-timing correction, and spatial normalization.

**Model training partition.** Single-trial responses to approximately 8,640 unique naturalistic images were
used for training.

**Model testing partition.** 100 test images, each repeated 12 times, were used for evaluation; the
target responses correspond to the average fMRI activity across repetitions.

**Training procedure.** The model was trained in 32 chunks (~6,604 voxels each) for memory efficiency.
Independent linear regression models were fitted for each voxel, predicting voxel responses from the PCA-reduced feature vectors. The resulting model weights provide a voxel-wise mapping from visual features to fMRI activity.

**Noise ceiling.** The noise ceiling was computed from split-half reliability of voxel responses across
the 12 repeated presentations of each test image. Two metrics are provided: (1) single-trial noise ceiling
based on individual trial reliability, and (2) test-set noise ceiling based on averaged test responses.
These represent the theoretical upper bound of prediction accuracy for each voxel.

**Output.** Each trained model predicts whole-brain fMRI responses for all 211,339 voxels (or user-specified
subsets via ROI selection) for each input image.

Metadata
--------

**fmri**

    **voxel_coords** : ``(211339, 3)`` - Voxel coordinates in volume space (x, y, z indices)

    **n_voxels** : ``int`` - Total number of voxels (211339)

    **subject_id** : ``int`` - Subject identifier (e.g., '1')
**encoding_model**

    **train_stimuli** : ``(8640,)`` - Stimulus filenames for training trials

    **train_concepts** : ``(8640,)`` - Concept labels for training trials

    **test_stimuli** : ``(1200,)`` - Stimulus filenames for test trials

    **test_concepts** : ``(1200,)`` - Concept labels for test trials

    **noise_ceiling_singletrial** : ``(211339,)`` - Max explainable variance per voxel based on single-trial repeat reliability

    **noise_ceiling_testset** : ``(211339,)`` - Max explainable variance per voxel based on averaged test-set repeats

    **splithalf_corrected** : ``(211339,)`` - Raw split-half voxel reliability without correction

    **splithalf_uncorrected** : ``(211339,)`` - Split-half reliability corrected to estimate full-data consistency

    **correlation_results** : ``(211339,)`` - Encoding model prediction accuracy (Pearson's r) for each voxel (computed on the test data)
**prf**

    **prf_eccentricity** : ``(211339,)`` - Distance of receptive field center from fixation (deg)

    **prf_polarangle** : ``(211339,)`` - Angular position of receptive field center (0–360°)

    **prf_rsquared** : ``(211339,)`` - Variance explained by pRF model (fit quality)

    **prf_size** : ``(211339,)`` - Estimated receptive field size (deg)
**roi**

    **V1, V2, V3, hV4, VO1, VO2, LO1_prf, LO2_prf, TO1, TO2, V3b, V3a, lFFA, rFFA, lOFA, rOFA, lEBA, rEBA, lPPA, rPPA, lRSC, rRSC, lTOS, rTOS, lLOC, rLOC, IT, lSTS, rSTS** : ``variable length`` - Each ROI entry contains voxel indices (variable length) for that functional region

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``['batch_size', 3, 'height', 'width']``
   * - Description
     - The input should be a batch of RGB images.
   * - Constraints
     - * Image values should be integers in range [0, 255].
       * Image dimensions (height, width) should be equal (square).
       * Minimum recommended image size: 224×224 pixels.

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``['batch_size', 'n_voxels']``
   * - Description
     - The output is a 2D array containing in silico fMRI responses.
   * - Dimensions
     - | **batch_size**: Number of stimuli in the batch.
       | **n_voxels**: Number of voxels (up to 211,339, based on ROI selection).

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model_id**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Unique identifier of the model to load.
       | **Valid Values:** fmri-things_fmri_1-vit_b_32
       | **Example:** "fmri-things_fmri_1-vit_b_32"
   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** Subject ID from the THINGS fMRI dataset.
       | **Valid Values:** 1, 2, 3
       | **Example:** 1
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific ROIs and/or voxel indices. If not provided,
       | fMRI responses are generated for all voxels.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of region-of-interest (ROI) labels to include. Each ROI
       |     represents a functionally defined brain region:
       |       • Early visual: V1, V2, V3, hV4, V3a, V3b
       |       • Ventral stream: VO1, VO2, LO1_prf, LO2_prf, TO1, TO2
       |       • High-level visual: IT (inferior temporal cortex)
       |       • Category-selective: lFFA/rFFA (faces), lOFA/rOFA (faces), 
       |         lEBA/rEBA (bodies), lPPA/rPPA (places), lRSC/rRSC (scenes),
       |         lTOS/rTOS (tools), lLOC/rLOC (objects)
       |       • Temporal: lSTS/rSTS (superior temporal sulcus)
       |     If multiple ROIs are listed, their voxels are concatenated.
       |     **Valid values:** "V1", "V2", "V3", "hV4", "VO1", "VO2", "LO1_prf", "LO2_prf", "TO1", "TO2", "V3b", "V3a", "lFFA", "rFFA", "lOFA", "rOFA", "lEBA", "rEBA", "lPPA", "rPPA", "lRSC", "rRSC", "lTOS", "rTOS", "lLOC", "rLOC", "IT", "lSTS", "rSTS"
       |     **Example:** ['V1', 'V2', 'IT']
       | 
       | **voxel_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which voxels to include.
       |     Must have exactly the same length as the number of available voxels (211,339).
       |     Each position set to 1 indicates that voxel should be included.
       |     **Example:** [0, 0, '...', 1, 1, 0]
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "auto"

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model**
     - | **Type:** BaseModelInterface
       | **Required:** Yes
       | **Description:** An instantiated and loaded encoding model.
   * - **stimulus**
     - | **Type:** numpy.ndarray
       | **Required:** Yes
       | **Description:** A batch of RGB images to be encoded. Images should be in integer format with values in the range [0, 255], and square dimensions (e.g. 224×224).
       | **Example:** "An array of shape [100, 3, 224, 224] representing 100 RGB images."
   * - **return_metadata**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to return the encoding model's metadata together with the in silico neural resposnes.
       | **Example:** True
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding (for large batches).
       | **Example:** True

Parameters used in ``get_model_metadata``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model's metadata without having to load the model itself.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model_id**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Unique identifier of the model to load.
       | **Valid Values:** fmri-things_fmri_1-vit_b_32
       | **Example:** "fmri-things_fmri_1-vit_b_32"
   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** Subject ID from the THINGS fMRI dataset.
       | **Valid Values:** 1, 2, 3
       | **Example:** 1

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-things_fmri_1/model-vit_b_32/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-things_fmri_1-vit_b_32",
        subject=1,
        selection={
            "roi": ["V1", "V2", "IT"],
            "voxel_index": [0, 0, '...', 1, 1, 0]
        }
    )
    
    # Prepare the stimulus images
    # Image shape should be [batch_size, 3 RGB channels, height, width]
    stimulus = np.random.randint(0, 255, (100, 3, 256, 256))
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # ['batch_size', 'n_voxels']
    # where:
    # - n_voxels: Number of voxels (up to 211,339, based on ROI selection).
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-things_fmri_1-vit_b_32",
        subject=1
    )
    

References
---------

* Model building code: https://github.com/gifale95/BERG/tree/main/berg_creation_code/02_train_encoding_models/train_dataset-things_fmri_1/model-vit_b_32
* THINGS MEG & fMRI Paper (Hebart et al., 2023): https://doi.org/10.7554/eLife.82580
* THINGS MEG & fMRI Data (Hebart et al., 2023): https://plus.figshare.com/collections/_/6161151
* THINGS initiative (Hebart et al., 2019): https://things-initiative.org/
* ViT-B/32 (Dosovitskiy et al., 2020): https://arxiv.org/abs/2010.11929