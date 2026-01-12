================================================
fmri-mosaic-CNN8_multihead_subAll_verticesVisual
================================================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - MOSAIC (8 datasets, visual cortical vertices)
   * - Species
     - Human
   * - Stimuli
     - Images
   * - Model Type
     - CNN8 (8-layer convolutional network)
   * - Creator
     - MOSAIC Team (Lahner et al., 2025)

Description
----------

This encoding model consists of a brain-optimized convolutional neural network (CNN8) trained to predict
visual cortex fMRI responses (GlasserGroups 1-22; 7,831 cortical vertices) across multiple large-scale datasets.
The model uses a shared 8-layer convolutional core with subject-specific linear factorized readout heads,
jointly trained across 93 subjects from 8 datasets.

**Neural data.** The model was trained on the MOSAIC (Meta-Organized Stimuli And fMRI Imaging data for
Computational modeling) aggregated dataset, which consists of 430,007 single-trial fMRI–stimulus pairs across
162,839 unique naturalistic and artificial stimuli from 93 subjects. All data underwent a shared preprocessing
pipeline (fMRIPrep and GLMsingle) to ensure consistency across datasets. The eight constituent datasets are:
BOLD5000 (4 subjects), Deeprecon (3 subjects), Generic Object Decoding / GOD (5 subjects),
Natural Scenes Dataset / NSD (8 subjects), THINGS-fMRI (3 subjects), BOLD Moments Dataset / BMD (10 subjects),
Natural Object Dataset / NOD (30 subjects), and Human Actions Dataset / HAD (30 subjects).

**Model architecture.** The CNN8 core consists of eight 2D convolutional blocks (each with 2D convolution,
batch normalization, and ReLU activation). Convolutional kernel sizes range from 5×5 (blocks 1–2) to 3×3
(blocks 3–8), with 384 channels throughout (except 3 RGB input channels). Average pooling layers (kernel
size 2, stride 2) are inserted after blocks 2, 4, and 6, with a final pooling layer (kernel size 2, stride 1)
after block 8. Each subject has a dedicated linear factorized readout head with spatial and feature weight
matrices initialized with L2-normalized spatial weights.

**Preprocessing.** Input images are center-cropped to square along the shorter dimension, resized to 224×224
pixels, and normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). For video stimuli (BMD, HAD),
a representative frame is extracted per stimulus. fMRI responses are modeled at the single-trial level.

**Model training partition.** The model was trained on naturalistic stimuli using curated MOSAIC train/validation
splits designed to prevent stimulus leakage across datasets. Training used the Adam optimizer (initial learning
rate 1e-4, weight decay 1e-4), ReduceLROnPlateau scheduling, mean squared error loss, and a batch size of 64.
Training was performed jointly across all subjects using a multi-head architecture.

**Model testing partition.** Models are evaluated on held-out naturalistic test images
and on artificial test images from the NSD or Deeprecon stimulus set. Naturalistic images consist
of real-world photographs (objects, scenes, people, animals) drawn from standard
computer vision datasets. Artificial images consist of controlled, non-naturalistic
visual stimuli such as gratings, noise patterns, checkerboards, and simple geometric
shapes that do not follow natural image statistics.

**Noise ceiling.** Vertex-wise noise ceilings were computed using the MOSAIC
preprocessing pipeline from GLMsingle beta estimates. Noise ceilings are provided
separately for naturalistic training stimuli, naturalistic test stimuli, and artificial
stimuli. Noise ceilings are reported for single-trial estimates (n-1) and for
repeat-averaged responses (n-avg).

**Vertex space mapping.** The model predicts 7,831 vertices, a subset of the full
HCP grayordinate space (91,282 vertices), because it only covers cortical surface vertices
within Glasser atlas groups 1-5 (excluding subcortical structures and remaining cortical
areas). ROI vertex indices from the Glasser atlas are defined in the full 91,282 space and
cannot be directly applied to the 7k model predictions. To align model predictions with
noise ceilings or extract ROI-specific responses: (1) expand predictions to full 91,282 space
using the vertex mapping from GlasserGroups 1-5, (2) index using ROI vertex indices or
noise ceiling values at the expanded positions. All noise ceilings and ROI definitions are
provided in the full 91,282 HCP grayordinate space for direct indexing after expansion.

**Output.** The model predicts fMRI responses for 7,831 vertices in visual cortex, corresponding to MMP 1.0
parcellation sections 1–5 (early visual cortex through higher-level ventral and dorsal visual streams).

Metadata
--------

**vertex_mapping_visual** : ``(7831,)`` - [visual variant only] Indices mapping visual cortex model predictions (GlasserGroups 1-5) to full 91k HCP space. Usage: pred_HCP = np.full((batch, 91282), np.nan); pred_HCP[:, vertex_mapping_visual] = predictions_7831

**glasser_group_id** : ``(7831,)`` - Array indicating which GlasserGroup (1-5 for visual) each prediction vertex belongs to. Allows filtering predictions by group.

**roi** : ``dict`` - ROI name → vertex indices in full HCP grayordinate space
    Available ROIs:
    L_V1, L_V2, L_V3, L_V4, L_V6, L_V3A, L_V7, L_IPS1, L_V3B,
    L_V6A, L_V8, L_FFC, L_PIT, L_VMV1, L_VMV3, L_VMV2, L_VVC,
    L_MST, L_LO1, L_LO2, L_MT, L_PH, L_V4t, L_FST, L_V3CD, L_LO3.
    Corresponding R_* entries exist for the right hemisphere.

**subject_info** : ``dict`` - Subject-specific information
    **participant_id** : ``str`` - Subject identifier

    **age** : ``int`` - Subject age

    **sex** : ``str`` - Subject sex

**stimuli** : ``dict`` - Stimulus-related arrays
    **filenames** : ``(70850,)`` - All stimulus filenames

    **alias** : ``(70850,)`` - Stimulus aliases

    **source** : ``(70850,)`` - Stimulus sources

    **train_idx** : ``(69566,)`` - Indices of training trials

    **test_idx** : ``(1284,)`` - Indices of test trials

    **train_filenames** : ``(69566,)`` - Training stimulus filenames

    **test_filenames** : ``(1284,)`` - Test stimulus filenames

    **reps** : ``(70850,)`` - Repetition count per stimulus for this subject

**noise_ceiling** : ``dict`` - Noise ceiling metrics
    **test_n-avg_noiseceiling** : ``(91282,)`` - Vertex-wise noise ceiling computed on naturalistic test stimuli (real-world photographic images) using repeat-averaged beta estimates.

    **test_n-1_noiseceiling** : ``(91282,)`` - Vertex-wise noise ceiling computed on naturalistic test stimuli (real-world photographic images) using single-trial beta estimates.

    **train_n-avg_noiseceiling** : ``(91282,)`` - Vertex-wise noise ceiling computed on naturalistic training stimuli (real-world photographic images used for model fitting) using repeat-averaged beta estimates.

    **train_n-1_noiseceiling** : ``(91282,)`` - Vertex-wise noise ceiling computed on naturalistic training stimuli (real-world photographic images used for model fitting) using single-trial beta estimates.

    **artificial_n-avg_noiseceiling** : ``(91282,)`` - Vertex-wise noise ceiling computed on artificial test stimuli (controlled non-naturalistic images such as gratings, noise patterns, and simple shapes) using repeat-averaged beta estimates.

    **artificial_n-1_noiseceiling** : ``(91282,)`` - Vertex-wise noise ceiling computed on artificial test stimuli (controlled non-naturalistic images such as gratings, noise patterns, and simple shapes) using single-trial beta estimates.

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

**Type**: ``dict``  
**Shape**: ``{dataset_name: {subject_id: [batch_size, n_vertices], ...}, ...}``  
**Description**:  
Nested dictionary with structure organized by dataset and subject:
{"BOLD5000": {"sub-01": array, "sub-02": array},
"NaturalScenesDataset": {"sub-01": array, "sub-02": array}, ...}
where each array has shape [batch_size, n_vertices] and dtype float32.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch.
   * - n_vertices
     - Number of visual cortex vertices (up to 7,831, based on ROI/vertex selection).

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
       | **Valid Values:** fmri-mosaic-CNN8_multihead_subAll_verticesVisual
       | **Example:** "fmri-mosaic-CNN8_multihead_subAll_verticesVisual"
   * - **subject**
     - | **Type:** str, list[str], or 'all'
       | **Required:** Yes
       | **Description:** Subject identifier(s). Can be:
       | - Single subject: "NSD-01"
       | - Multiple subjects: ["NSD-01", "BOLD5000-02", "THINGS-01"]
       | - All subjects: "all"
       | Format is DATASET-## where DATASET is one of the eight constituent datasets 
       | and ## is the zero-padded subject number. When multiple subjects are specified, 
       | the same ROI/vertex selection is applied to all subjects.
       | **Valid Values:** "BOLD5000-01", "BOLD5000-02", "BOLD5000-03", "BOLD5000-04", "deeprecon-01", "deeprecon-02", "deeprecon-03", "GOD-01", "GOD-02", "GOD-03", "GOD-04", "GOD-05", "NSD-01", "NSD-02", "NSD-03", "NSD-04", "NSD-05", "NSD-06", "NSD-07", "NSD-08", "THINGS-01", "THINGS-02", "THINGS-03", "BMD-01", "BMD-02", "BMD-03", "BMD-04", "BMD-05", "BMD-06", "BMD-07", "BMD-08", "BMD-09", "BMD-10", "NOD-01", "NOD-02", "NOD-03", "NOD-04", "NOD-05", "NOD-06", "NOD-07", "NOD-08", "NOD-09", "NOD-10", "NOD-11", "NOD-12", "NOD-13", "NOD-14", "NOD-15", "NOD-16", "NOD-17", "NOD-18", "NOD-19", "NOD-20", "NOD-21", "NOD-22", "NOD-23", "NOD-24", "NOD-25", "NOD-26", "NOD-27", "NOD-28", "NOD-29", "NOD-30", "HAD-01", "HAD-02", "HAD-03", "HAD-04", "HAD-05", "HAD-06", "HAD-07", "HAD-08", "HAD-09", "HAD-10", "HAD-11", "HAD-12", "HAD-13", "HAD-14", "HAD-15", "HAD-16", "HAD-17", "HAD-18", "HAD-19", "HAD-20", "HAD-21", "HAD-22", "HAD-23", "HAD-24", "HAD-25", "HAD-26", "HAD-27", "HAD-28", "HAD-29", "HAD-30"
       | **Example:** "NSD-01"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific ROIs and/or vertex indices. If not provided,
       | fMRI responses are generated for all visual cortex vertices.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of region-of-interest (ROI) labels from visual cortex to include.
       |     Available ROIs from the Glasser MMP 1.0 atlas.
       |     Prefix with 'L_' or 'R_' for hemisphere-specific selection.
       |     If multiple ROIs are listed, their vertices are concatenated.
       |     **Valid values:** "L_V1", "R_V1", "L_V2", "L_V3", "L_V4", "R_V2", "R_V3", "R_V4", "L_V6", "L_V3A", "L_V7", "L_IPS1", "L_V3B", "L_V6A", "R_V6", "R_V3A", "R_V7", "R_IPS1", "R_V3B", "R_V6A", "L_V8", "L_FFC", "L_PIT", "L_VMV1", "L_VMV3", "L_VMV2", "L_VVC", "R_V8", "R_FFC", "R_PIT", "R_VMV1", "R_VMV3", "R_VMV2", "R_VVC", "L_MST", "L_LO1", "L_LO2", "L_MT", "L_PH", "L_V4t", "L_FST", "L_V3CD", "L_LO3", "R_MST", "R_LO1", "R_LO2", "R_MT", "R_PH", "R_V4t", "R_FST", "R_V3CD", "R_LO3"
       |     **Example:** ['L_V1', 'R_V1', 'L_V4', 'R_V4']
       | 
       | **glasser_group**
       |     **Type:** int, list[int]
       |     **Description:** Glasser group(s) to include in predictions. Each group represents a functional
       |     subdivision of cortex based on the Glasser MMP 1.0 parcellation.
       |     
       |     For visual cortex model:
       |       Valid values: 1-5 Visual cortex
       |     
       |     Can be a single integer or list of integers. If multiple groups are specified,
       |     their vertices are concatenated.
       |     **Valid values:** 1, 2, 3, 4, 5
       |     **Example:** [1, 2]
       | 
       | **voxel_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which vertices to include.
       |     Must have exactly the same length as the number of available vertices (7,831).
       |     Each position set to 1 indicates that vertex should be included.
       |     WARNING: This operates in the model's ~7k prediction space, which does NOT
       |     directly correspond to positions in the full fsLR32k/HCP grayordinate space
       |     (91,282 vertices) where ROIs and noise ceilings are defined. If you need to
       |     select specific brain regions, use the 'roi' parameter instead, which handles
       |     the coordinate mapping automatically.
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
       | **Valid Values:** fmri-mosaic-CNN8_multihead_subAll_verticesVisual
       | **Example:** "fmri-mosaic-CNN8_multihead_subAll_verticesVisual"
   * - **subject**
     - | **Type:** str, list[str], or 'all'
       | **Required:** Yes
       | **Description:** Subject identifier(s). Can be:
       | - Single subject: "NSD-01"
       | - Multiple subjects: ["NSD-01", "BOLD5000-02", "THINGS-01"]
       | - All subjects: "all"
       | Format is DATASET-## where DATASET is one of the eight constituent datasets 
       | and ## is the zero-padded subject number. When multiple subjects are specified, 
       | the same ROI/vertex selection is applied to all subjects.
       | **Valid Values:** "BOLD5000-01", "BOLD5000-02", "BOLD5000-03", "BOLD5000-04", "deeprecon-01", "deeprecon-02", "deeprecon-03", "GOD-01", "GOD-02", "GOD-03", "GOD-04", "GOD-05", "NSD-01", "NSD-02", "NSD-03", "NSD-04", "NSD-05", "NSD-06", "NSD-07", "NSD-08", "THINGS-01", "THINGS-02", "THINGS-03", "BMD-01", "BMD-02", "BMD-03", "BMD-04", "BMD-05", "BMD-06", "BMD-07", "BMD-08", "BMD-09", "BMD-10", "NOD-01", "NOD-02", "NOD-03", "NOD-04", "NOD-05", "NOD-06", "NOD-07", "NOD-08", "NOD-09", "NOD-10", "NOD-11", "NOD-12", "NOD-13", "NOD-14", "NOD-15", "NOD-16", "NOD-17", "NOD-18", "NOD-19", "NOD-20", "NOD-21", "NOD-22", "NOD-23", "NOD-24", "NOD-25", "NOD-26", "NOD-27", "NOD-28", "NOD-29", "NOD-30", "HAD-01", "HAD-02", "HAD-03", "HAD-04", "HAD-05", "HAD-06", "HAD-07", "HAD-08", "HAD-09", "HAD-10", "HAD-11", "HAD-12", "HAD-13", "HAD-14", "HAD-15", "HAD-16", "HAD-17", "HAD-18", "HAD-19", "HAD-20", "HAD-21", "HAD-22", "HAD-23", "HAD-24", "HAD-25", "HAD-26", "HAD-27", "HAD-28", "HAD-29", "HAD-30"
       | **Example:** "NSD-01"

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-mosaic/model-CNN8_multihead_subAll_verticesVisual/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-mosaic-CNN8_multihead_subAll_verticesVisual",
        subject="NSD-01",
        selection={
            "roi": ["L_V1", "R_V1", "L_V4", "R_V4"],
            "glasser_group": [1, 2],
            "voxel_index": [0, 0, '...', 1, 1, 0]
        }
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
    
    # The in silico fMRI responses will be a dict of shape:
    # {dataset_name: {subject_id: [batch_size, n_vertices], ...}, ...}
    # where:
    # - n_vertices: Number of visual cortex vertices (up to 7,831, based on ROI/vertex selection).
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        images,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-mosaic-CNN8_multihead_subAll_verticesVisual",
        subject="NSD-01"
    )
    

References
---------

* MOSAIC Paper (Lahner et al., 2025): https://www.biorxiv.org/content/10.64898/2025.11.28.690060v1
* MOSAIC Repository: https://github.com/murtylab/mosaic-dataset/tree/1d26b5b4ccdf77eba76e47404f6b9041e28e9a33
* BOLD5000 Dataset (Chang et al., 2019): https://openneuro.org/datasets/ds001499
* BOLD5000 Paper (Chang et al., 2019): https://arxiv.org/abs/1809.01281
* deeprecon Dataset (Shen et al., 2019): https://openneuro.org/datasets/ds001506
* deeprecon Paper (Shen et al., 2019): https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633
* GenericObjectDecoding/GOD Dataset (Horikawa & Kamitani, 2017): https://openneuro.org/datasets/ds001246
* GenericObjectDecoding/GOD Paper (Horikawa & Kamitani, 2017): https://www.nature.com/articles/ncomms15037
* NaturalScenesDataset/NSD Dataset (Allen et al., 2022): https://registry.opendata.aws/nsd/
* NaturalScenesDataset/NSD Paper (Allen et al., 2022): https://www.nature.com/articles/s41593-021-00962-x
* THINGS fMRI Dataset (Hebart et al., 2023): https://openneuro.org/datasets/ds004192
* THINGS fMRI Paper (Hebart et al., 2023): https://elifesciences.org/articles/82580
* BOLDMomentsDataset/BMD Dataset: https://openneuro.org/datasets/ds005165
* BOLDMomentsDataset/BMD Paper: https://www.nature.com/articles/s41467-024-50310-3
* NaturalObjectDataset/NOD Dataset (Gong et al., 2023): https://openneuro.org/datasets/ds004496
* NaturalObjectDataset/NOD Paper (Gong et al., 2023): https://www.nature.com/articles/s41597-023-02471-x
* HumanActionsDataset/HAD Dataset (Zhou et al., 2023): https://openneuro.org/datasets/ds004488
* HumanActionsDataset/HAD Paper (Zhou et al., 2023): https://www.nature.com/articles/s41597-023-02325-6