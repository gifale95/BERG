=============================================
fmri-mosaic-CNN8_multihead_subNSD_verticesAll
=============================================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - MOSAIC (NSD subjects, all cortical vertices)
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
whole-brain cortical fMRI responses (GlasserGroups 1-22,57,051 cortical vertices) for Natural Scenes
Dataset (NSD) subjects. The model uses a shared 8-layer convolutional core with subject-specific
linear factorized readout heads.

**Neural data.** The model was trained on 8 subjects from the Natural Scenes Dataset (NSD).
All data underwent the shared MOSAIC preprocessing pipeline (fMRIPrep and GLMsingle) to ensure consistency.

**Model architecture.** The CNN8 core consists of eight 2D convolutional blocks (each with 2D convolution,
batch normalization, and ReLU activation). Convolutional kernel sizes range from 5×5 (blocks 1-2) to 3×3
(blocks 3-8), with 384 channels throughout (except 3 RGB input channels). Average pooling layers (kernel
size 2, stride 2) are inserted after blocks 2, 4, and 6, with a final pooling layer (kernel size 2, stride 1)
after block 8. Each NSD subject has a dedicated linear factorized readout head with spatial and feature weight
matrices initialized with L2-normalized spatial weights.

**Preprocessing.** Input images are center-cropped to square along the shorter dimension, resized to 224×224
pixels, and normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). fMRI data are averaged
across trials per stimulus to increase signal-to-noise ratio.

**Model training partition.** The model was trained on naturalistic stimuli using a 90%/10% train/validation
split with early stopping (patience=7 epochs, delta=0.002). Training used Adam optimizer (initial learning rate
1e-4, weight decay 1e-4) with ReduceLROnPlateau scheduling (factor=0.3, patience=6 epochs), mean squared
error loss, and batch size 64 with random sampling across all NSD subjects. Maximum training duration was 
100 epochs.

**Model testing partition.** Models are evaluated on held-out naturalistic test images
and on artificial test images from the NSD stimulus set. Naturalistic images consist
of real-world photographs (objects, scenes, people, animals) drawn from standard
computer vision datasets. Artificial images consist of controlled, non-naturalistic
visual stimuli such as gratings, noise patterns, checkerboards, and simple geometric
shapes that do not follow natural image statistics.

**Noise ceiling.** Vertex-wise noise ceilings were computed using the MOSAIC
preprocessing pipeline from GLMsingle beta estimates. Noise ceilings are provided
separately for naturalistic training stimuli, naturalistic test stimuli, and artificial
stimuli. Noise ceilings are reported for single-trial estimates (n-1) and for
repeat-averaged responses (n-avg).

**Vertex space mapping.** The model predicts 57,051 vertices, a subset of the full
HCP grayordinate space (91,282 vertices), because it only covers cortical surface vertices
within Glasser atlas groups 1-22 (excluding subcortical structures and remaining cortical
areas). ROI vertex indices from the Glasser atlas are defined in the full 91,282 space and
cannot be directly applied to the 57k model predictions. To align model predictions with
noise ceilings or extract ROI-specific responses: (1) expand predictions to full 91,282 space
using the vertex mapping from GlasserGroups 1-22, (2) index using ROI vertex indices or
noise ceiling values at the expanded positions. All noise ceilings and ROI definitions are
provided in the full 91,282 HCP grayordinate space for direct indexing after expansion.

**Output.** The model predicts fMRI responses for 57,051 vertices spanning most of cortex, corresponding
to MMP 1.0 parcellation GlasserGroups 1-22 (visual, somatomotor, auditory, and higher-order association
cortices).

Metadata
--------

**vertex_mapping_all** : ``(57051,)`` - [whole_cortex variant only] Indices mapping full cortex model predictions (GlasserGroups 1-22) to full 91k HCP space. Usage: pred_HCP = np.full((batch, 91282), np.nan); pred_HCP[:, vertex_mapping_all] = predictions_57051

**glasser_group_id** : ``(57051,)`` - Array indicating which GlasserGroup (among Galsser groups 1-22) each prediction vertex belongs to. Allows filtering predictions by group.

**roi** : ``dict`` - ROI name → vertex indices in full HCP grayordinate space
    Available ROIs:
    L_V1, L_V2, L_V3, L_V4, L_V6, L_V3A, L_V7, L_IPS1, L_V3B, L_V6A,
    L_V8, L_FFC, L_PIT, L_VMV1, L_VMV3, L_VMV2, L_VVC, L_MST, L_LO1,
    L_LO2, L_MT, L_PH, L_V4t, L_FST, L_V3CD, L_LO3, L_4, L_3b, L_1,
    L_2, L_3a, L_5m, L_5mv, L_5L, L_24dd, L_24dv, L_SCEF, L_6ma,
    L_6mp, L_FEF, L_PEF, L_55b, L_6d, L_6v, L_6r, L_6a, L_43, L_OP4,
    L_OP1, L_OP2-3, L_PFcm, L_FOP1, L_A1, L_RI, L_PBelt, L_MBelt,
    L_LBelt, L_TA2, L_STGa, L_A5, L_STSda, L_STSdp, L_STSvp, L_A4,
    L_STSva, L_52, L_PoI2, L_FOP4, L_MI, L_Pir, L_AVI, L_AAIC, L_FOP3,
    L_FOP2, L_PoI1, L_Ig, L_FOP5, L_PI, L_EC, L_PreS, L_H, L_PeEc,
    L_PHA1, L_PHA3, L_PHA2, L_TGd, L_TE1a, L_TE1p, L_TE2a, L_TF, L_TE2p,
    L_PHT, L_TGv, L_TE1m, L_PSL, L_STV, L_TPOJ1, L_TPOJ2, L_TPOJ3,
    L_7Pm, L_7AL, L_7Am, L_7PL, L_7PC, L_LIPv, L_VIP, L_MIP, L_LIPd,
    L_AIP, L_PFt, L_PGp, L_IP2, L_IP1, L_IP0, L_PFop, L_PF, L_PFm, L_PGi,
    L_PGs, L_RSC, L_POS2, L_PCV, L_7m, L_POS1, L_23d, L_v23ab, L_d23ab,
    L_31pv, L_23c, L_ProS, L_DVT, L_31pd, L_31a, L_p24pr, L_33pr, L_a24pr,
    L_p32pr, L_a24, L_d32, L_8BM, L_p32, L_10r, L_9m, L_10v, L_25, L_s32,
    L_a32pr, L_p24, L_47m, L_10d, L_a47r, L_a10p, L_10pp, L_11l, L_13l,
    L_OFC, L_47s, L_pOFC, L_p10p, L_44, L_45, L_47l, L_IFJa, L_IFJp, L_IFSp,
    L_IFSa, L_p47r, L_SFL, L_8Av, L_8Ad, L_8BL, L_9p, L_8C, L_p9-46v, L_46,
    L_a9-46v, L_9-46d, L_9a, L_i6-8, L_s6-8.
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

**Type**: ``dict or numpy.ndarray``  
**Shape**: ``{'NaturalScenesDataset': {'sub-01': [batch_size, n_vertices], ...}} ``  
**Description**:  
It returns a nested dictionary with structure:
{"NaturalScenesDataset": {"sub-01": array, "sub-02": array, ...}}
where each array has shape [batch_size, n_vertices].

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch.
   * - n_vertices
     - Number of cortical vertices (up to 57,051, based on ROI/vertex selection).

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
       | **Valid Values:** fmri-mosaic-CNN8_multihead_subNSD_verticesAll
       | **Example:** "fmri-mosaic-CNN8_multihead_subNSD_verticesAll"
   * - **subject**
     - | **Type:** int, list[int], or str
       | **Required:** Yes
       | **Description:** NSD subject ID(s). Can be:
       | - Single subject: 1
       | - Multiple subjects: [1, 2, 3]
       | - All subjects: "all"
       | When multiple subjects are specified, the same ROI/vertex selection is applied to all subjects.
       | **Valid Values:** 1, 2, 3, 4, 5, 6, 7, 8, "all"
       | **Example:** 1
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific ROIs and/or vertex indices. If not provided,
       | fMRI responses are generated for all cortical vertices.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of region-of-interest (ROI) labels from the Glasser MMP 1.0 atlas to include.
       |     Available ROIs span visual, somatomotor, auditory, and association cortices
       |     (GlasserGroups 1-22).
       |     
       |     Prefix with 'L_' or 'R_' for hemisphere-specific selection.
       |     If multiple ROIs are listed, their vertices are concatenated.
       |     **Valid values:** "L_V1", "R_V1", "L_V2", "L_V3", "L_V4", "R_V2", "R_V3", "R_V4", "L_V6", "L_V3A", "L_V7", "L_IPS1", "L_V3B", "L_V6A", "R_V6", "R_V3A", "R_V7", "R_IPS1", "R_V3B", "R_V6A", "L_V8", "L_FFC", "L_PIT", "L_VMV1", "L_VMV3", "L_VMV2", "L_VVC", "R_V8", "R_FFC", "R_PIT", "R_VMV1", "R_VMV3", "R_VMV2", "R_VVC", "L_MST", "L_LO1", "L_LO2", "L_MT", "L_PH", "L_V4t", "L_FST", "L_V3CD", "L_LO3", "R_MST", "R_LO1", "R_LO2", "R_MT", "R_PH", "R_V4t", "R_FST", "R_V3CD", "R_LO3", "L_4", "L_3b", "L_1", "L_2", "L_3a", "R_4", "R_3b", "R_1", "R_2", "R_3a", "L_5m", "L_5mv", "L_5L", "L_24dd", "L_24dv", "L_SCEF", "L_6ma", "L_6mp", "R_5m", "R_5mv", "R_5L", "R_24dd", "R_24dv", "R_SCEF", "R_6ma", "R_6mp", "L_FEF", "L_PEF", "L_55b", "L_6d", "L_6v", "L_6r", "L_6a", "R_FEF", "R_PEF", "R_55b", "R_6d", "R_6v", "R_6r", "R_6a", "L_43", "L_OP4", "L_OP1", "L_OP2-3", "L_PFcm", "L_FOP1", "R_43", "R_OP4", "R_OP1", "R_OP2-3", "R_PFcm", "R_FOP1", "L_A1", "L_RI", "L_PBelt", "L_MBelt", "L_LBelt", "R_A1", "R_RI", "R_PBelt", "R_MBelt", "R_LBelt", "L_TA2", "L_STGa", "L_A5", "L_STSda", "L_STSdp", "L_STSvp", "L_A4", "L_STSva", "R_TA2", "R_STGa", "R_A5", "R_STSda", "R_STSdp", "R_STSvp", "R_A4", "R_STSva", "L_52", "L_PoI2", "L_FOP4", "L_MI", "L_Pir", "L_AVI", "L_AAIC", "L_FOP3", "L_FOP2", "L_PoI1", "L_Ig", "L_FOP5", "L_PI", "R_52", "R_PoI2", "R_FOP4", "R_MI", "R_Pir", "R_AVI", "R_AAIC", "R_FOP3", "R_FOP2", "R_PoI1", "R_Ig", "R_FOP5", "R_PI", "L_EC", "L_PreS", "L_H", "L_PeEc", "L_PHA1", "L_PHA3", "L_PHA2", "R_EC", "R_PreS", "R_H", "R_PeEc", "R_PHA1", "R_PHA3", "R_PHA2", "L_TGd", "L_TE1a", "L_TE1p", "L_TE2a", "L_TF", "L_TE2p", "L_PHT", "L_TGv", "L_TE1m", "R_TGd", "R_TE1a", "R_TE1p", "R_TE2a", "R_TF", "R_TE2p", "R_PHT", "R_TGv", "R_TE1m", "L_PSL", "L_STV", "L_TPOJ1", "L_TPOJ2", "L_TPOJ3", "R_PSL", "R_STV", "R_TPOJ1", "R_TPOJ2", "R_TPOJ3", "L_7Pm", "L_7AL", "L_7Am", "L_7PL", "L_7PC", "L_LIPv", "L_VIP", "L_MIP", "L_LIPd", "L_AIP", "R_7Pm", "R_7AL", "R_7Am", "R_7PL", "R_7PC", "R_LIPv", "R_VIP", "R_MIP", "R_LIPd", "R_AIP", "L_PFt", "L_PGp", "L_IP2", "L_IP1", "L_IP0", "L_PFop", "L_PF", "L_PFm", "L_PGi", "L_PGs", "R_PFt", "R_PGp", "R_IP2", "R_IP1", "R_IP0", "R_PFop", "R_PF", "R_PFm", "R_PGi", "R_PGs", "L_RSC", "L_POS2", "L_PCV", "L_7m", "L_POS1", "L_23d", "L_v23ab", "L_d23ab", "L_31pv", "L_23c", "L_ProS", "L_DVT", "L_31pd", "L_31a", "R_RSC", "R_POS2", "R_PCV", "R_7m", "R_POS1", "R_23d", "R_v23ab", "R_d23ab", "R_31pv", "R_23c", "R_ProS", "R_DVT", "R_31pd", "R_31a", "L_p24pr", "L_33pr", "L_a24pr", "L_p32pr", "L_a24", "L_d32", "L_8BM", "L_p32", "L_10r", "L_9m", "L_10v", "L_25", "L_s32", "L_a32pr", "L_p24", "R_p24pr", "R_33pr", "R_a24pr", "R_p32pr", "R_a24", "R_d32", "R_8BM", "R_p32", "R_10r", "R_9m", "R_10v", "R_25", "R_s32", "R_a32pr", "R_p24", "L_47m", "L_10d", "L_a47r", "L_a10p", "L_10pp", "L_11l", "L_13l", "L_OFC", "L_47s", "L_pOFC", "L_p10p", "R_47m", "R_10d", "R_a47r", "R_a10p", "R_10pp", "R_11l", "R_13l", "R_OFC", "R_47s", "R_pOFC", "R_p10p", "L_44", "L_45", "L_47l", "L_IFJa", "L_IFJp", "L_IFSp", "L_IFSa", "L_p47r", "R_44", "R_45", "R_47l", "R_IFJa", "R_IFJp", "R_IFSp", "R_IFSa", "R_p47r", "L_SFL", "L_8Av", "L_8Ad", "L_8BL", "L_9p", "L_8C", "L_p9-46v", "L_46", "L_a9-46v", "L_9-46d", "L_9a", "L_i6-8", "L_s6-8", "R_SFL", "R_8Av", "R_8Ad", "R_8BL", "R_9p", "R_8C", "R_p9-46v", "R_46", "R_a9-46v", "R_9-46d", "R_9a", "R_i6-8", "R_s6-8"
       |     **Example:** ['L_V1', 'R_V1', 'L_FEF', 'R_FEF']
       | 
       | **glasser_group**
       |     **Type:** int, list[int]
       |     **Description:** Glasser group(s) to include in predictions. Each group represents a functional
       |     subdivision of cortex based on the Glasser MMP 1.0 parcellation.
       |     
       |     For whole cortex model:
       |       Valid values: 1-22
       |       Groups 1-5: Visual cortex
       |       Groups 6-22: Sensorimotor, auditory, language, and association cortex
       |     
       |     Can be a single integer or list of integers. If multiple groups are specified,
       |     their vertices are concatenated.
       |     **Valid values:** "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"
       |     **Example:** None
       | 
       | **voxel_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which vertices to include.
       |     Must have exactly the same length as the number of available vertices (57,051).
       |     Each position set to 1 indicates that vertex should be included.
       |     WARNING: This operates in the model's ~57k prediction space, which does NOT
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
       | **Valid Values:** fmri-mosaic-CNN8_multihead_subNSD_verticesAll
       | **Example:** "fmri-mosaic-CNN8_multihead_subNSD_verticesAll"
   * - **subject**
     - | **Type:** int, list[int], or str
       | **Required:** Yes
       | **Description:** NSD subject ID(s). Can be:
       | - Single subject: 1
       | - Multiple subjects: [1, 2, 3]
       | - All subjects: "all"
       | When multiple subjects are specified, the same ROI/vertex selection is applied to all subjects.
       | **Valid Values:** 1, 2, 3, 4, 5, 6, 7, 8, "all"
       | **Example:** 1

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-mosaic/model-CNN8_multihead_subNSD_verticesAll/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-mosaic-CNN8_multihead_subNSD_verticesAll",
        subject=1,
        selection={
            "roi": ["L_V1", "R_V1", "L_FEF", "R_FEF"],
            "glasser_group": None,
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
    
    # The in silico fMRI responses will be a dict or numpy.ndarray of shape:
    # {'NaturalScenesDataset': {'sub-01': [batch_size, n_vertices], ...}} 
    # where:
    # - n_vertices: Number of cortical vertices (up to 57,051, based on ROI/vertex selection).
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        images,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-mosaic-CNN8_multihead_subNSD_verticesAll",
        subject=1
    )
    

References
---------

* MOSAIC Paper (Lahner et al., 2025): https://www.biorxiv.org/content/10.64898/2025.11.28.690060v1
* MOSAIC Repository: https://github.com/murtylab/mosaic-dataset/tree/1d26b5b4ccdf77eba76e47404f6b9041e28e9a33
* NaturalScenesDataset/NSD Dataset (Allen et al., 2022): https://registry.opendata.aws/nsd/
* NaturalScenesDataset/NSD Paper (Allen et al., 2022): https://www.nature.com/articles/s41593-021-00962-x