=======================
fmri-lebel2023-opt_1_3b
=======================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - LeBel et al. (2023)
   * - Species
     - Human
   * - Stimuli
     - Text (spoken narrative stories with word onset times)
   * - Model Type
     - OPT-1.3B–based linear encoding model (contextual LLM embeddings + ridge regression)
   * - Creator
     - Richard J. Antonello

Description
----------

This encoding model predicts voxelwise BOLD fMRI responses from natural language
input using contextual embeddings from OPT-1.3B (layer 18) mapped to brain
activity via voxelwise ridge regression, following the scaling-laws approach of
Antonello (NeurIPS 2023).

**Neural data.** The model was trained on the LeBel et al. (2023) dataset, in
which 8 participants passively listened to narrative stories from The Moth and
Modern Love podcasts during fMRI scanning. Three participants (UTS01–UTS03)
listened to 84 stories (~16 hours) across 15 sessions; the remaining five
(UTS04–UTS08) listened to 27 stories (~6 hours) across 5 sessions. Functional
data were acquired at 3T (TR=2s, 2.6mm isotropic) and preprocessed with motion
correction, cross-run alignment, Savitzky-Golay detrending, and z-scoring.
The data lives in volumetric voxel space (cortical mask applied to the 84×84×54
acquisition grid); the number of cortical voxels varies per subject (81K–109K).

**Feature extraction.** Each word in the input is processed through OPT-1.3B
(Zhang et al., 2022), a 1.3-billion-parameter decoder-only transformer language
model. The hidden state at the last BPE token of each word is extracted from
layer 18 (of 24 total layers), yielding a 2,048-dimensional contextual embedding
per word. A dynamic context window is used for computational efficiency: the
context grows word-by-word until 512 words, then resets to 256 words
(Antonello et al., 2023, Section 2.3).

**Temporal processing.** The model requires word onset times as input, since the
temporal structure of the stimulus is essential for accurate predictions.
The temporal pipeline is:

  1. **Lanczos downsampling.** Word-level feature vectors (2,048-dim impulses at
     each word onset) are low-pass filtered and resampled to the fMRI acquisition
     rate (TR=2s) using a Lanczos filter with a 3-lobe window. This converts
     discrete word events into a continuous feature time series aligned to the
     fMRI sampling grid.

  2. **Z-scoring.** The downsampled features are standardised (zero mean, unit
     variance) across time for each feature dimension.

  3. **Finite Impulse Response (FIR) delays.** To model the hemodynamic response
     delay, the features are concatenated with copies delayed by 1, 2, 3, and
     4 TRs (2, 4, 6, and 8 seconds). This expands the feature vector from
     2,048 to 8,192 dimensions at each TR.

  4. **Prediction.** The delayed feature matrix is multiplied by the pre-trained
     ridge regression weights to produce predicted BOLD responses at each TR.

**Training.** For subjects UTS01–UTS03, 83 stories were used for training
(~16 hours of speech); for UTS04–UTS08, 25–26 stories were used (~5.5 hours).
Ridge regression was fitted independently per voxel. The ridge regularisation
parameter was selected per voxel via bootstrap cross-validation. Training features
were trimmed by 10 TRs from the start and 5 TRs from the end. One story
("Where There's Smoke") was held out for testing and repeated across scanning
sessions (10 repeats for UTS01–UTS03, 5 repeats for UTS04–UTS08). Test features
were trimmed by 50 TRs from the start to exclude long-context artifacts
(Antonello et al., Section 3.5) and 5 TRs from the end.

**Noise ceiling.** Computed using the Schoppe et al. (2016) signal/noise power
decomposition on repeated presentations of the test story. For each voxel,
noise power (NP) is the mean within-repeat temporal variance across repeats,
and signal power (SP) is derived by removing the noise contribution from the
variance of the repeat-averaged response: SP = (1/(N−1)) × (N × var(mean) − NP).
The maximum attainable correlation is then CCmax = √(1 / (1 + (1/N) × (NP/SP − 1))).
CCmax is floored at 0.25 to regularise noisy voxels (Antonello et al., Section 2.5).
The first 40 TRs of each repeat are excluded to match the test evaluation window.
Noise ceiling estimates from 5 repeats (UTS04–UTS08) are noisier than from 10
repeats (UTS01–UTS03).

**Output.** The model returns a 2D array of predicted BOLD responses at each TR,
across all cortical voxels (or a user-specified subset via ROI selection).
Responses are in z-scored units consistent with the training data preprocessing.

Metadata
--------

**fmri**

    **subject_id** : ``str`` - Subject identifier (e.g., 'UTS01')

    **n_voxels** : ``int`` - Total number of cortical voxels (varies per subject)

    **tr** : ``float`` - Repetition time in seconds (2.0)

    **voxel_size_mm** : ``float`` - Isotropic voxel size in mm (2.6)
**roi**

    **{roi_name}** : ``(n_voxels,) bool`` - Voxel mask per ROI
**encoding_model**

    **train_stories** : ``(n_train,)`` - Story names used for training (83 for UTS01–03, 25–26 for UTS04–08)

    **test_stories** : ``(n_test,)`` - Story names used for testing (1 story)

    **noise_ceiling** : ``(n_voxels,)`` - Voxelwise noise ceiling CCmax (Schoppe et al., floored at 0.25)

    **correlation** : ``(n_voxels,)`` - Voxelwise prediction accuracy (Pearson's r) on test story

    **cc_norm** : ``(n_voxels,)`` - Noise-ceiling-normalised correlation (CCabs / CCmax)

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``dict``
   * - Description
     - | A dictionary with two required keys:
       |   ``words``        — list of str: the words of the stimulus in order.
       |   ``word_onsets``  — list of float or numpy.ndarray: the onset time of each
       |                      word in seconds (relative to an arbitrary t=0).
       | Both lists must have the same length. Output TRs are automatically generated
       | from word onsets at the fMRI acquisition rate (TR=2s).
   * - Example
     - {
       "words": ["I", "reached", "over", "and", "slowly", "undid", "my", "seatbelt"],
       "word_onsets": [0.0, 0.3, 0.6, 0.85, 1.1, 1.5, 1.8, 2.0]
       }

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``(n_TRs, n_voxels)``
   * - Description
     - | The output is a 2D array containing predicted z-scored BOLD fMRI responses.
       | Each row corresponds to one fMRI volume (TR=2s), each column to one cortical
       | voxel (or a subset if ROI selection is applied).
   * - Dimensions
     - | **n_TRs**: Number of fMRI volumes (determined by stimulus duration and TR=2s).
       | **n_voxels**: Number of selected voxels for which in silico fMRI responses are generated.

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
       | **Valid Values:** fmri-lebel2023-opt_1_3b
       | **Example:** "fmri-lebel2023-opt_1_3b"
   * - **subject**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Subject ID from the LeBel et al. (2023) dataset. UTS01–UTS03 have the
       | extended dataset (~16 hours, 83 training stories, 10 test repeats).
       | UTS04–UTS08 have the base dataset (~5.5 hours, 25–26 training stories,
       | 5 test repeats). Encoding performance scales with training data size.
       | **Valid Values:** "UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"
       | **Example:** "UTS03"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which voxels to include in the model responses.
       | If not provided, responses are generated for all cortical voxels.
       | Not all ROIs are available for every subject — use get_model_metadata()
       | to check availability.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of ROI names for which in silico fMRI responses are generated.
       |     Not all ROIs are available for every subject — use
       |     get_model_metadata() to check availability.
       |     **Valid values:** "A1", "AC", "ATFP", "Broca", "EBA", "FBA", "FEF", "FFA", "FFA1", "FO", "IFSFP", "IPS", "LO", "M1F", "M1H", "M1M", "OFA", "OPA", "PMvh", "PPA", "RSC", "S1F", "S1H", "S1M", "S2F", "S2H", "S2M", "SEF", "SMFA", "SMHA", "TOS", "V1", "V2", "V3", "V3A", "V3B", "V4", "V7", "VO", "hMT", "pSTS", "sPMv"
       |     **Example:** ['AC', 'Broca']
       | 
       | **voxel_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector with ones indicating the voxels for
       |     which in silico fMRI responses are generated. This vector must have
       |     exactly the same length as the number of voxels for the selected
       |     subject:
       |     - UTS01:   81,126 voxels
       |     - UTS02:   94,251 voxels
       |     - UTS03:   95,556 voxels
       |     - UTS04:  109,469 voxels
       |     - UTS05:   99,322 voxels
       |     - UTS06:   92,198 voxels
       |     - UTS07:   94,395 voxels
       |     - UTS08:   97,023 voxels
       |     The voxels from the one-hot encoded vector are included in addition to
       |     any voxels selected via the "roi" key. If both are provided, the union
       |     of all selected voxels is used.
       |     **Example:** [0, 0, '...', 1, 1, 0]
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. OPT-1.3B requires approximately 3 GB of VRAM
       | in float16 (GPU) or approximately 5 GB of RAM in float32 (CPU). Using
       | 'auto' will select CUDA if available, otherwise CPU. GPU inference is
       | recommended for faster feature extraction.
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
     - | **Type:** dict
       | **Required:** Yes
       | **Description:** A dictionary containing the words and their onset times:
       |   - "words": list of str — the words of the stimulus in presentation order.
       |   - "word_onsets": list of float — onset time of each word in seconds.
       | Both lists must have the same length.
       | **Example:**
       | {
       | "words": ["I", "reached", "over", "and", "slowly", "undid", "my", "seatbelt"],
       | "word_onsets": [0.0, 0.3, 0.6, 0.85, 1.1, 1.5, 1.8, 2.0]
       | }
   * - **return_metadata**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to return the encoding model's metadata together with the in silico neural responses.
       | **Example:** True
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding.
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
       | **Valid Values:** fmri-lebel2023-opt_1_3b
       | **Example:** "fmri-lebel2023-opt_1_3b"
   * - **subject**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Subject ID from the LeBel et al. (2023) dataset. UTS01–UTS03 have the
       | extended dataset (~16 hours, 83 training stories, 10 test repeats).
       | UTS04–UTS08 have the base dataset (~5.5 hours, 25–26 training stories,
       | 5 test repeats). Encoding performance scales with training data size.
       | **Valid Values:** "UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"
       | **Example:** "UTS03"

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-lebel2023/model-opt_1_3b_ridge/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-lebel2023-opt_1_3b",
        subject="UTS03",
        selection={
            "roi": ["AC", "Broca"],
            "voxel_index": [0, 0, '...', 1, 1, 0]
        }
    )
    
    # Prepare the stimulus 
    words = ["the", "audience", "erupted", "into", "laughter", "and", "applause",
            "she", "walked", "off", "the", "stage", "quietly"]
    # Onsets
    onsets = [0.0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0,
              4.0, 4.33, 4.66, 5.0, 5.33, 5.66]

    stimulus = {
        "words": words,
        "word_onsets": onsets
    }

    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # (n_TRs, n_voxels)
    # where:
    # - n_TRs: Number of fMRI volumes (determined by stimulus duration and TR=2s).
    # - n_voxels: Number of selected voxels for which in silico fMRI responses are generated.
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-lebel2023-opt_1_3b",
        subject="UTS03"
    )
    

References
---------

* Model building code: https://github.com/gifale95/BERG/tree/main/berg_creation_code/02_train_encoding_models/train_dataset-lebel2023/model-ridge/train_ridge.py
* Scaling laws for language encoding models in fMRI paper (Antonello et al., 2023): https://arxiv.org/abs/2305.11863
* Scaling laws code & data: https://github.com/HuthLab/encoding-model-scaling-laws
* Dataset paper (LeBel et al., 2023): https://doi.org/10.1038/s41597-023-02437-z
* Dataset (OpenNeuro): https://openneuro.org/datasets/ds003020
* Dataset code: https://github.com/HuthLab/deep-fMRI-dataset
* OPT language models (Zhang et al., 2022): https://arxiv.org/abs/2205.01068
* Noise ceiling method (Schoppe et al., 2016): https://doi.org/10.3389/fncom.2016.00010