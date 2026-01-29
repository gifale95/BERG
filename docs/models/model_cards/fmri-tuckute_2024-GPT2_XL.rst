=========================
fmri-tuckute_2024-GPT2_XL
=========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - Tuckute et. al (2024)
   * - Species
     - Human
   * - Stimuli
     - Text
   * - Model Type
     - GPT2-XL–based linear encoding model (LLM embeddings + ridge regression)
   * - Creator
     - Greta Tuckute

Description
----------

This encoding model predicts activity in the human left-hemisphere (LH) language network from 
natural language input. It uses GPT2-XL representations mapped to fMRI BOLD responses via 
linear ridge regression, enabling prediction and closed-loop control of language network activity.

**Neural data.** Trained on fMRI data from five participants reading 1,000 diverse six-word sentences 
in a rapid event-related design. The LH language network was defined per participant using a functional 
localizer (top 10% language-responsive voxels across five parcels: IFGorb, IFG, MFG, AntTemp, PostTemp). 
BOLD responses were averaged across voxels within each fROI, then averaged across participants and 
z-scored session-wise. Target values are participant-averaged scalars (one per sentence per fROI).

**Architecture.** Linear ridge regression model mapping GPT2-XL layer 22 last-token embeddings 
(sentence-level representations) to predicted BOLD responses for six ROIs: lang_LH_AntTemp, lang_LH_IFG, 
lang_LH_IFGorb, lang_LH_MFG, lang_LH_PostTemp, and lang_LH_netw (network average). Layer 22 was selected 
via 5-fold cross-validation across all 48 GPT2-XL layers, achieving r ≈ 0.38 predictivity (noise ceiling 
r ≈ 0.56).

**Training.** Final model trained on all 1,000 sentences using GPT2-XL layer 22 embeddings. Ridge 
regularization parameter selected via nested cross-validation. No fine-tuning of GPT2-XL performed. 
Sentences preprocessed to six words with printable ASCII and standardized punctuation.

**Evaluation.** Model tested on held-out baseline sentences, novel model-selected "drive" and "suppress" 
sentences from external corpora, and new participants. Successfully demonstrated closed-loop control by 
reliably modulating fMRI responses with model-selected sentences.

**Output.** Six scalar values per sentence: z-scored, participant-averaged BOLD response magnitudes for 
five language fROIs plus the network average, representing predicted engagement of each region and the 
overall language system.

Metadata
--------

**rois** : ``list`` - Six functionally defined ROIs of the language network ['lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb', 'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw']

**noise_ceiling** : ``(6,)`` - Noise ceiling estimates (Pearson's r) for each ROI

**noise_ceiling_snr** : ``(6,)`` - Noise ceiling signal-to-noise ratio for each ROI

**optimal_layer** : ``int`` - GPT2-XL transformer layer selected via cross-validation (layer 22)

**train_stimulus_ids** : ``(1000,)`` - Stimulus identifiers for training sentences (e.g., 'beta-control-neural-T.1' ... 'beta-control-neural-T.1000')

**train_sentences** : ``(1000,)`` - Text content of the 1000 training sentences

**train_targets** : ``(1000,)`` - Participant-averaged (N=5), z-scored BOLD response magnitude for each training sentence (lang_LH_netw)

**drive_stimulus_ids** : ``(250,)`` - Stimulus identifiers for drive sentences predicted to elicit maximal language network responses

**drive_sentences** : ``(250,)`` - Text content of the 250 drive sentences

**suppress_stimulus_ids** : ``(250,)`` - Stimulus identifiers for suppress sentences predicted to elicit minimal language network responses

**suppress_sentences** : ``(250,)`` - Text content of the 250 suppress sentences

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``list[str] or numpy.ndarray``
   * - Description
     - List or array of natural language sentences to encode.
       
       While the model accepts sentences of any length, it was trained exclusively on
       six-word sentences. Performance may vary for sentences of different lengths.
   * - Example
     - ["Taste that fowl and those fish.", 
       "I'm progressive and you fall right."]

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``(n_sentences, n_rois)``
   * - Description
     - The output is a numpy array containing predicted z-scored BOLD response magnitudes 
       for the left-hemisphere language network.
       Each row corresponds to one input sentence (in input order), and each column
       corresponds to one language network ROI (or a subset if ROI selection is applied).
       
       **ROI ordering:**
       - When no selection is used: ROIs appear in their default order (see below)
       - When selection is used: ROIs appear in the order specified in the selection parameter

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - n_sentences
     - Number of input sentences.
   * - n_rois
     - | Number of ROIs in the output (6 by default, or fewer if ROI selection is applied).
       | Default ROI order (when no selection is specified):
       | - lang_LH_AntTemp: Anterior temporal lobe
       | - lang_LH_IFG: Inferior frontal gyrus
       | - lang_LH_IFGorb: Inferior frontal gyrus orbitalis
       | - lang_LH_MFG: Middle frontal gyrus
       | - lang_LH_PostTemp: Posterior temporal lobe
       | - lang_LH_netw: Network average across all five fROIs

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
       | **Valid Values:** fmri-tuckute_2024-GPT2_XL
       | **Example:** "fmri-tuckute_2024-GPT2_XL"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which ROIs to include in the model output.
       | If not provided, responses are generated for all six language network ROIs.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of language network ROI names to include in the output.
       |     ROIs will be returned in the order specified.
       |     **Valid values:** "lang_LH_AntTemp", "lang_LH_IFG", "lang_LH_IFGorb", "lang_LH_MFG", "lang_LH_PostTemp", "lang_LH_netw"
       |     **Example:** ['lang_LH_IFG', 'lang_LH_netw']
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. This model uses linear regression and always runs on CPU, 
       | but this parameter is included for API consistency with other encoding models.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "cpu"

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
     - | **Type:** list[str] or numpy.ndarray
       | **Required:** Yes
       | **Description:** List or array of natural language sentences to encode.
       | 
       | The model accepts sentences of any length, though it was trained on six-word sentences.
       | **Example:**
       | ["Taste that fowl and those fish.",
       | "I'm progressive and you fall right."]
   * - **return_metadata**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to return the encoding model's metadata together with the in silico neural responses.
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
       | **Valid Values:** fmri-tuckute_2024-GPT2_XL
       | **Example:** "fmri-tuckute_2024-GPT2_XL"

Performance
----------

**Metrics:**

* **Optimal Layer**: 22
* **Predictivity R**: 0.38
* **Noise Ceiling R**: 0.56

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-tuckute_2024/model-GPT2_XL/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-tuckute_2024-GPT2_XL",
        selection={
            "roi": ["lang_LH_IFG", "lang_LH_netw"]
        }
    )
    
    # Prepare the stimulus (text/sentences)
    stimulus = ["Taste that fowl and those fish.", 
 "I'm progressive and you fall right."]
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # (n_sentences, n_rois)
    # where:
    # - n_sentences: Number of input sentences.
    # - n_rois: Number of ROIs in the output (6 by default, or fewer if ROI selection is applied).
    #   Default ROI order (when no selection is specified):
    #   - lang_LH_AntTemp: Anterior temporal lobe
    #   - lang_LH_IFG: Inferior frontal gyrus
    #   - lang_LH_IFGorb: Inferior frontal gyrus orbitalis
    #   - lang_LH_MFG: Middle frontal gyrus
    #   - lang_LH_PostTemp: Posterior temporal lobe
    #   - lang_LH_netw: Network average across all five fROIs
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-tuckute_2024-GPT2_XL",
    )
    

References
---------

* Model building code: .berg/models/fmri/tuckute_2024/load_regr_weights_and_predict.py
* Model Paper (Tuckute et al., 2024): https://www.nature.com/articles/s41562-023-01783-7
* Model Repository: https://github.com/gretatuckute/drive_suppress_brains
* GPT2-XL (Radford et al., 2020): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf