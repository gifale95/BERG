=====================
ecog-zada2025-gpt2_xl
=====================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - ECoG
   * - Training Dataset
     - Zadal et al. (2025)
   * - Species
     - Human
   * - Stimuli
     - Text (natural speech transcript)
   * - Model Type
     - GPT2-XL
   * - Creator
     - Zaid Zada

Description
----------

This encoding model consists of a linear mapping (through ridge regression) of GPT-2 XL
(Radford et al., 2019) contextual word embeddings onto intracranial electrocorticographic (ECoG)
high-gamma band activity during natural speech comprehension. Prior to mapping onto neural responses,
word embeddings are extracted from layer 24 (1,600-dimensional) of GPT-2 XL, a 48-layer autoregressive
language model with 1.5 billion parameters. When a word is split into multiple sub-word tokens by the
GPT-2 tokenizer, the token embeddings are averaged to produce a single word-level embedding.

The encoding models were trained on the Podcast ECoG dataset (Zada et al., Scientific Data 2025),
consisting of 9 human participants with a total of 1,330 intracranial electrodes listening to a
30-minute audio podcast (~5,100 words).

**Neural data.** The preprocessed high-gamma band power (70–150 Hz) was extracted following the
dataset's official pipeline. The continuous signal was downsampled to 32 Hz, then epoched from
-2.0 s to +2.0 s relative to each word onset, producing 129 time lags per electrode. Each lag
represents the temporal offset from word onset (e.g., lag +0.3 s captures neural activity 300 ms
after the word began playing). No noise ceiling is available for this dataset because each word
is heard exactly once, precluding estimation of trial-to-trial variability.

**Model training partition.** All available word epochs (~5,100 per subject) were used for training.
Independent encoding models were trained for each subject. Features and neural data were standardized
prior to fitting. The regularization hyperparameter was selected via 5-fold inner cross-validation
from 10 log-spaced alpha values between 10^1 and 10^10.

**Model evaluation.** Encoding accuracy was evaluated using 2-fold cross-validation matching the
paper's methodology: models were trained on each half of the podcast and tested on the other, and
the two fold correlations (Pearson's r per electrode × lag) were averaged. Encoding performance varies substantially 
across subjects due to differences in electrode placement. Subjects 01, 06, and 09 show the strongest encoding
(best electrodes r ≈ 0.30–0.35), while subjects 05 and 07 show near-chance performance.

**Output.** Each encoding model predicts time-resolved high-gamma responses for all electrodes
(or user-specified subsets) across 129 time lags for each input word.

Metadata
--------

**ecog**

    **subject_id** : ``str`` - Subject identifier

    **n_electrodes** : ``int`` - Number of electrodes (varies by subject)

    **n_lags** : ``int`` - Number of time lags (129)

    **sfreq** : ``float`` - Sampling frequency (32 Hz)

    **tmin** : ``float`` - Epoch start (-2.0 s)

    **tmax** : ``float`` - Epoch end (+2.0 s)

    **times** : ``(129,)`` - Time points in seconds

    **ch_names** : ``(n_electrodes,)`` - Electrode names

    **ch_coords** : ``(n_electrodes,3)`` - Electrode MNI coordinates
**encoding_model**

    **epoch_selection** : ``(n_epochs,)`` - Word indices of surviving epochs

    **correlation_results** : ``(n_electrodes, 129)`` - 2-fold CV encoding accuracy (Pearson's r)

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``list[str]``
   * - Description
     - The input should be a list of words (strings). Context is built from all preceding words in the list.
   * - Constraints
     - * Each element should be a single word (string).
       * Context is built from the full preceding word sequence.
   * - Example
     - ``['Once', 'upon', 'a', 'time', 'there', 'was', 'a', 'king']``

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``['n_words', 'n_electrodes', 'n_lags']``
   * - Description
     - The output is a 3D array containing in silico high-gamma ECoG responses.
   * - Dimensions
     - | **n_words**: Number of words in the input.
       | **n_electrodes**: Number of electrodes (subject-dependent, or filtered by selection).
       | **n_lags**: Number of time lags relative to word onset.

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
       | **Valid Values:** ecog-zada2025-gpt2_xl
       | **Example:** "ecog-zada2025-gpt2_xl"
   * - **subject**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Subject ID from the Podcast ECoG dataset.
       | **Valid Values:** "01", "02", "03", "04", "05", "06", "07", "08", "09"
       | **Example:** "03"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific electrodes and/or lags (timepoints). If not provided,
       | responses are generated for all electrodes and time lags.
       | 
       | **Properties:**
       | 
       | **electrode_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which electrodes to include.
       |     Length must match the number of electrodes for the selected subject:
       |       - Subject 01: 99 electrodes
       |       - Subject 02: 90 electrodes
       |       - Subject 03: 235 electrodes
       |       - Subject 04: 143 electrodes
       |       - Subject 05: 159 electrodes
       |       - Subject 06: 166 electrodes
       |       - Subject 07: 116 electrodes
       |       - Subject 08: 72 electrodes
       |       - Subject 09: 188 electrodes
       |     **Example:** [0, 0, 1, 1, 0, '...', 1]
       | 
       | **lags**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which time lags (timepoints) to include.
       |     Must have exactly 129 elements.
       |     Each position set to 1 indicates that time lag should be included.
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
     - | **Type:** list[str]
       | **Required:** Yes
       | **Description:** A list of words (strings) to encode. Context is built from the full preceding word sequence.
       | **Example:** "["The", "quick", "brown", "fox", "jumped"]"
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
       | **Valid Values:** ecog-zada2025-gpt2_xl
       | **Example:** "ecog-zada2025-gpt2_xl"
   * - **subject**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Subject ID from the Podcast ECoG dataset.
       | **Valid Values:** "01", "02", "03", "04", "05", "06", "07", "08", "09"
       | **Example:** "03"

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-ecog/train_dataset-zada2025/model-gpt2_xl/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "ecog-zada2025-gpt2_xl",
        subject="03",
        selection={
            "electrode_index": [0, 0, 1, 1, 0, '...', 1],
            "lags": [0, 0, '...', 1, 1, 0]
        }
    )
    
    # Prepare the stimulus (text/sentences)
    stimulus = ["The", "quick", "brown", "fox", "jumped"]
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # ['n_words', 'n_electrodes', 'n_lags']
    # where:
    # - n_words: Number of words in the input.
    # - n_electrodes: Number of electrodes (subject-dependent, or filtered by selection).
    # - n_lags: Number of time lags relative to word onset.
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "ecog-zada2025-gpt2_xl",
        subject="03"
    )
    

References
---------

* Model building code: https://github.com/gifale95/BERG/tree/main/berg_creation_code/02_train_encoding_models/train_dataset-zada2025/train_encoding.py
* Podcast ECoG Paper (Zada et al., 2025): https://doi.org/10.1038/s41597-025-05462-2
* Podcast ECoG Data (Zada et al., 2025): https://openneuro.org/datasets/ds005574
* GPT-2 (Radford et al., 2019): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf