===================
brainscore_language
===================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - BrainScore
   * - Species
     - Human
   * - Stimuli
     - Text
   * - Model Type
     - BrainScore Language Models (External)
   * - Creator
     - Martin Schrimpf

Description
----------

Access to BrainScore's language models benchmarked against human fMRI recordings from the Pereira 2018 study.

**Neural data.** Benchmark uses fMRI from 9 subjects reading 384 factual sentences:

- Stimuli: Wikipedia-style sentences (7-18 words) across 24 semantic topics (professions, instruments, animals, etc.)

- Coverage: 12,155 voxels pooled (or ~1,350 per subject) from language network

- Output: Predicted BOLD signal (z-scored per voxel)

**Workflow.** For each model and optional subject:

1. Load language model (GPT-2, GPT-Neo, etc.)

2. Load Pereira2018_384sentences benchmark

3. Filter to single subject's voxels

4. Extract model representations for benchmark sentences

5. Train PLS regression: representations → BOLD responses (~few min, cached)

6. Predict BOLD responses for your sentences using cached regression

**Usage.** Use `berg.list_models(expand_brainscore_language=True)` to see all models.
Model IDs: `"brainscore-language-{model_name}"` (e.g., `"brainscore-language-gpt2"`).

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``str or list[str]``
   * - Description
     - A sentence or list of sentences to encode
   * - Constraints
     - * BrainScore handles tokenization internally

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``['n_sentences', 'n_voxels']``
   * - Description
     - Predicted BOLD voxel responses for each sentence

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - n_sentences
     - Number of input sentences
   * - n_voxels
     - ~1,350 voxel per subject

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
       | **Description:** Model identifier. Format: "brainscore_language-{model_name}"
       | Example: "brainscore_language-gpt2"
       | Available models are discovered dynamically from the BrainScore language
       | model registry. Use berg.list_models(expand_brainscore_language=True) to
       | see all available models.
       | **Example:** "brainscore_language-gpt2"
   * - **subject**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Subject identifier for single-subject predictions.
       | 
       | Each subject contributes language-selective voxels from left hemisphere language network:
       |   • Subject 018: 1,358 voxels
       |   • Subject 199: 1,358 voxels
       |   • Subject 288: 1,341 voxels
       |   • Subject 289: 1,356 voxels
       |   • Subject 296: 1,323 voxels
       |   • Subject 343: 1,355 voxels
       |   • Subject 366: 1,355 voxels
       |   • Subject 407: 1,352 voxels
       |   • Subject 426: 1,357 voxels
       | 
       | Separate regressions are trained and cached per subject.
       | **Valid Values:** "018", "199", "288", "289", "296", "343", "366", "407", "426"
       | **Example:** "018"
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

   * - **stimulus**
     - | **Type:** str or list[str]
       | **Required:** Yes
       | **Description:** Input sentences for neural response prediction. Accepts:
       | 
       | 1. Single sentence (str):
       |    Example: "The cat sat on the mat."
       | 
       | 2. List of sentences (list[str]):
       |    Example: ["Sentence one.", "Sentence two."]
       | 
       | Single strings are coerced to a list internally.
       | BrainScore handles tokenization and feature extraction.
       | **Example:** "The cat sat on the mat."
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to print progress messages during encoding

Performance
----------

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "brainscore_language",
        subject="018",
    )
    
    # Prepare the stimulus (text/sentences)
    stimulus = The cat sat on the mat.
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # ['n_sentences', 'n_voxels']
    # where:
    # - n_sentences: Number of input sentences
    # - n_voxels: ~1,350 voxel per subject
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    

References
---------

* BrainScore Website: https://www.brain-score.org/
* BrainScore Language Repository: https://github.com/brain-score/language
* BrainScore Paper (Schrimpf et al., 2018): https://www.biorxiv.org/content/10.1101/407007v1
* Pereira 2018 Paper: https://doi.org/10.1038/s41467-018-03068-4