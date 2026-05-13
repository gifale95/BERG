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

**Installation.** BrainScore models require a separate installation step and Python 3.11:

*> pip install -U git+https://github.com/gifale95/BERG.git*
*> pip install berg[brainscore]*

For available models and scores, see the `BrainScore language leaderboard <https://www.brain-score.org/language/leaderboard/>`_.

**What is BrainScore?**
BrainScore is an open benchmarking platform where researchers submit computational models and evaluate how well those models
predict neural responses to visual of language stimuli against benchmarks. A benchmark consists of a neural dataset on which the
encoding models are trained and evaluated, together with an evaluation protocol. There are many benchmarks available on BrainScore,
each targeting a different brain region, species, or recording modality. BERG supports the use of hundreds of BrainScore language models
to generate in silico fMRI responses to text sentences.

**How it works.**
For each language model submitted to BrainScore, the submitter has specified which internal layer of the model best predicts neural activity in the human
language network. BERG extracts activations from that layer and trains a Partial Least Squares (PLS) regression (i.e., a brain encoding model), which finds
a low-dimensional mapping from high-dimensional model representations onto neural responses. This regression is trained on the Pereira et al. (2018) benchmark:
human fMRI recordings collected while participants read 384 factual sentences. Once trained, the regression is cached to disk and reused for all future
predictions, so the training step (~few minutes) only happens once per model and subject combination.

**Neural data.**
The Pereira et al. (2018) benchmark uses fMRI recordings from 9 subjects reading 384 Wikipedia-style sentences (7 to 18 words) spanning 24 semantic topics
(professions, instruments, animals, etc.). Recordings cover language-selective voxels from the left hemisphere language network, yielding approximately 1,350 voxels
per subject (12,155 voxels pooled across all subjects). Neural responses are z-scored per voxel. Each subject has a separately trained and cached regression.

**Workflow.**
The following steps happen automatically under the hood when you call ``get_encoding_model()`` and ``encode()``:

1. Load the language model (e.g. GPT-2, GPT-Neo) via the BrainScore model registry.
2. Run the model on the 384 Pereira benchmark sentences to extract layer activations (the layer is pre-selected by the model submitter on BrainScore).
3. Filter the benchmark fMRI responses to the selected subject's voxels.
4. Train a PLS regression mapping model activations to fMRI voxel responses (~few minutes, only done once).
5. Cache the regression weights to disk in your BERG directory.
6. For your new sentences: extract activations from the same layer, apply the cached regression, and return predicted BOLD responses.

**Usage.**
Use ``berg.list_models(expand_brainscore_language=True)`` to see all available models.
Model IDs follow the format ``"brainscore_language-{model_name}"`` (e.g. ``"brainscore_language-gpt2"``).

A hands-on tutorial demonstrating how to use BrainScore models within BERG is available as a `Colab notebook <https://colab.research.google.com/drive/1B-gRZmdN6ZhxUUgUXgxfTgJc344a8Z17>`_.

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
   * - Dimensions
     - | **n_sentences**: Number of input sentences
       | **n_voxels**: ~1,350 voxel per subject

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
* BERG BrainScore Tutorial (Colab): https://colab.research.google.com/drive/1B-gRZmdN6ZhxUUgUXgxfTgJc344a8Z17