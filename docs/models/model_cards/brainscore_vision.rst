=================
brainscore_vision
=================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - ephys
   * - Training Dataset
     - BrainScore
   * - Species
     - Macaque
   * - Stimuli
     - Images
   * - Model Type
     - BrainScore Vision Models (External)
   * - Creator
     - Martin Schrimpf

Description
----------

**Installation.** BrainScore models require a separate installation step and Python 3.11:

*> pip install -U git+https://github.com/gifale95/BERG.git*
*> pip install berg[brainscore]*

For available models and scores, see the `BrainScore vision leaderboard <https://www.brain-score.org/vision/leaderboard/>`_.

**What is BrainScore?**
BrainScore is an open benchmarking platform where researchers submit computational models and evaluate how well those models
predict neural responses to visual of language stimuli against benchmarks. A benchmark consists of a neural dataset on which the
encoding models are trained and evaluated, together with an evaluation protocol. There are many benchmarks available on BrainScore,
each targeting a different brain region, species, or recording modality. BERG supports the use of hundreds of BrainScore vision models
to generate in silico electrophysiology responses to images, for predifined benchmarks.

**How it works.**
For each vision model submitted to BrainScore, the submitter has specified which internal layer of the model best predicts neural activity in
each brain region of interest (ROI). BERG extracts activations from that layer for the selected ROI, and trains a Partial Least Squares (PLS) regression
(i.e., a brain encoding model), which finds a low-dimensional mapping from high-dimensional model representations onto neural responses. This regression
is trained on the benchmark data for the selected ROI (see Neural data below). Once trained, the regression is cached to disk and reused for all future
predictions, so the training step (~few minutes) only happens once per model and ROI combination.

**Neural data.**
BrainScore models predict in silico neural responses for visual ROIs V1, V2, V4, and IT. When used in BERG, the benchmark on which the encoding models are
trained (and, as a consequence, for which these encoding models generate in silico neural responses to visual stimuli) is determined by the selected ROI.
For V1 and V2, BERG uses the Freeman et al. (2013) benchmark (102 and 103 neurons for V1 and V2, respectively), recorded using synthetic texture patches
designed to probe low-level visual processing. For V4 and IT, BERG uses the Majaj et al. (2015) benchmark (88 and 168 neurons for V4 and IT, respectively),
recorded using grayscale images of objects on natural backgrounds, designed to probe mid- and high-level object recognition. Each ROI has a separately
trained and cached regression.

**Workflow.**
The following steps happen automatically under the hood when you call ``get_encoding_model()`` and ``encode()``:

1. Load the vision model (e.g. AlexNet, ResNet, ViT) via the BrainScore model registry.
2. Load the benchmark for the selected ROI. Each ROI determined the benchmark dataset.
3. Extract activations from the model layer pre-selected by the model submitter for that ROI.
4. Train a PLS regression mapping model activations to neural responses (~few minutes, only done once).
5. Cache the regression weights to disk in your BERG directory.
6. For your new images: extract activations from the same layer, apply the cached regression, and return predicted neural responses.

**Usage.**
Use ``berg.list_models(expand_brainscore_vision=True)`` to see all available models.
Model IDs follow the format ``"brainscore_vision-{model_name}"`` (e.g. ``"brainscore_vision-alexnet"``).

A hands-on tutorial demonstrating how to use BrainScore models within BERG is available as a `Colab notebook <https://colab.research.google.com/drive/1B-gRZmdN6ZhxUUgUXgxfTgJc344a8Z17>`_.

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``str, list[str], or np.ndarray``
   * - Description
     - | Input stimuli for neural response prediction. Accepts multiple formats:
       | 
       | 1. Directory path (str): Path to directory containing images
       |     Example: "/path/to/images"
       | 
       | 
       | 2. List of paths (list[str]): List of image file paths
       |     Example: ["img1.jpg", "img2.jpg"]
       | 
       | 
       | 3. Numpy array (np.ndarray): Batch of RGB images
       |     Shape: (batch_size, 3, height, width)
       |     Values: [0, 255] (uint8)
       |     Example: np.array with shape (10, 3, 224, 224)
       | 
       | 
       | For numpy arrays, images are temporarily saved to disk (BrainScore requirement)
       | and automatically cleaned up after prediction.
       | 
       | For file paths, images should be in JPEG or PNG.
       | BrainScore handles preprocessing (resizing, normalization) internally.

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``['batch_size', 'n_units']``
   * - Description
     - Neural responses for specified recording target
   * - Dimensions
     - | **batch_size**: Number of stimuli
       | **n_units**: Number of neural units (varies by model and recording target)

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
       | **Description:** Model identifier. Format: "brainscore_{model_name}"
       | Example: "brainscore_vonegrcnn_47e"
       | Use berg.list_models(expand_brainscore=True) to see all available models.
       | **Example:** "brainscore_vonegrcnn_47e"
   * - **selection**
     - | **Type:** dict
       | **Required:** Yes
       | **Description:** Specifies recording target (brain region).
       | ROI selection is REQUIRED for BrainScore models.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** str
       |     **Description:** Brain region to record from. Each region uses a different benchmark:
       |       • V1: 102 neurons (FreemanZiemba2013.V1.public-pls)
       |       • V2: 103 neurons (FreemanZiemba2013.V2.public-pls)
       |       • V4: 88 neurons (MajajHong2015.V4.public-pls)
       |       • IT: 168 neurons (MajajHong2015.IT.public-pls)
       |     
       |     Regression weights are trained on benchmark data and cached after first use.
       |     **Valid values:** "V1", "V2", "V4", "IT"
       |     **Example:** IT
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
     - | **Type:** str, list[str], or np.ndarray
       | **Required:** Yes
       | **Description:** Input stimuli for neural response prediction. Accepts multiple formats:
       | 
       | 1. Directory path (str): Path to directory containing images
       |    Example: "/path/to/images"
       | 2. List of paths (list[str]): List of image file paths
       |    Example: ["img1.jpg", "img2.jpg"]
       | 3. Numpy array (np.ndarray): Batch of RGB images
       |    Shape: (batch_size, 3, height, width)
       |    Values: [0, 255] (uint8)
       |    Example: np.array with shape (10, 3, 224, 224)
       | 
       | For numpy arrays, images are temporarily saved to disk (BrainScore requirement)
       | and automatically cleaned up after prediction.
       | 
       | For file paths, images should be standard formats (JPEG, PNG, etc.).
       | BrainScore handles preprocessing (resizing, normalization) internally.
       | **Example:** "/path/to/images"
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show progress bar during encoding

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
        "brainscore_vision",
        selection={
            "roi": "IT"
        }
    )
    
    # Prepare the stimulus (text/sentences)
    stimulus = /path/to/images
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # ['batch_size', 'n_units']
    # where:
    # - n_units: Number of neural units (varies by model and recording target)
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    

References
---------

* BrainScore Website: https://www.brain-score.org/
* BrainScore Vision Models Repository: https://github.com/brain-score/vision
* BrainScore Paper (Schrimpf et al., 2018): https://www.biorxiv.org/content/10.1101/407007v1
* Freeman et al., 2013 Paper: https://doi.org/10.1038/nn.3402
* Majaj et al., 2015 Paper: https://doi.org/10.1523/JNEUROSCI.5181-14.2015
* BrainScore Tutorial (Colab): https://colab.research.google.com/drive/1B-gRZmdN6ZhxUUgUXgxfTgJc344a8Z17