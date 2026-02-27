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

*> pip install berg[brainscore]*

For available models and scores, see the `BrainScore vision leaderboard <https://www.brain-score.org/vision/leaderboard/>`_.

Access to BrainScore's 440+ vision models benchmarked against macaque electrophysiology recordings.

**Neural data.** Benchmarks use electrophysiology recordings from macaque visual cortex:

- V1/V2: Freeman & Ziemba 2013 (102 & 103 neurons, synthetic texture patches)

- V4/IT: Majaj & Hong 2015 (88 & 168 neurons, grayscale objects on natural backgrounds)

- Output: Predicted firing rates (spikes/second)

**Workflow.** For each model and region:

1. Load vision model (AlexNet, ResNet, ViT, etc.)

2. Load benchmark for selected region (each ROI uses different benchmark)

3. Train PLS regression: model activations → neural responses (~3 min, cached)

4. Predict neural responses for your images using cached regression

**Usage.** Use `berg.list_models(expand_brainscore_vision=True)` to see all models.
Model IDs: `"brainscore-vision-{model_name}"` (e.g., `"brainscore-vision-alexnet"`).

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

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli
   * - n_units
     - Number of neural units (varies by model and recording target)

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

    # Or use a numpy array
    stimulus = np.random.randint(0, 255, (100, 3, 256, 256))
    
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
* Freeman & Ziemba 2013 Paper: https://pubmed.ncbi.nlm.nih.gov/23685719/
* Majaj & Hong 2015 Paper: https://pubmed.ncbi.nlm.nih.gov/26424887/
* BrainScore Tutorial (Colab): https://colab.research.google.com/drive/1B-gRZmdN6ZhxUUgUXgxfTgJc344a8Z17