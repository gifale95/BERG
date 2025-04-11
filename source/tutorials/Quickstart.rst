Quickstart
=========================================

This tutorial will walk you through the fundamental functionality of NEST.

Initialization
-----------

First, import the NEST package:

.. code-block:: python

    from nest import NEST
    
    # Initialize NEST with the path to the root directory
    nest = NEST(nest_dir="neural_encoding_simulation_toolkit")


Exploring Available Models
--------------------------

You can list all available models:

.. code-block:: python

    # List all available models and their versions
    available_models = nest.list_models()
    print(f"Available models: {available_models}")
    
    # See what modalities are available
    catalog = nest.get_model_catalog(print_format=True)
    print(f"Model Catalog as Dict: {catalog}")


Getting Model Information
------------------------

The ``describe`` function is the key to understanding which parameters each function takes. It provides comprehensive information about the model, including required parameters for both ``get_encoding_model()`` and ``encode()`` functions.

There are two ways to get detailed information about a model:

1. Using the ``describe`` method on a model ID:

.. code-block:: python

    # Get comprehensive model information
    model_info = nest.describe("fmri_nsd_fwrf")

This will output detailed information about the model, including the required parameters:

.. code-block:: text

    ================================================================================
    🧠 Model: fmri_nsd_fwrf
    ================================================================================

    Modality: fmri
    Dataset: NSD
    Features: feature-weighted receptive fields (fwrf)
    Repeats: single
    Subject level: True

    📋 Description:
    This model generates in silico fMRI responses to visual stimuli using feature-
    weighted receptive fields (fwrf)...
    
    ... (shortened for view)

    📌 Parameters for encode():

    • stimulus (numpy.ndarray, required)
      ↳ A batch of RGB images to be encoded. Images should be in integer format with
        values in the range [0, 255], and square dimensions (e.g. 224x224).
      ↳ Example: An array of shape [100, 3, 224, 224] representing 100 RGB images.

    • device (str, optional, default='auto')
      ↳ Device to run the model on. 'auto' will use CUDA if available, otherwise
        CPU.
      ↳ Valid values: ['cpu', 'cuda', 'auto']
      ↳ Example: auto

    📌 Parameters for get_encoding_model():

    • subject (int, required)
      ↳ Subject ID from the NSD dataset (1-8)
      ↳ Valid values: [1, 2, 3, 4, 5, 6, 7, 8]
      ↳ Example: 1

    • roi (str, required)
      ↳ Region of Interest (ROI) for voxel prediction. Early visual areas (V1-V3),
        category-selective regions (EBA, FFA, etc.), or composite regions (lateral,
        ventral).
      ↳ Valid values: 'V1', 'V2', 'V3', 'hV4', 'EBA', 'FBA-2', 'OFA', 'FFA-1', 'FFA-2', 'PPA', 'RSC', 'OPA', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'early', 'midventral', 'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral'
      ↳ Example: V1

    • nest_dir (str, optional)
      ↳ Root directory of the NEST repository (optional if default paths are set)
      ↳ Example: ./

    ... (shortened for view)

2. Using the ``describe`` method on an instantiated model:

.. code-block:: python

    # Load Encoding Model
    fwrf_model = nest.get_encoding_model("fmri_nsd_fwrf", 
                                         subject=1, 
                                         roi="V1")
    
    # Get model description
    fwrf_model.describe()

Both methods return the same comprehensive information. Always refer to the Parameters section to understand what inputs each function requires.

Example: Working with the feature-weighted receptive field (fwRF) Model
-----------------------

This is an example on how to use the fwRF model with NEST. For more information on this model, please see the :doc:`Model Overview </models/overview>`.

.. code-block:: python

    # Load the fMRI encoding model
    fwrf_model = nest.get_encoding_model("fmri_nsd_fwrf", 
                                         subject=1, 
                                         roi="V1",
                                         device="cpu")

    # Assume images is a numpy array with shape (batch_size, 3, height, width)
    # For example: (100, 3, 227, 227) for 100 RGB images
    
    # Generate fMRI responses
    fwrf_silico = nest.encode(fwrf_model, images)
    
    # To get both responses and metadata
    fwrf_silico, fwrf_metadata = nest.encode(fwrf_model, images, return_metadata=True)
    
    # Just get the metadata of the model
    metadata = fwrf_model.get_metadata()

The output shape for the fMRI model will be `(batch_size, n_voxels)` where `n_voxels` depends on the selected ROI.

Always refer to the `describe` method to understand the specific parameters and requirements of each model type before using it.