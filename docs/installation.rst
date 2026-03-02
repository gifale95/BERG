Installation
============
This guide covers how to install the Brain Encoding Response Generator (BERG) `Python package <https://github.com/gifale95/BERG>`_ and its dependencies.

Install BERG
------------
Install BERG from GitHub using pip:

.. code-block:: bash

   pip install -U git+https://github.com/gifale95/BERG.git

Install BrainScore Models (Optional)
-----------------------------
BERG integrates with `BrainScore <https://www.brain-score.org>`_, giving you access to 440+ vision models scored against macaque electrophysiology recordings (V1, V2, V4, IT), as well as GPT-family language models scored against human fMRI data.

.. note::

   BrainScore requires **Python 3.11** specifically. If you are on a different Python version, the rest of BERG will work normally — only BrainScore models will be unavailable.

To install BERG with BrainScore support, run both of the following commands:

.. code-block:: bash

   pip install -U git+https://github.com/gifale95/BERG.git
   pip install berg[brainscore]

We recommend creating a dedicated conda environment for this:

.. code-block:: bash

   conda create -n berg_brainscore python=3.11
   conda activate berg_brainscore
   pip install -U git+https://github.com/gifale95/BERG.git
   pip install berg[brainscore]

Verify Installation
-------------------
To verify that BERG is correctly installed, run:

.. code-block:: python

   from berg import BERG

   # Initialize BERG with default data directory
   berg = BERG("/path/to/brain-encoding-response-generator")

   # List available models
   models = berg.list_models()
   print(f"Available models: {models}")

You should see a list of available models in the output.

Quick Example
-------------
Here's a simple example of how to generate in silico neural responses using BERG:

.. code-block:: python

   from berg import BERG
   import numpy as np

   # Initialize BERG
   berg = BERG("/path/to/brain-encoding-response-generator")

   # Get an encoding model
   model = berg.get_encoding_model("fmri-nsd-fwrf", subject=1, selection={"roi": "V1"})

   # Generate responses to stimuli
   stimuli = np.random.randint(0, 255, (10, 3, 224, 224), dtype=np.uint8)
   responses = berg.encode(model, stimuli)
   print(f"Generated responses shape: {responses.shape}")

For more detailed examples and usage instructions, please see the :doc:`Quickstart tutorial </tutorials/Quickstart>`.
For a tutorial on BrainScore models specifically, please see the :doc:`BrainScore tutorial </tutorials/BrainScore>`.