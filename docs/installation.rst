Installation
============
This guide covers how to install the Brain Encoding Response Generator (BERG) `Python package <https://github.com/gifale95/BERG>`_ and its dependencies.

Install BERG
------------

.. note::

   BERG requires **Python ≥ 3.11**.

Recommended (includes TRIBEv2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -U "berg[full] @ git+https://github.com/gifale95/BERG.git"

We recommend creating a dedicated conda environment:

.. code-block:: bash

   conda create -n berg python=3.11
   conda activate berg
   pip install -U "berg[full] @ git+https://github.com/gifale95/BERG.git"

BrainScore (optional, replaces TRIBEv2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BERG integrates with `BrainScore <https://www.brain-score.org>`_, giving you access to hundreds of vision models scored against macaque electrophysiology recordings (V1, V2, V4, IT), as well as GPT-family language models scored against human fMRI data.

.. code-block:: bash

   pip install -U "berg[brainscore] @ git+https://github.com/gifale95/BERG.git"

We recommend creating a dedicated conda environment for this:

.. code-block:: bash

   conda create -n berg_brainscore python=3.11
   conda activate berg_brainscore
   pip install -U "berg[brainscore] @ git+https://github.com/gifale95/BERG.git"

.. note::

   TRIBEv2 and BrainScore cannot be installed together due to a NumPy version conflict. Choose one depending on your use case. BrainScore requires **Python = 3.11** specifically.

Switching between versions
^^^^^^^^^^^^^^^^^^^^^^^^^^
If you already have one version installed and want to switch, uninstall the conflicting packages first:

**Full → BrainScore:**

.. code-block:: bash

   pip uninstall tribev2 neuralset neuraltrain -y
   pip install -e ".[brainscore]"

**BrainScore → Full:**

.. code-block:: bash

   pip uninstall brainscore-vision brainscore-language -y
   pip install -e ".[full]"

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