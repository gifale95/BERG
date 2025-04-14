Installation
============

This guide covers how to install the Neural Encoding Simulation Toolkit (NEST) `Python package <https://github.com/gifale95/NEST>`_ and its dependencies.


Installing NEST from PyPI
--------------

The simplest way to install NEST is via pip:

.. code-block:: bash

   pip install -U git+https://github.com/gifale95/NEST.git


Verifying Installation
---------------------

To verify that NEST is correctly installed, run:

.. code-block:: python

   from nest import NEST
   
   # Initialize NEST with default data directory
   nest = NEST("/path/to/neural_encoding_simulation_toolkit")
   
   # List available models
   models = nest.list_models()
   print(f"Available models: {models}")

You should see a list of available models in the output.


Quick Example
------------
Here's a simple example of how to generate in silico neural responses using NEST:

.. code-block:: python

   from nest import NEST
   import numpy as np
   
   # Initialize NEST
   nest = NEST("/path/to/neural_encoding_simulation_toolkit")
   
   # Get an encoding model
   model = nest.get_encoding_model("fmri-nsd-fwrf", subject=1, roi="V1")
   
   # Generate responses to stimuli
   stimuli = np.random.randint(0, 255, (10, 3, 224, 224), dtype=np.uint8)
   responses = nest.encode(model, stimuli)
   
   print(f"Generated responses shape: {responses.shape}")

For more detailed examples and usage instructions, please see the :doc:`Quickstart tutorial </tutorials/Quickstart>`.


