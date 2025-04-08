Installation
============

This guide covers how to install the Neural Encoding Simulation Toolkit (NEST) and its dependencies.


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
   nest = NEST()
   
   # List available models
   models = nest.list_models()
   print(f"Available models: {models}")

You should see a list of available models in the output.