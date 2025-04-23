Welcome to NEST's Documentation
============================

.. .. image:: static/nest_logo.png
   :width: 300
   :alt: NEST Logo

**Neural Encoding Simulation Toolkit (NEST)** is a resource consisting of multiple pre-trained encoding models of the brain and an accompanying `Python package <https://github.com/gifale95/NEST>`_ to generate accurate in silico neural responses to arbitrary stimuli with just a few lines of code.

In silico neural responses generated from encoding models increasingly resemble in vivo responses recorded from real brains, enabling the novel research paradigm of in silico neuroscience. These simulated responses are fast and cost-effective to generate, allowing researchers to explore and test scientific hypotheses across vastly larger solution spaces than possible in vivo. Novel findings from large-scale in silico experimentation can then be validated through targeted small-scale in vivo data collection, in this way optimizing research resources. Thus, in silico neuroscience scales beyond what is possible with in vivo data, and democratizes research across groups with diverse data collection infrastructure and resources.

NEST includes a growing, well-documented library of encoding models trained on different neural data acquisition modalities, datasets, subjects, stimulation types, and brain areas, offering broad versatility for addressing a wide range of research questions through in silico neuroscience.

If anything in this manual is not clear, or if you think some information is missing, please get in touch with us at brain.nest.contact@gmail.com.



----

Contents
-------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   data_storage
   contribution

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   tutorials/Quickstart
   tutorials/Adding_New_Models_to_Nest

.. toctree::
   :maxdepth: 2
   :caption: Models
   
   models/overview
   models/fmri-nsd-fwrf.rst
   models/eeg-things_eeg_2-vit_b_32.rst

.. toctree::
   :maxdepth: 1
   :caption: About
   
   about/terms_and_conditions
   about/citation

----

Quick Start
----------

Install NEST using pip:

.. code-block:: bash

   pip install -U git+https://github.com/gifale95/NEST.git

Generate in silico fMRI responses:

.. code-block:: python

   from nest import NEST
   import numpy as np
   
   # Initialize NEST
   nest = NEST("/path/to/neural_encoding_simulation_toolkit")
   
   # Get an encoding model
   model = nest.get_encoding_model("fmri-nsd-fwrf", subject=1, selection={"roi": "V1"})
   
   # Generate responses to stimuli
   stimuli = np.random.randint(0, 255, (10, 3, 224, 224), dtype=np.uint8)
   responses = nest.encode(model, stimuli)
   
   print(f"Generated responses shape: {responses.shape}")
