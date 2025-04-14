====================
How to Access NEST
====================

NEST is stored in a Google Drive public folder called *neural_encoding_dataset*. To access this folder, follow these steps:

1. Fill out the **NEST Data Access Agreement** form `here <https://forms.gle/ZKxEcjBmdYL6zdrg9>`_, where you will also need to agree to NEST's :doc:`Terms and Conditions </about/terms_and_conditions>`.

2. After completing the form, you will automatically receive the link to the Google Drive folder where NEST is stored.

.. note::
   The *neural_encoding_dataset* folder contains several hundred GBs of data, so downloading might take a while. Depending on your needs, you may choose to download only specific parts of the dataset. This documentation provides a detailed description of NEST's content to help you decide what to download.

Recommended Download Method
---------------------------

We recommend downloading the dataset directly from Google Drive via terminal using **Rclone**. 

**Prerequisites:**

* Before downloading NEST via terminal, add a shortcut of the *neural_encoding_dataset* folder to your Google Drive:
   1. Right-click on the *neural_encoding_dataset* folder
   2. Select *Organise* → *Add shortcut*
   3. This creates a shortcut (without copying or taking space) to a desired path in your Google Drive

**Using Rclone:**

* Rclone is a command-line program to manage files on cloud storage
* For installation and usage instructions, visit the `Rclone website <https://rclone.org/>`_
* For a step-by-step guide on using Rclone with Google Drive, see `this guide <https://noisyneuron.github.io/nyu-hpc/transfer.html>`_


============================
NEST Dataset Structure
============================

Overview of NEST Folder Organization
------------------------------------

NEST is organized in a structured hierarchy on Google Drive, designed to make it easy to locate specific encoding models by their modality, training dataset, and model type.

The main folder structure follows this pattern:

.. code-block:: text

    neural_encoding_simulation_toolkit/
    ├── TERMS_AND_CONDITIONS
    ├── encoding_models/
    │   ├── modality-{modality}/
    │   │   ├── train_dataset-{dataset}/
    │   │   │   └── model-{model_type}/
    │   │   │       ├── encoding_models_accuracy/
    │   │   │       ├── encoding_models_weights/
    │   │   │       └── metadata/
    └── nest_tutorials/

Detailed Structure of Encoding Models
-------------------------------------

The ``encoding_models`` directory contains all trained models organized hierarchically by:

1. **Modality**: The neural recording type (e.g., ``fmri``, ``eeg``)
2. **Training Dataset**: The dataset used to train the model (e.g., ``nsd``, ``things_eeg_2``)
3. **Model Type**: The architecture or approach used (e.g., ``fwrf``, ``vit_b_32``)



Contents of Model Directories
----------------------------

Each model directory contains three important subdirectories:

**encoding_models_accuracy/**

This directory contains visualizations and data showing how well the models perform:

* Performance plots for different subjects, regions, and channels

**encoding_models_weights/**

This directory contains the actual trained model weights that NEST uses to generate predictions:

* Model-specific weight files organized by subject and other relevant parameters

**metadata/**

This directory contains important contextual information about the models:

* Data structures needed to properly interpret model outputs
* Information about channels, ROIs, or other model-specific details
* Subject-specific metadata

The NEST Python package automatically handles access to these files based on your requested parameters, making it easy to use without managing these paths directly.