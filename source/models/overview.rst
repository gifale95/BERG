===================
Available Models
===================

This page provides an overview of the brain encoding models currently available in BERG.


Model Naming Convention
----------------------

BERG contains several encoding models, defined by the following model ID naming convention:

``{modality}-{dataset}-{model}``

where

* ``modality``: The neural recording recording modality on which the encoding model was trained.
* ``dataset``: The neural dataset on which the encoding model was trained.
* ``model``: The type of encoding model used.

For example:

- ``fmri-nsd-fwrf``: An fMRI encoding model trained on the NSD using feature-weighted receptive fields.
- ``eeg-things_eeg_2-vit_b_32``: An EEG model trained on the THINGS-EEG2 dataset using the ViT-B/32 visual transformer architecture.


Get Model Information
------------------------

You can get detailed information about any model using:

.. code-block:: python

    from berg import BERG
    
    berg = BERG("path/to/brain-encoding-response-generator")

    # List all available models
    all_models = berg.list_models()
    
    # Get detailed model information
    model_info = berg.describe("fmri-nsd-fwrf")


Available models
----------------------

Following is a list of all available models, grouped by ``modality`` and ``dataset``.

modality-fmri
~~~~~~~~~~

Encoding models trained on neural responses recorded with functional Magnetic Resonance Imaging (fMRI).

.. list-table::
   :header-rows: 1
   :widths: 20 55 20 10 10
   :class: wrap-table

   * - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
   * - :doc:`model_cards/fmri-nsd_fsaverage-vit_b_32`
     - Linear mapping of vision transformer image features onto fMRI responses.
     - Natural Scenes Dataset(surface space)
     - Human
     - Images
   * - :doc:`model_cards/fmri-nsd-fwrf`
     - Feature-weighted receptive fields, convolutional neural networks trained end-to-end to predict fMRI responses from input images.
     - Natural Scenes Dataset (volume space)
     - Human
     - Images


modality-eeg
~~~~~~~~~~~~

Encoding models trained on neural responses recorded with Electroencephalography (EEG).

.. list-table::
   :header-rows: 1
   :widths: 20 55 20 10 10
   :class: wrap-table

   * - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
   * - :doc:`model_cards/eeg-things_eeg_2-vit_b_32`
     - Linear mapping of vision transformer image features onto EEG responses.
     - THINGS EEG2
     - Human
     - Images

.. raw:: html

   <style>
   .wrap-table td {
     white-space: normal !important;
     word-wrap: break-word !important;
   }
   </style>

