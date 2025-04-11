===================
Available Models
===================

NEST provides a collection of pre-trained neural encoding models for different brain recording modalities. This page provides an overview of all currently available models.


fMRI Models
----------

fMRI (functional Magnetic Resonance Imaging) models in NEST generate simulated brain activity patterns in response to visual stimuli, with responses mapped to specific brain regions (ROIs).

.. list-table::
   :header-rows: 1
   :widths: 20 55 15
   :class: wrap-table

   * - Model ID
     - Description
     - Subjects
   * - :doc:`model_cards/fmri_nsd_fwrf`
     - Feature-weighted receptive field model trained on the Natural 
       Scenes Dataset (NSD). Predicts fMRI responses across multiple 
       visual regions of interest.
     - 1-8

EEG Models
----------

EEG (Electroencephalography) models in NEST generate simulated electrical brain activity patterns in response to visual stimuli, providing high temporal resolution across multiple electrodes.

.. list-table::
   :header-rows: 1
   :widths: 20 55 15
   :class: wrap-table

   * - Model ID
     - Description
     - Subjects
   * - :doc:`model_cards/eeg_things_eeg_2_vit_b_32`
     - Vision Transformer (ViT-B/32) model trained on the THINGS-EEG-2 
       dataset. Predicts EEG responses across all channels and time 
       points.
     - 1-4

.. raw:: html

   <style>
   .wrap-table td {
     white-space: normal !important;
     word-wrap: break-word !important;
   }
   </style>

Model Naming Convention
----------------------

NEST models follow a consistent naming convention:

``{modality}_{dataset}_{model_architecture}``

For example:

- ``fmri_nsd_fwrf``: An fMRI model trained on the NSD dataset using feature-weighted receptive fields
- ``eeg_things_eeg_2_vit_b_32``: An EEG model trained on the THINGS-EEG-2 dataset using ViT-B/32 architecture

Getting Model Information
------------------------

You can get detailed information about any model using:

.. code-block:: python

    from nest import NEST
    
    nest = NEST()

    # List all available models
    all_models = nest.list_models()
    
    # Get detailed model information
    model_info = nest.describe("fmri_nsd_fwrf")
