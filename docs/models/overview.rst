===================
Available Models
===================
This page provides an overview of the brain encoding models currently available in BERG.

Model Naming Convention
----------------------
BERG contains several encoding models, defined by the following model ID naming convention:

``{modality}-{dataset}-{model}``

where

* ``modality``: The neural recording modality on which the encoding model was trained.
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
Following is a list of all available models, grouped by ``modality``. The ✅ icon indicates the best model for each ``dataset``.

modality-fmri
~~~~~~~~~~
Encoding models trained on neural responses recorded with functional Magnetic Resonance Imaging (fMRI).

.. list-table::
   :header-rows: 1
   :widths: 3 60 40 20 10 10 10
   :class: wrap-table

   * - Best model
     - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
     - Encoding accuracy
   * - ✅
     - :doc:`model_cards/fmri-nsd_fsaverage-huze`
     - Mapping of vision transformer image features onto fMRI responses.
     - Natural Scenes Dataset (surface space)
     - Human
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-fmri/train_dataset-nsd_fsaverage/model-huze/encoding_models_accuracy/>`_
   * - 
     - :doc:`model_cards/fmri-nsd_fsaverage-vit_b_32`
     - Linear mapping of vision transformer image features onto fMRI responses.
     - Natural Scenes Dataset (surface space)
     - Human
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-fmri/train_dataset-nsd_fsaverage/model-vit_b_32/encoding_models_accuracy/>`_
   * - ✅
     - :doc:`model_cards/fmri-nsd-fwrf`
     - Feature-weighted receptive fields, convolutional neural networks trained end-to-end to predict fMRI responses from input images.
     - Natural Scenes Dataset (volume space)
     - Human
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-fmri/train_dataset-nsd/model-fwrf/encoding_models_accuracy/>`_
   * - 
     - :doc:`model_cards/fmri-things_fmri_1-vit_b_32`
     - Linear mapping of vision transformer image features onto whole-brain fMRI responses.
     - THINGS fMRI1
     - Human
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-fmri/train_dataset-things_fmri_1/model-vit_b_32/encoding_models_accuracy/>`_

modality-eeg
~~~~~~~~~~~~
Encoding models trained on neural responses recorded with Electroencephalography (EEG).

.. list-table::
   :header-rows: 1
   :widths: 3 60 40 20 10 10 10
   :class: wrap-table

   * - Best model
     - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
     - Encoding accuracy
   * - ✅
     - :doc:`model_cards/eeg-things_eeg_2-vit_b_32`
     - Linear mapping of vision transformer image features onto EEG responses.
     - THINGS EEG2
     - Human
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-eeg/train_dataset-things_eeg_2/model-vit_b_32/encoding_models_accuracy/>`_

modality-meg
~~~~~~~~~~~~
Encoding models trained on neural responses recorded with Magnetoencephalography (MEG).

.. list-table::
   :header-rows: 1
   :widths: 3 60 40 20 10 10 10
   :class: wrap-table

   * - Best model
     - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
     - Encoding accuracy
   * - ✅
     - :doc:`model_cards/meg-things_meg_1-vit_b_32`
     - Linear mapping of vision transformer image features onto time-resolved whole-brain MEG responses.
     - THINGS MEG1
     - Human
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-meg/train_dataset-things_meg_1/model-vit_b_32/encoding_models_accuracy/>`_

modality-utah_array
~~~~~~~~~~~~~~~~~~~
Encoding models trained on neural responses recorded with Utah arrays (intracortical electrophysiology).

.. list-table::
   :header-rows: 1
   :widths: 3 60 40 20 10 10 10
   :class: wrap-table

   * - Best model
     - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
     - Encoding accuracy
   * - ✅
     - :doc:`model_cards/utah_array-tvsd-vit_b_32`
     - Linear mapping of vision transformer image features onto time-resolved intracortical spiking activity.
     - THINGS Ventral Stream Spiking Dataset (TVSD)
     - Macaque
     - Images
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-utah_array/train_dataset-tvsd/model-vit_b_32/encoding_models_accuracy/>`_

modality-calcium_2p
~~~~~~~~~~~~~~~~~~~
Encoding models trained on neural responses recorded with two-photon calcium imaging.

.. list-table::
   :header-rows: 1
   :widths: 3 60 40 20 10 10 10
   :class: wrap-table

   * - Best model
     - Model ID
     - Description
     - Training dataset
     - Species
     - Stimuli
     - Encoding accuracy
   * - ✅
     - :doc:`model_cards/calcium_2p-wang_2025-3DCNN`
     - Foundation model of mouse visual cortex.
     - `Wang et al., 2025 <https://doi.org/10.1038/s41586-025-08829-y>`_
     - Mouse
     - Videos
     - `Accuracy plots <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html#encoding_models/modality-calcium_2p/train_dataset-wang_2025/model-3DCNN/encoding_models_accuracy/>`_

.. raw:: html

   <style>
   .wrap-table td {
       white-space: normal !important;
       word-wrap: break-word !important;
   }
   </style>