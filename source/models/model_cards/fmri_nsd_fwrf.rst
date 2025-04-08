==============
fmri_nsd_fwrf
==============

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fmri
   * - Dataset
     - NSD
   * - Features
     - feature-weighted receptive fields (fwrf)
   * - Repeats
     - single
   * - Subject-specific
     - True

Description
----------

This model generates in silico fMRI responses to visual stimuli using feature-weighted receptive fields (fwrf).
It was trained on the Natural Scenes Dataset (NSD), a large-scale 7T fMRI dataset of subjects viewing natural images.
The model extracts visual features using a convolutional neural network and maps these features to brain activity 
patterns across multiple visual regions of interest (ROIs).

The model takes as input a batch of RGB images in the shape [batch_size, 3, height, width], with pixel values ranging from 0 to 255 and square dimensions (e.g., 224×224).

Input
-----

**Type**: ``numpy.ndarray``  
**Shape**: ``[batch_size, 3, height, width]``  
**Description**: The input should be a batch of RGB images.

**Constraints:**

* Image values should be integers in range [0, 255]
* Image dimensions (height, width) should be equal (square)
* Minimum recommended image size: 224x224 pixels

Output
------

**Type**: ``numpy.ndarray``  
**Shape**: ``[batch_size, n_voxels]``  
**Description**:  
The output is a 2D array containing predicted fMRI responses.
The second dimension (n_voxels) corresponds to the number of voxels in the selected ROI,
which varies by ROI and subject. For subject 1, the number of voxels per ROI is as follows:

* V1: 1350
* V2: 1433
* V3: 1187
* hV4: 687
* EBA: 2971
* FBA-2: 430
* OFA: 355
* FFA-1: 484
* FFA-2: 310
* PPA: 1033
* RSC: 566
* OPA: 1611
* OWFA: 464
* VWFA-1: 773
* VWFA-2: 505
* mfs-words: 165
* early: 5917
* midventral: 986
* midlateral: 834
* midparietal: 950
* parietal: 3548
* lateral: 7799
* ventral: 7604  

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch
   * - n_voxels
     - Number of voxels in the selected ROI, varies by ROI and subject

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 10 10 40 15 10
   :header-rows: 1

   * - Name
     - Type
     - Required
     - Description
     - Example
     - Valid Values
   * - subject
     - int
     - True
     - Subject ID from the NSD dataset (1-8)
     - 1
     - 1, 2, 3, 4, 5, 6, 7, 8
   * - roi
     - str
     - True
     - Region of Interest (ROI) for voxel prediction. Early visual areas (V1-V3), category-selective regions (EBA, FFA, etc.), or composite regions (lateral, ventral).
     - V1
     - V1, V2, V3, hV4, EBA, FBA-2, OFA, FFA-1, FFA-2, PPA, RSC, OPA, OWFA, VWFA-1, VWFA-2, mfs-words, early, midventral, midlateral, midparietal, parietal, lateral, ventral
   * - nest_dir
     - str
     - False
     - Root directory of the NEST repository (optional if default paths are set)
     - ./
     - -

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 10 10 40 15 10
   :header-rows: 1

   * - Name
     - Type
     - Required
     - Description
     - Example
     - Valid Values
   * - stimulus
     - numpy.ndarray
     - True
     - A batch of RGB images to be encoded. Images should be in integer format with values in the range [0, 255], and square dimensions (e.g. 224x224).
     - An array of shape [100, 3, 224, 224] representing 100 RGB images.
     - -
   * - device
     - str
     - False
     - Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
     - auto
     - cpu, cuda, auto

Performance
----------

**Accuracy Plots:**

* ``neural_encoding_simulation_toolkit/encoding_models/modality-fmri/train_dataset-nsd/model-fwrf/encoding_models_accuracy``

Example Usage
------------

.. code-block:: python

    from nest import NEST
    
    # Initialize NEST
    nest = NEST(nest_dir="path/to/nest")
    
    # Load the model for subject 1, region V1
    model = nest.get_encoding_model("fmri_nsd_fwrf", subject=1, roi="V1")
    
    # Prepare your stimuli (a batch of images)
    # stimulus shape should be [batch_size, 3, height, width]
    
    # Generate fMRI responses
    responses = nest.encode(model, stimulus)
    
    # responses shape will be [batch_size, n_voxels]
    # where n_voxels depends on the ROI (e.g., 1350 for V1)

References
---------

* x