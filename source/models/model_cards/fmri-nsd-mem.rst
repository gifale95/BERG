============
fmri-nsd-mem
============

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - Natural Scenes Dataset (NSD)
   * - Species
     - Human
   * - Stimuli
     - Images
   * - Model Architecture
     - memory_encoding_model
   * - Creator
     - Huzheng Yang

Description
----------

The Memory Encoding Model (MEM) won the Algonauts 2023 
visual brain competition with a score of 70.8, demonstrating that the entire cortex becomes 
largely predictable when considering memory information.

The model was trained on the Natural Scenes Dataset using up to 32 previous image frames
along with the current frame, learning periodic delayed response patterns correlated with
6th-7th prior images. This enables prediction of brain responses across both visual and
non-visual regions including Somatomotor (r=0.16), Auditory (r=0.14), and Anterior cortex (r=0.18).

**Architecture:** The model uses a Vision Transformer backbone with memory compression modules
and time-aware embeddings. A two-part model architecture handles the computational complexity
of predicting responses across ~160k brain vertices with memory integration.

**Training:** Models were trained separately for each of 8 NSD subjects using subject-specific
brain anatomy and response patterns. The ensemble approach with random-ROI training recipe
improved performance from single model score of 66.8 to ensemble score of 70.8.

Input
-----

**Type**: ``numpy.ndarray``  
**Shape**: ``['batch_size', 3, 'height', 'width']``  
**Description**: The input should be a batch of RGB images.

**Constraints:**

* Image values should be integers in range [0, 255].
* Image dimensions (height, width) should be equal (square).
* Minimum recommended image size: 224×224 pixels.

Output
------

**Type**: ``tuple of numpy.ndarray``  
**Shape**: ``([batch_size, lh_vertices], [batch_size, rh_vertices])``  
**Description**:  
The output is a tuple containing the left hemisphere (LH) and right hemisphere (RH) in silico fMRI
responses for the batch images.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - batch_size
     - Number of stimuli in the batch.
   * - lh_vertices
     - Number of selected LH vertices for which the in silico fMRI responses are generated.
   * - rh_vertices
     - Number of selected RH vertices for which the in silico fMRI responses are generated.

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** Subject ID from the NSD dataset (1-8).
       | **Valid Values:** 1, 2, 3, 4, 5, 6, 7, 8
       | **Example:** 1
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses. If not provided, fMRI responses are generate for all LH and RH fMRI vertices.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** str
       |     **Description:** Region of Interest (ROI) for voxel prediction.
       |     Early visual areas (V1-V3), category-selective regions (EBA, FFA, etc.),
       |     or composite regions (lateral, ventral).
       |     **Valid values:** "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"
       |     **Example:** V1
       | 
       | **lh_vertices**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector with ones indicating the left hemisphere (LH)
       |     vertices for which the in silico fMRI responses are generated. This vector must
       |     have exactly the same length as the number of LH fsaverage vertices (163,842).
       |     The vertices from the one-hot encoded vector are only selected if the "roi" key
       |     is not provided, or has value None.
       | 
       | **rh_vertices**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector with ones indicating the right hemisphere (RH)
       |     vertices for which the in silico fMRI responses are generated. This vector must
       |     have exactly the same length as the number of RH fsaverage vertices (163,842).
       |     The vertices from the one-hot encoded vector are only selected if the "roi" key
       |     is not provided, or has value None.

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **stimulus**
     - | **Type:** numpy.ndarray
       | **Required:** Yes
       | **Description:** A batch of RGB images to be encoded. Images should be in integer format with values in the range [0, 255], and square dimensions (e.g. 224x224).
       | **Example:** An array of shape [100, 3, 224, 224] representing 100 RGB images.
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
       | **Valid Values:** cpu, cuda, auto
       | **Example:** auto

Performance
----------

**Accuracy Plots:**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-nsd/model-mem/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model("fmri-nsd-mem", subject=1, selection={"roi": "V1v"})
    # This function loads the encoding model.
    
    # Prepare your stimuli
    # stimulus shape should be ['batch_size', 3, 'height', 'width']
    
    # Generate responses
    responses = berg.encode(model, stimulus, device="auto")
    # This function generates in silico neural responses using the encoding model previously loaded.
    

References
---------

* {'Model building code': 'https://github.com/huzeyann/MemoryEncodingModel/tree/main'}
* {'Model Paper': 'https://arxiv.org/abs/2308.01175'}