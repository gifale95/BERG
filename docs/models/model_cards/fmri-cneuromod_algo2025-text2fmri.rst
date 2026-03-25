=================================
fmri-cneuromod_algo2025-text2fmri
=================================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - CNeuroMod (Algonauts 2025 challenge preparation)
   * - Species
     - Human
   * - Stimuli
     - Text
   * - Model Type
     - Transformers
   * - Creator
     - Shrey Dixit

Description
----------

Text2fMRI offers a suite of lightweight encoding models, available through the Hugging Face 
collection 'ShreyDixit/Text2fMRI', designed to predict whole-brain fMRI responses 
solely from movie language transcripts.

Trained on the CNeuroMods dataset (Friends and Movie10)—the same data used for 
the Algonauts 2025 Challenge—this model generates in silico neural responses 
to movies without requiring visual or audio input.

Multiple model configurations are available to suit different resource constraints. 
The smallest and most lightweight model configuration consists of approximately 52M 
trainable parameters, leveraging a frozen 500M parameter LLM (Qwen-2.5-0.5B) 
for feature extraction.

Additionally, this model includes specific utility functions to query available 
Hugging Face model variants prior to instantiation (`berg.get_model_variants()`) 
and to render animated spatial visualizations of predicted activity 
(`model.generate_glass_brain_animation()`). The atlas files required for glass brain visualization are provided separately in the BERG directory.

Metadata
--------

.. note::

   Atlas files for glass brain visualization (Schaefer 1000-parcel MNI coordinates) are provided separately in the BERG directory and are not part of the per-subject metadata files.

**roi_masks**

    **Cont** : ``(1000,)`` - Binary mask for Control/Frontoparietal network parcels

    **Default** : ``(1000,)`` - Binary mask for Default Mode network parcels

    **DorsAttn** : ``(1000,)`` - Binary mask for Dorsal Attention network parcels

    **Limbic** : ``(1000,)`` - Binary mask for Limbic network parcels

    **SalVentAttn** : ``(1000,)`` - Binary mask for Salience/Ventral Attention network parcels

    **SomMot** : ``(1000,)`` - Binary mask for Somatomotor network parcels

    **Vis** : ``(1000,)`` - Binary mask for Visual network parcels

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``list[str]``
   * - Description
     - | A list of strings where each string corresponds to the text spoken during a 
       | single fMRI Time Repetition (TR).
   * - Example
     - ``["Hello, are you", "awake? Yes,"]``

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``torch.Tensor``
   * - Shape
     - ``['num_timepoints', 'num_rois']``
   * - Description
     - The predicted fMRI activity for the given stimulus.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - num_timepoints
     - Number of TRs (timepoints) in the input stimulus.
   * - num_rois
     - Number of Regions of Interest (1000 parcels from Schaefer 2018 atlas).

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model_id**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Unique identifier of the model to load.
       | **Valid Values:** fmri-cneuromod_algo2025-text2fmri
       | **Example:** "fmri-cneuromod_algo2025-text2fmri"
   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** The ID of the subject to generate predictions for.
       | **Valid Values:** 1, 2, 3, 5
       | **Example:** 1
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** The computing device to use for inference.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "auto"
   * - **model_variant**
     - | **Type:** str
       | **Required:** No
       | **Description:** HuggingFace repository ID of a specific pretrained variant to load.
       | If provided, the model config associated with this variant is used and any user-passed `config` argument is ignored.
       | If None (default), loads the default configuration (Qwen-2.5-0.5B).
       | Use model.get_pretrained_variants() on any loaded model to see all available options.
       | **Example:** "ShreyDixit/Text2fMRI-Qwen-2.5-0.5B"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Optional filter to restrict the output to specific brain networks.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** Filter output by Schaefer 2018 (7-network) atlas labels.
       |     **Valid values:** "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"
       |     **Example:** ['Vis']
       | 
       | **voxel_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which voxels to include.
       |     Must have exactly the same length as the number of available voxels (1000).
       |     Each position set to 1 indicates that voxel should be included.
       |     **Example:** [0, 0, '...', 1, 1, 0]

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model**
     - | **Type:** BaseModelInterface
       | **Required:** Yes
       | **Description:** An instantiated and loaded encoding model.
   * - **stimulus**
     - | **Type:** list[str]
       | **Required:** Yes
       | **Description:** A list of strings where each string corresponds to the text spoken during a 
       | single fMRI Time Repetition (TR).
       | **Example:**
       | ["Hello, are you", "awake? Yes,"]
   * - **low_mem_use**
     - | **Type:** bool
       | **Required:** No
       | **Description:** If True, sequentially loads/unloads the Feature Extractor and the Encoding Model 
       | to minimize VRAM usage, at the cost of slower execution.
       | **Example:** True
   * - **return_metadata**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to return the encoding model's metadata together with the in silico neural responses.
       | **Example:** True
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding (for large batches).
       | **Example:** True

Parameters used in ``get_model_metadata``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model's metadata without having to load the model itself.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model_id**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Unique identifier of the model to load.
       | **Valid Values:** fmri-cneuromod_algo2025-text2fmri
       | **Example:** "fmri-cneuromod_algo2025-text2fmri"
   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** The ID of the subject to generate predictions for.
       | **Valid Values:** 1, 2, 3, 5
       | **Example:** 1

Model-specific utility methods
------------------------------

``get_model_variants()``
~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve available pretrained variants for this model without instantiating it.
This is called from the main BERG class.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model_id**
     - | **Type:** ``str``
       | **Required:** Yes
       | **Description:** Unique identifier of the model to load.

.. code-block:: python

    variants = berg.get_model_variants("fmri-cneuromod_algo2025-text2fmri")

----

``generate_glass_brain_animation()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates and saves an animated glass brain GIF from the predicted responses.
Called directly on the loaded model instance.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **responses**
     - | **Type:** ``numpy.ndarray``
       | **Required:** Yes
       | **Description:** Model predictions generated by the encode() function.
   * - **out_path**
     - | **Type:** ``str``
       | **Required:** No
       | **Default:** brain_activation.gif
       | **Description:** Where to save the generated GIF.

.. code-block:: python

    model.generate_glass_brain_animation(responses, out_path="activation.gif")

Performance
----------

**Metrics:**

* **Performance Metrics**: Available in Hugging Face Collection: ShreyDixit/text2fmri

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")

    # Discover all model variants
    variants = berg.get_model_variants("fmri-cneuromod_algo2025-text2fmri")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-cneuromod_algo2025-text2fmri",
        subject=1,
        model_variant="ShreyDixit/Text2fMRI-Qwen-2.5-0.5B",
        selection={
            "roi": ["Vis"],
            "voxel_index": [0, 0, '...', 1, 1, 0]
        }
    )
    
    # Prepare the stimulus (text/sentences)
    stimulus = ["Hello, are you", "awake? Yes,"]
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
    )
    
    # The in silico fMRI responses will be a torch.Tensor of shape:
    # ['num_timepoints', 'num_rois']
    # where:
    # - num_timepoints: Number of TRs (timepoints) in the input stimulus.
    # - num_rois: Number of Regions of Interest (1000 parcels from Schaefer 2018 atlas).
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-cneuromod_algo2025-text2fmri",
        subject=1
    )
    
    # Generate a gif out of the responses
    gif_path = model.generate_glass_brain_animation(
      responses=responses, 
      out_path="brain_activation.gif")
    

References
---------

* Course Materials: Dixit, S. (2026). Text2fMRI: Brain Encoding Models using LLMs (Course Materials) (v0.1.2). Zenodo. https://doi.org/10.5281/zenodo.18369862
* Huggingface Collection: https://huggingface.co/ShreyDixit/Text2fMRI-Qwen-2.5-0.5B
