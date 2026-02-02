========================
fmri-cneuromod-text2fmri
========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - CNeuroMod
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
solely from video transcripts.

Multiple configurations are available to suit different resource constraints. 
The smallest and most lightweight configuration consists of approximately 52M 
trainable parameters, leveraging a frozen 500M parameter LLM (Qwen-2.5-0.5B) 
for feature extraction. 

Trained on the CNeuroMods dataset (Friends and Movie10)—the same data used for 
the Algonauts 2025 Challenge—this model generates in silico neural responses 
without requiring visual or audio inputs. Despite its efficiency, even the 
smallest model outperforms standard baselines and achieves near-SOTA performance 
in auditory and language-selective cortices.

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``list[str]``
   * - Description
     - A list of strings where each string corresponds to the text spoken during a 
       single fMRI Time Repetition (TR).
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
       | **Valid Values:** fmri-cneuromod-text2fmri
       | **Example:** "fmri-cneuromod-text2fmri"
   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** The ID of the subject to generate predictions for.
       | **Valid Values:** 0, 1, 2, 3
       | **Example:** 0
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** The computing device to use for inference.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "auto"
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
       | **Valid Values:** fmri-cneuromod-text2fmri
       | **Example:** "fmri-cneuromod-text2fmri"
   * - **subject**
     - | **Type:** int
       | **Required:** Yes
       | **Description:** The ID of the subject to generate predictions for.
       | **Valid Values:** 0, 1, 2, 3
       | **Example:** 0

Performance
----------

**Metrics:**

* **Pearson Correlation**: Refer to Hugging Face Collection for specific values

* **Collection**: ShreyDixit/text2fmri

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "fmri-cneuromod-text2fmri",
        subject=0,
        selection={
            "roi": ["Vis"]
        }
    )
    
    # Prepare the stimulus (text/sentences)
    stimulus = ["Hello, are you", "awake? Yes,"]
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        stimulus,
        low_mem_use=True
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
        "fmri-cneuromod-text2fmri",
        subject=0
    )
    

References
---------

* Course Materials: Dixit, S. (2026). Text2fMRI: Brain Encoding Models using LLMs (Course Materials) (v0.1.2). Zenodo. https://doi.org/10.5281/zenodo.18369862
* Huggingface Collection: https://huggingface.co/ShreyDixit/Text2fMRI-Qwen-2.5-0.5B