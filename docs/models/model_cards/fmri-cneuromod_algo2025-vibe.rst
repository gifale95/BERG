============================
fmri-cneuromod_algo2025-vibe
============================

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
     - Video + Audio + Text
   * - Model Type
     - Transformers
   * - Creator
     - Shrey Dixit, Daniel Carlström Schad, Janis Keck, Viktor Studenyak, Aleksandr Shpilevoi, Andrej Bicanski

Description
----------

VIBE (Video-Input Brain Encoder) is a multimodal fMRI encoding model trained
on CNeuroMod movie data. It combines per-TR language transcripts, movie audio,
and video features to predict whole-brain fMRI activity in Schaefer-1000
parcel space.

Architecture overview:
VIBE uses a two-stage Transformer architecture. In the first stage, a
modality-fusion transformer performs cross-attention across modalities
independently at each time point (TR). Each feature stream (text, audio,
video) is linearly projected to a shared 256-dimensional space together with
a learned subject embedding, and fused via a single-layer Transformer encoder.
The fused per-TR representations are concatenated and passed to the second
stage: a prediction transformer (2 layers) that models temporal dependencies
across TRs using Rotary Positional Embeddings (RoPE). A final feed-forward
layer maps to the 1000-parcel Schaefer output space. The model is trained
with a combined Pearson-correlation + MSE loss and ensembled across multiple
seeds. For full details see Schad, Dixit, Keck et al. (2025),
arXiv:2507.17958.

These BERG-integrated models are modified from the original to use fewer
feature extractors for faster inference and lower memory usage.

Temporal resolution:
The model was trained with a TR of 1.49 s, which is also the prediction
resolution. The number of transcript strings passed as `stimulus` must
exactly match the number of TRs derived from the video (i.e.,
floor(video_duration / 1.49)). A mismatch will raise an error.

The best model (when ensembled) reaches 0.3193 on in-distribution and 0.2122
on out-of-distribution data.

Pretrained variants are available from the Hugging Face collection
'ShreyDixit/vibe'. You can inspect variants via `berg.get_model_variants()`
and load a specific variant using `model_variant=...` in get_encoding_model().

Metadata
--------

.. note::

   Atlas files for glass brain visualization (Schaefer 1000-parcel MNI coordinates) are provided separately in the BERG directory and are not part of the per-subject metadata files.

**roi_masks**

    **cont** : ``(1000,)`` - Binary mask for Control/Frontoparietal network parcels

    **default** : ``(1000,)`` - Binary mask for Default Mode network parcels

    **dorsattn** : ``(1000,)`` - Binary mask for Dorsal Attention network parcels

    **limbic** : ``(1000,)`` - Binary mask for Limbic network parcels

    **salventattn** : ``(1000,)`` - Binary mask for Salience/Ventral Attention network parcels

    **sommot** : ``(1000,)`` - Binary mask for Somatomotor network parcels

    **vis** : ``(1000,)`` - Binary mask for Visual network parcels

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``list[str], str``
   * - Description
     - | Two inputs are required:
       | 1. `stimulus`: A list of per-TR transcripts (one string per TR, where TR = 1.49 s).
       |    The length must match the number of TRs derived from the video.
       | 2. `video_path`: Path to the source video used for audio/video feature extraction.
   * - Example
     - stimulus = ["Hello, are you", "awake? Yes,"]
       video_path = "/path/to/movie.mp4"

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
     - Predicted fMRI activity for each TR.

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - num_timepoints
     - Number of predicted TRs.
   * - num_rois
     - Number of regions (up to 1000 Schaefer parcels, or selected subset).

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
       | **Valid Values:** fmri-cneuromod_algo2025-vibe
       | **Example:** "fmri-cneuromod_algo2025-vibe"
   * - **subject**
     - | **Type:** int
       | **Required:** No
       | **Description:** Subject ID for subject-conditioned prediction.
       | Uses Algonauts-style IDs [1,2,3,5].
       | If omitted in get_encoding_model(), pass subject to encode(..., subject=...).
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
       | **Description:** Hugging Face repository ID of a specific pretrained VIBE variant to load.
       | If provided, its associated config is used and the `config` argument is ignored.
       | Use model.get_pretrained_variants() or berg.get_model_variants(model_id).
       | **Example:** "ShreyDixit/VIBE-Qwen2.5-14B"
   * - **low_mem_use**
     - | **Type:** bool
       | **Required:** No
       | **Description:** If True, unloads heavy components between calls to reduce memory usage
       | (slower but lower VRAM footprint).
       | **Example:** True
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Optional output filtering by network label and/or parcel index mask.
       | If both are provided, they are combined with OR.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** Schaefer 2018 (7-network) labels to keep.
       |     **Valid values:** "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"
       |     **Example:** ['Vis']
       | 
       | **voxel_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector selecting parcels.
       |     Must have length 1000 and contain at least one 1.
       |     **Example:** [0, 0, '...', 1, 1, 0]

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **subject**
     - | **Type:** int
       | **Required:** No
       | **Description:** Subject ID for subject-conditioned prediction.
       | Uses Algonauts-style IDs [1,2,3,5].
       | If omitted in get_encoding_model(), pass subject to encode(..., subject=...).
       | **Valid Values:** 1, 2, 3, 5
       | **Example:** 1
   * - **model**
     - | **Type:** BaseModelInterface
       | **Required:** Yes
       | **Description:** An instantiated and loaded encoding model.
   * - **stimulus**
     - | **Type:** list[str]
       | **Required:** Yes
       | **Description:** A list of transcript strings, one per TR (TR = 1.49 s).
       | The length of this list must exactly match the number of TRs
       | derived from the video duration (floor(video_duration / 1.49)).
       | **Example:**
       | ["Hello, are you", "awake? Yes,"]
   * - **video_path**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Path to the video stimulus file.
       | **Example:** "/path/to/movie.mp4"
   * - **return_metadata**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to return model metadata together with responses.
       | **Example:** True
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding.
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
       | **Valid Values:** fmri-cneuromod_algo2025-vibe
       | **Example:** "fmri-cneuromod_algo2025-vibe"
   * - **subject**
     - | **Type:** int
       | **Required:** No
       | **Description:** Subject ID for subject-conditioned prediction.
       | Uses Algonauts-style IDs [1,2,3,5].
       | If omitted in get_encoding_model(), pass subject to encode(..., subject=...).
       | **Valid Values:** 1, 2, 3, 5
       | **Example:** 1

Model-specific utility methods
------------------------------

``get_model_variants()``
~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve available pretrained variants for this model without instantiating it.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **model_id**
     - | **Type:** ``str``
       | **Required:** Yes
       | **Description:** Unique identifier of the model to load.

.. code-block:: python

    variants = berg.get_model_variants("fmri-cneuromod_algo2025-vibe")

----

``generate_glass_brain_animation()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates and saves an animated glass brain GIF from predicted responses.
Called directly on the loaded model instance.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **responses**
     - | **Type:** ``torch.Tensor``
       | **Required:** Yes
       | **Description:** Model predictions generated by encode().
   * - **out_path**
     - | **Type:** ``str``
       | **Required:** No
       | **Default:** brain_activation.gif
       | **Description:** Path for the generated GIF.

.. code-block:: python

    model.generate_glass_brain_animation(responses, out_path="activation.gif")

Performance
----------

**Metrics:**

* **Mean parcel-wise Pearson correlation**: ID Friends S07: 0.3193; OOD (6 films): 0.2122
* **Model variants**: Available in Hugging Face Collection: ShreyDixit/vibe

Example Usage
------------


.. code-block:: python

    from berg import BERG

    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")

    # Discover all model variants
    variants = berg.get_model_variants("fmri-cneuromod_algo2025-vibe")

    # Load the model
    model = berg.get_encoding_model(
        "fmri-cneuromod_algo2025-vibe",
        subject=1,
        device="auto",
        model_variant="ShreyDixit/VIBE-Qwen2.5-14B",
        low_mem_use=True,
        selection={
            "roi": ["Vis"],
            "voxel_index": [0, 0, '...', 1, 1, 0]
        }
    )
    
    # Prepare stimulus: one transcript string per TR, matching video duration
    transcripts = ["Hello, are you", "awake? Yes,", "I just woke up."]
    video_path = "/path/to/movie.mp4"
    
    # Generates the in silico neural responses using the encoding model previously loaded
    responses = berg.encode(
        model,
        transcripts, 
        video_path=video_path
    )
    
    # The in silico fMRI responses will be a torch.Tensor of shape:
    # ['num_timepoints', 'num_rois']
    # where:
    # - num_timepoints: Number of predicted TRs.
    # - num_rois: Number of regions (up to 1000 Schaefer parcels, or selected subset).
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        stimulus,
        return_metadata=True
    )
    
    # Load the encoding model's metadata without having to load the model itself
    metadata = berg.get_model_metadata(
        "fmri-cneuromod_algo2025-vibe",
    )
    
    # Generate a gif out of the responses
    gif_path = model.generate_glass_brain_animation(
      responses=responses, 
      out_path="brain_activation.gif")

References
---------

* Schad, Daniel Carlström; Dixit, Shrey; Keck, Janis; Studenyak, Viktor; Shpilevoi, Aleksandr; Bicanski, Andrej. VIBE: Video-Input Brain Encoder for fMRI Response Modeling. arXiv:2507.17958 (2025).