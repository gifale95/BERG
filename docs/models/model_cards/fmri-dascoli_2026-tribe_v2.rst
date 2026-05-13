==========================
fmri-dascoli_2026-tribe_v2
==========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - fMRI
   * - Training Dataset
     - d'Ascoli et al. (2026) (CNeuroMod, BoldMoments, Lebel2023, Wen2017)
   * - Species
     - Human
   * - Stimuli
     - Video, Audio, Text
   * - Model Type
     - Multimodal Transformer (TRIBE v2)
   * - Creator
     - Stéphane d’Ascoli (FAIR at Meta)

Description
----------

**Setup.** This model requires Python >= 3.11, FFmpeg, and HuggingFace authentication.

1. Install FFmpeg:
   - Linux: ``sudo apt install ffmpeg``
   - macOS: ``brew install ffmpeg``
   - Conda: ``conda install -c conda-forge ffmpeg``
2. Install the HuggingFace CLI: ``pip install huggingface_hub``
3. Request access to LLaMA-3.2-3B at https://huggingface.co/meta-llama/Llama-3.2-3B
4. Authenticate: ``hf auth login``
5. Enter your HuggingFace username and access token when prompted.

GPU with >= 16 GB VRAM is recommended. CPU inference is supported but very slow.

**What is TRIBE v2?**
TRIBE v2 is a tri-modal (video, audio, and language) foundation model for predicting human fMRI
brain activity. It uses frozen pretrained feature extractors — V-JEPA2-Giant (video), Wav2Vec-BERT-2.0
(audio), and LLaMA-3.2-3B (text) — whose embeddings are fed into a trainable 8-layer, 8-head
Transformer encoder that maps multimodal representations onto the cortical surface (fsaverage5,
20,484 vertices).

**Architecture.** Stimulus features are extracted at 2 Hz from each modality, projected to a shared
384-dimensional space per modality (1,152 total), and processed by an 8-layer, 8-head Transformer with
100-second context windows. A subject-conditioned final layer maps latent representations to cortical
vertices. For inference, the model uses a special "unseen subject" layer trained via subject dropout,
producing group-average-like predictions without requiring subject-specific data. No subject
parameter is needed, the model always runs in this mode.

**Training data.** The model was trained on over 450 hours of fMRI across 25 subjects from four
naturalistic datasets: Courtois NeuroMod (4 subjects, 269h — movies with speech), BoldMoments
(10 subjects, 62h — short video clips), Lebel2023 (8 subjects, 86h — podcast listening), and
Wen2017 (3 subjects, 35h — silent videos).

**Feature extraction pipeline.** When given a video, the model automatically (1) extracts audio from
the video track, (2) transcribes speech with WhisperX to get word-level timings, (3) extracts visual
features from V-JEPA2-Giant (64 frames spanning 4 seconds per time bin), audio features from
Wav2Vec-BERT-2.0, and text features from LLaMA-3.2-3B with 1,024 tokens of preceding context. When
given text only, it first synthesizes speech via gTTS and then runs the same audio+text pipeline. Which means,
you provide only one file to the model, where a video file triggers the whole pipeline.

**Output.** Predictions are time-resolved fMRI activity at 1 Hz (1 TR = 1 second) across 20,484
cortical vertices on the fsaverage5 surface mesh. ROI selection is available via the Glasser
HCP-MMP1.0 parcellation (180 bilateral cortical regions).

**Performance.** An earlier iteration of TRIBE v2 achieved first place in the Algonauts 2025 brain
prediction competition (263 teams, mean score 0.2146). The current model, trained on over 1,000
hours of fMRI across 720 subjects, significantly outperforms linear encoding baselines across all
training datasets and generalizes zero-shot to unseen subjects and tasks, including non-naturalistic
experimental paradigms such as visual and language functional localizers.

Metadata
--------

**fmri**

    **subject_id** : ``str`` - Subject identifier ('average')

    **n_vertices** : ``int`` - Total cortical vertices (20484)

    **n_vertices_lh** : ``int`` - Left hemisphere vertices (10242)

    **n_vertices_rh** : ``int`` - Right hemisphere vertices (10242)

    **surface_mesh** : ``str`` - Surface mesh name ('fsaverage5')

    **output_frequency_hz** : ``float`` - Temporal resolution of predictions (1.0 Hz)
**roi**

    **parcellation** : ``str`` - Parcellation name ('Glasser_HCP-MMP1.0')

    **roi_labels** : ``(180,)`` - Bilateral ROI names (e.g., 'V1', 'V2', 'FFC')

    **roi_assignments** : ``(20484,)`` - ROI index per vertex (-1 = medial wall)

    **roi_index** : ``dict`` - Mapping from ROI name to integer index in roi_assignments

Input
-----

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``str (file path)``
   * - Description
     - | The input is a single file path (string) to a video, audio, or text file of any duration.
       | The file type is auto-detected from the extension, and determines which
       | modalities are activated:
       | 
       | • Video input → visual + audio + text features (full multimodal)
       | • Audio input → audio + text features (speech is transcribed)
       | • Text input  → audio + text features (text is first synthesized to speech)
       | 
       | Exactly one file path must be provided per call. Audio and text are automatically
       | extracted from the video file.
       | 
       | Stimuli are processed as temporal sequences. Features are extracted at 2 Hz and
       | predictions are returned at 1 Hz, producing one predicted fMRI sample per second
       | of stimulus duration.

Output
------

.. list-table::
   :widths: 20 80
   :stub-columns: 1

   * - Type
     - ``numpy.ndarray``
   * - Shape
     - ``[n_timesteps, n_vertices]``
   * - Description
     - | The output is a 2D array containing predicted fMRI activity on the fsaverage5
       | cortical surface. Shape is (n_timesteps, n_vertices), where n_timesteps depends
       | on stimulus duration (1 TR = 1 second) and n_vertices depends on ROI selection. However, these predictions
       | should not be interpreted as a direct one-to-one mapping from a single stimulus second to the same fMRI second,
       | because fMRI responses are delayed and temporally blurred by the hemodynamic response.
       | TRIBE v2 also uses temporal context, so each prediction can depend on surrounding/preceding stimulus information, not only the current second.
   * - Dimensions
     - | **n_timesteps**: Number of seconds of predicted brain activity (1 per second, depends on stimulus duration)
       | **n_vertices**: Number of cortical vertices in the selection (up to 20,484)

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
       | **Valid Values:** fmri-multi_study-tribe_v2
       | **Example:** "fmri-multi_study-tribe_v2"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which cortical vertices to include in the model output.
       | Can include ROI names and/or a binary vertex mask. If both are provided,
       | their union (logical OR) is used. If not provided, all 20,484 vertices
       | are returned.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of Glasser HCP-MMP1.0 ROI names to include.
       |     Selects vertices from both hemispheres for each named region.
       |     **Valid values:** "V1", "MST", "V6", "V2", "V3", "V4", "V8", "4", "3b", "FEF", "PEF", "55b", "V3A", "RSC", "POS2", "V7", "IPS1", "FFC", "V3B", "LO1", "LO2", "PIT", "MT", "A1", "PSL", "SFL", "PCV", "STV", "7Pm", "7m", "POS1", "23d", "v23ab", "d23ab", "31pv", "5m", "5mv", "23c", "5L", "24dd", "24dv", "7AL", "SCEF", "6ma", "7Am", "7PL", "7PC", "LIPv", "VIP", "MIP", "1", "2", "3a", "6d", "6mp", "6v", "p24pr", "33pr", "a24pr", "p32pr", "a24", "d32", "8BM", "p32", "10r", "47m", "8Av", "8Ad", "9m", "8BL", "9p", "10d", "8C", "44", "45", "47l", "a47r", "6r", "IFJa", "IFJp", "IFSp", "IFSa", "p9-46v", "46", "a9-46v", "9-46d", "9a", "10v", "a10p", "10pp", "11l", "13l", "OFC", "47s", "LIPd", "6a", "i6-8", "s6-8", "43", "OP4", "OP1", "OP2-3", "52", "RI", "PFcm", "PoI2", "TA2", "FOP4", "MI", "Pir", "AVI", "AAIC", "FOP1", "FOP3", "FOP2", "PFt", "AIP", "EC", "PreS", "H", "ProS", "PeEc", "STGa", "PBelt", "A5", "PHA1", "PHA3", "STSda", "STSdp", "STSvp", "TGd", "TE1a", "TE1p", "TE2a", "TF", "TE2p", "PHT", "PH", "TPOJ1", "TPOJ2", "TPOJ3", "DVT", "PGp", "IP2", "IP1", "IP0", "PFop", "PF", "PFm", "PGi", "PGs", "V6A", "VMV1", "VMV3", "PHA2", "V4t", "FST", "V3CD", "LO3", "VMV2", "31pd", "31a", "VVC", "25", "s32", "pOFC", "PoI1", "Ig", "FOP5", "p10p", "p47r", "TGv", "MBelt", "LBelt", "A4", "STSva", "TE1m", "PI", "a32pr", "p24"
       |     **Example:** ['V1', 'V2', 'FFC']
       | 
       | **vertices**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which vertices to include.
       |     Must have exactly 20,484 elements (10,242 left + 10,242 right hemisphere).
       |     Each position set to 1 indicates that vertex should be included.
       |     **Example:** [0, 0, ..., 1, 1, 0]
   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
       | GPU with >= 16 GB VRAM is strongly recommended. CPU inference is supported but very slow.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "auto"

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
     - | **Type:** str
       | **Required:** Yes
       | **Description:** File path (string) to the stimulus. Exactly one file must be provided.
       | The file type is auto-detected from the extension:
       | 
       |   • Video: .mp4, .avi, .mkv, .mov, .webm → activates video + audio + text features
       |   • Audio: .wav, .mp3, .flac, .ogg → activates audio + text features
       |   • Text:  .txt → text is synthesized to speech, then activates audio + text features
       | **Example:** "'/path/to/video.mp4'"
   * - **return_metadata**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to return the encoding model's metadata together with the in silico neural responses.
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
       | **Valid Values:** fmri-multi_study-tribe_v2
       | **Example:** "fmri-multi_study-tribe_v2"

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-fmri/train_dataset-multi_study/model-tribe_v2/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")

    # Create optional vertex mask
    vertex_mask = np.zeros(20484, dtype=int)
    vertex_mask[100:200] = 1

    # Load the model with ROI and/or vertex selection
    model = berg.get_encoding_model(
        "fmri-dascoli_2026-tribe_v2",
        selection={
            "roi": ["V1", "V2", "FFC"],
            "vertices": vertex_mask,
        },
    )

    # Prepare the stimulus (file path to video, audio, or text)
    # Video: audio is extracted and speech transcribed automatically
    stimulus = "/path/to/video.mp4"

    # Generate in silico neural responses
    responses = berg.encode(model, stimulus, show_progress=True)

    # responses shape: [n_timesteps, n_vertices]
    # - n_timesteps: one per second of stimulus duration
    # - n_vertices: cortical vertices in the selection (up to 20,484)

    # Generate responses with metadata
    responses, metadata = berg.encode(model, stimulus, return_metadata=True)

    # Load metadata without loading the model
    metadata = berg.get_model_metadata("fmri-multi_study-tribe_v2")

References
---------

* TRIBE v2 paper (d'Ascoli et al., 2026): https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
* TRIBE v2 code: https://github.com/facebookresearch/tribev2
* TRIBE v2 weights: https://huggingface.co/facebook/tribev2
* TRIBE v2 demo: https://aidemos.atmeta.com/tribev2/
* Glasser parcellation (Glasser et al., 2016): https://doi.org/10.1038/nature18933
* Algonauts 2025 challenge (Gifford et al., 2024): https://arxiv.org/abs/2501.00504