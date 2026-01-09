==========================
calcium_2p-wang_2025-3DCNN
==========================

Model Summary
------------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - Modality
     - calcium_2p
   * - Training Dataset
     - wang_2025
   * - Species
     - Mouse
   * - Stimuli
     - Videos
   * - Model Type
     - Spatiotemporal convolutional neural network (3D CNN + ConvLSTM)
   * - Creator
     - Tolias Lab (Wang et al., Nature 2025)

Description
----------

This encoding model is a large-scale foundation model of mouse visual cortex trained to predict
dynamic neural responses to natural video stimuli. The model consists of a shared spatiotemporal
neural network core combined with session-specific linear readout layers. The shared core captures
common visual computations across mice and cortical areas, while readout layers map these shared
representations to neurons recorded in individual imaging sessions.

**Neural data.** The model is based on in vivo two-photon (2P) calcium imaging recordings of
excitatory neurons in the visual cortex of awake, behaving mice. Two-photon imaging measures
fluorescence signals that increase when neurons are active due to calcium influx, allowing
large populations of neurons to be recorded simultaneously with cellular resolution. Importantly,
each imaging session samples a different population of neurons, as neuron identities cannot be
reliably tracked across sessions due to changes in imaging depth, field of view, and optical
conditions.

Training data for the shared foundation core consist of approximately 900 minutes of natural video
stimulation pooled from 8 mice (the foundation cohort), covering multiple visual cortical areas
including V1, LM, AL, RL, AM, and PM, and spanning approximately 66,000 neurons. Behavioral signals
(locomotion, pupil size, and eye position) were recorded concurrently and used as modulatory inputs.

**Model architecture.** The original model comprises four modules during training: a perspective module, a modulation
module, a shared core, and a readout module. The perspective module uses ray tracing and eye
position estimates to transform stimulus frames into a retinotopic representation. The modulation
module processes behavioral state variables (locomotion and pupil dilation) to generate dynamic
gain signals. The shared core consists of spatiotemporal convolutional layers and recurrent
components that produce nonlinear visual feature representations over time. The readout module
maps core features to neural responses using linear weights at neuron-specific spatial locations
corresponding to receptive field positions. For inference, the input remains video input.

**Session-specific readouts.** Because two-photon imaging does not provide stable neuron identities
across recording sessions, each session and scan is associated with a distinct readout layer.
Different sessions therefore have different numbers of neurons and independent readout weights,
even when recorded from the same animal. The released model includes a single shared foundation
core together with multiple readout heads corresponding to different sessions and scans from one
mouse, rather than a single readout spanning multiple sessions or animals.

**Stimuli.** Visual stimuli consist primarily of natural videos presented during each session.
Although the exact video clips and temporal segments differ across sessions, all training stimuli
are drawn from the same class of natural videos and share similar statistical
properties. No task labels or semantic annotations are used.

**Model training.** During foundation training, all model components (perspective, modulation,
core, and readout) were trained end-to-end on natural video data pooled across the 8 foundation
mice. For transfer to new mice or sessions, the shared core is frozen and only the perspective,
modulation, and readout modules are trained using limited amounts of natural video data.

**Model testing.** Model performance is evaluated on held-out natural videos as well as on
out-of-distribution stimulus domains not used for training, including static natural images,
drifting Gabor filters, flashing Gaussian dots, directional noise patterns, and random dot
kinematograms. These stimuli are used solely for evaluation and in silico experimentation.

**Noise ceiling.** Predictive performance is reported using a normalized correlation coefficient
(CC_norm), which normalizes the correlation between predicted and recorded responses by an
estimated noise ceiling derived from trial-to-trial variability in the neural data.

**Spatial organization.** The model learns neuron-specific readout positions that recapitulate the
retinotopic organization of mouse visual cortex. Readout feature weights form a functional
embedding for each neuron that can be related to anatomical and physiological properties.

**Output.** The model predicts time-resolved neural activity traces for excitatory neurons in
mouse visual cortex, aligned to video frames and suitable for analyses of tuning, generalization,
and structure–function relationships.

Metadata
--------

**calcium_2p**

    **session** : ``int`` - Session identifier

    **scan** : ``int`` - Scan index within session

    **animal_id** : ``int`` - Animal identifier (17797)

    **unit_id** : ``(N,)`` - Unit identifiers (N = number of neurons)

    **coordinates** : ``(N, 3)`` - 3D spatial coordinates (x, y, z) for each unit

    **OSI** : ``(N,)`` - Orientation Selectivity Index

    **DSI** : ``(N,)`` - Direction Selectivity Index

    **gOSI** : ``(N,)`` - Global Orientation Selectivity Index

    **gDSI** : ``(N,)`` - Global Direction Selectivity Index

    **pref_ori** : ``(N,)`` - Preferred orientation (degrees)

    **pref_dir** : ``(N,)`` - Preferred direction (degrees)

    **roi** : ``dict`` - Binary masks for brain regions
        **V1** : ``(N,)`` - Visual area 1 (1 = unit in V1, 0 = not in V1)

        **LM** : ``(N,)`` - Lateral medial area

        **AL** : ``(N,)`` - Anterolateral area

        **RL** : ``(N,)`` - Rostrolateral area

    **field_masks** : ``dict`` - Binary masks for 2p-calcium imaging fields
        **field_1** : ``(N,)`` - Imaging field 1 (1 = unit in field, 0 = not)

        **field_2** : ``(N,)`` - Imaging field 2

        **...** : ``...`` - Additional fields as present in data
**encoding_model**

    **cc_abs** : ``(N,)`` - Absolute correlation coefficient

    **cc_max** : ``(N,)`` - Maximum correlation coefficient (noise ceiling)

    **cc_norm** : ``(N,)`` - Normalized correlation coefficient

Input
-----

**Type**: ``numpy.ndarray``  
**Shape**: ``[n_frames, height, width] or [n_batches, n_frames, height, width]``  
**Description**: The input should be a sequence of grayscale video frames, or a batch of video sequences.
- Single video: shape [n_frames, height, width]
- Batch of videos: shape [n_batches, n_frames, height, width]


**Constraints:**

* Frame values should be integers in range [0, 255].

Output
------

**Type**: ``numpy.ndarray``  
**Shape**: ``[n_frames, n_neurons] or [n_batches, n_frames, n_neurons]``  
**Description**:  
The output is a 2D or 3D array containing in silico calcium imaging responses.
* Single video: shape [n_frames, n_neurons]
* Batch of videos: shape [n_batches, n_frames, n_neurons]

The last dimension (n_neurons) corresponds to the number of neurons in the selected
session/scan and ROI combination.

Neuron counts vary by session and scan:

* Session 4, Scan 7:  7,493 neurons

* Session 5, Scan 6:  8,592 neurons

* Session 5, Scan 7:  8,138 neurons

* Session 6, Scan 2:  8,158 neurons

* Session 6, Scan 4:  8,221 neurons

* Session 6, Scan 6:  7,971 neurons

* Session 6, Scan 7:  7,887 neurons

* Session 7, Scan 3:  8,618 neurons

* Session 7, Scan 5:  8,194 neurons

* Session 8, Scan 5:  9,941 neurons

* Session 9, Scan 3:  7,973 neurons

* Session 9, Scan 4:  7,855 neurons

* Session 9, Scan 6:  5,130 neurons

**Dimensions:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - n_batches
     - Number of batches chosen for inference
   * - n_frames
     - Number of video frames in the sequence
   * - n_neurons
     - Number of neurons in the selected session/scan and ROI

Parameters
---------

Parameters used in ``get_encoding_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function loads the encoding model.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **train_session**
     - | **Type:** str
       | **Required:** Yes
       | **Description:** Recording session and scan identifier in format 'session-X_scan-Y'
       | **Valid Values:** "session-4_scan-7", "session-5_scan-6", "session-5_scan-7", "session-6_scan-2", "session-6_scan-4", "session-6_scan-6", "session-6_scan-7", "session-7_scan-3", "session-7_scan-5", "session-8_scan-5", "session-9_scan-3", "session-9_scan-4", "session-9_scan-6"
       | **Example:** "session8_scan5"
   * - **selection**
     - | **Type:** dict
       | **Required:** No
       | **Description:** Specifies which outputs to include in the model responses.
       | Can include specific brain areas and/or individual neurons. If not provided,
       | calcium responses are generated for all neurons in the session/scan.
       | 
       | **Properties:**
       | 
       | **roi**
       |     **Type:** list[str]
       |     **Description:** List of brain areas to include in the output
       |     **Valid values:** "AL", "LM", "RL", "V1"
       |     **Example:** ['V1', 'LM']
       | 
       | **field**
       |     **Type:** list[int]
       |     **Description:** Imaging Fields from the recording. Each Scan/Session has a different imaging fields available.
       |     **Valid values:** "1", "2", "3", "4", "5", "6", "7", "8"
       |     **Example:** [1, 3]
       | 
       | **unit_index**
       |     **Type:** numpy.ndarray
       |     **Description:** Binary one-hot encoded vector indicating which neurons to include.
       |     Must have exactly the same length as the total number of neurons in the
       |     selected session/scan (varies by session, e.g., 9,941 for session 8, scan 5).
       |     Each position set to 1 indicates that neuron should be included.
       |     **Example:** [0, 0, '...', 1, 1, 0]

Parameters used in ``encode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function generates in silico neural responses using the encoding model previously loaded.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **device**
     - | **Type:** str
       | **Required:** No
       | **Description:** Device to run the model on. 'auto' will use CUDA if available, otherwise CPU.
       | **Valid Values:** "cpu", "cuda", "auto"
       | **Example:** "auto"
   * - **show_progress**
     - | **Type:** bool
       | **Required:** No
       | **Description:** Whether to show a progress bar during encoding (for large batches).
       | **Example:** True

Performance
----------

**Accuracy Plots (AWS directory):**

* ``brain-encoding-response-generator/encoding_models/modality-calcium_2p/train_dataset-wang_2025/model-3DCNN/encoding_models_accuracy``

Example Usage
------------


.. code-block:: python

    from berg import BERG
    
    # Initialize BERG
    berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
    
    # Load the model
    model = berg.get_encoding_model(
        "calcium_2p-wang_2025-3DCNN",
        train_session="session8_scan5",
        selection={
            "roi": ["V1", "LM"],
            "field": [1, 3],
            "unit_index": [0, 0, '...', 1, 1, 0]
        }
    )
    
    # Prepare the stimulus images
    # Image shape should be [batch_size, 3 RGB channels, height, width]
    images = np.random.randint(0, 255, (100, 3, 256, 256))
    
    # Generates the in silico neural responses to images using the encoding model previously loaded
    responses = berg.encode(
        model,
        images,
        show_progress=True
    )
    
    # The in silico fMRI responses will be a numpy.ndarray of shape:
    # [n_frames, n_neurons] or [n_batches, n_frames, n_neurons]
    # where:
    # - n_batches: Number of batches chosen for inference
    # - n_frames: Number of video frames in the sequence
    # - n_neurons: Number of neurons in the selected session/scan and ROI
    
    # Generate in silico neural responses with metadata
    responses, metadata = berg.encode(
        model,
        images,
        return_metadata=True
    )
    

References
---------

* Model repository: https://github.com/cajal/fnn
* Model paper (Wang et al., Nature 2025): https://www.nature.com/articles/s41586-025-08829-y