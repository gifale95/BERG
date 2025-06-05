=======================
Adding Models to NEST
=======================

This guide walks you through the process of adding new models to the Neural Encoding Simulation Toolkit (NEST). You can also execute this `tutorial on Google Colab <https://colab.research.google.com/drive/1nBxEiJATzJdWwfzRPmyai2G76HkeBhAU>`_.


Overview
=========

Adding a new model to NEST requires creating two main files:

1. **A Python model file** — implements the model's core logic (i.e., how it processes input stimuli to produce predicted neural responses) and defines a standard interface (i.e., functions like ``predict()`` that allow the model to interact with NEST).
2. **A YAML configuration file** — describes the model's parameters and behavior, including information about the model architecture, training data, and training procedure.

These two files work together to register your encoding model with NEST and make it compatible with NEST's unified interface — a common structure that allows users to call any model in the same way, regardless of its internal details.



Step 1: Determining Your Model's Location in the GitHub repository
==========================================

First, determine which **modality** your model belongs to:

- If it's an **fMRI model**, add it to:  
  ``nest/models/fmri/``

- If it's an **EEG model**, add it to:  
  ``nest/models/eeg/``

- If it's a **new modality**, create a new directory:  
  ``nest/models/your_new_modality/``

Your corresponding **YAML configuration file** should go here:  
``nest/models/model_cards/your_model_id.yaml``

**Note**: If you are adding a new modality, there are a few additional considerations. These are discussed in **Step 4** below.

Step 2: Creating the YAML Configuration File
============================================

In the NEST toolkit, you'll find a template YAML file at: ``nest/models/model_cards/template.yaml``
This template serves as a guide for creating your model's configuration.

Your YAML file is **crucial** because it:

- Provides metadata about your model
- Defines input/output specifications
- Documents valid parameters and constraints
- Is used for parameter validation in your model class
- Generates model cards for end users

Be as detailed as possible — this helps others understand how to work with your model.  
**You can reference existing model YAML files as examples.**

Here's a template for your YAML configuration file:

.. code-block:: yaml

    # Template YAML file for NEST model specification
    # Replace placeholder values with actual model information

    # Basic metadata
    model_id: modality-dataset-model_type  # e.g., fmri-nsd-fwrf
    modality: modality  # e.g., fmri, eeg, meg, ...
    training_dataset: dataset_name
    species: Human  # e.g., Human, Macaque, etc.
    stimuli: Images  # e.g., Images, Sounds, Text, etc.
    model_architecture: feature_extraction_method  # e.g., ViT-B/32, fwRF, etc.
    creator: your_name

    # General description of the model
    description: |
    Provide a concise but informative description of the model, including:
    - What kind of in silico neural responses it generates
    - What dataset it was trained on
    - The basic approach/architecture
    - Any notable characteristics or limitations
    Keep this to 3-5 sentences for readability.

    # Input stimulus information
    input:
    type: "numpy.ndarray"  # or other appropriate type
    shape: [shape_description]  # e.g., [batch_size, 3, height, width]
    description: "Brief description of input format"
    constraints:
        - "List any constraints on input values"
        - "e.g., value ranges, size requirements, etc."

    # Output information
    output:
    type: "numpy.ndarray"  # or other appropriate type
    shape: [shape_description]  # e.g., [batch_size, n_voxels]
    description: "Brief description of output format"
    dimensions:
        - name: "dimension_name"
        description: "What this dimension represents"
        - name: "dimension_name"
        description: "What this dimension represents"
        # Add more dimensions as needed

    # Model parameters and their usage
    parameters:
    # First parameter
    param_name:
        type: param_type  # e.g., int, str, float
        required: true/false
        valid_values: list_of_valid_values  # or range, or omit if not applicable
        default: default_value  # include if there's a default value
        example: example_value
        description: "Description of what this parameter represents"
        function: "Which function uses this parameter: get_encoding_model, load_model, .."

    # Selection parameter to define specific outputs (ROI, channels, timepoints, etc.)
    selection:
        type: dict
        required: true
        description: |
        Specifies which outputs to include in the in silico model responses.
        This parameter defines for which data the in silico responses should be generated
        (e.g., specific ROI, timepoints, channels, etc.)
        function: get_encoding_model
        properties:
        key_name:  # Replace with model-specific keys, e.g., "roi", "channels", "timepoints"
            type: any
            description: "Description of Model-specific selection criterion."
            example: "V1"

    # Add more parameters as needed
    param_name:
        type: param_type
        required: true/false
        valid_values: list_of_valid_values  # or range, or omit if not applicable
        default: default_value  # include if there's a default value
        example: example_value
        description: "Description of what this parameter represents"
        function: "Which function uses this parameter"

    # Performance metrics (if needed) and references
    performance:
    metrics:
        - name: "metric_name"
        value: "metric_value"
        description: "What this metric represents"
        
        # Add more metrics as needed
        - name: "metric_name"
        value: "metric_value"
        description: "What this metric represents"
    
    plots: "URL_to_performance_plots"  # URL or path to visualizations

    # Add References here
    references:
        - "Citation for your model or training dataset"


Step 3: Implementing the Model Class
====================================

Now we'll build the complete model implementation step by step. The required functions must be named **exactly as shown** to work with the ``BaseModelInterface``. You are free to add additional helper functions as needed — but the core methods must be implemented.

3.1: Model Registration
-----------------------

First, set up the model registration code that makes your model discoverable by the NEST toolkit.


This code:

1. Loads your model's configuration from the YAML file  
2. Registers your model with the NEST registry, making it discoverable  
3. Specifies the module path, class name, and modality

.. code-block:: python

    import os
    import yaml
    from nest.core.model_registry import register_model

    # Load model info from YAML
    def load_model_info():
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
        with open(os.path.abspath(yaml_path), "r") as f:
            return yaml.safe_load(f)

    # Load model_info once at the top
    model_info = load_model_info()

    # Register this model with the registry using model_info
    register_model(
        model_id=model_info["model_id"],
        module_path="nest.models.your_modality.your_model_file",  # Replace with actual path
        class_name="YourModelClass",
        modality=model_info.get("modality", "your_modality"),
        training_dataset=model_info.get("training_dataset", "your_dataset"),
        yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
    )

3.2: Class Initialization and Parameter Validation
-------------------------------------------------

Next, define your model class by inheriting from ``BaseModelInterface`` and implement the initialization logic.

The initialization method:

1. Stores user-provided parameters (e.g., subject ID, device, NEST directory)  
2. Validates parameters against the specifications in the YAML file  
3. Sets up the compute device (CPU or GPU)  
4. Can process additional model-specific parameters through `**kwargs`

Most importantly, you can include a `selection` parameter to specify which parts of the model output should be returned.  
This is useful for selecting specific brain regions (e.g., "V1"), timepoints, or channels from the full in silico response.  
It allows users to work with only the subset of data relevant to their analysis, reducing memory usage and improving flexibility.  
The structure and valid values of this parameter should be defined in the model’s YAML configuration file (see above).

.. code-block:: python


    class YourModelClass(BaseModelInterface):
        """
        Your model description here. Explain what this model does, what
        in silico neural responses it generates, and any other important details.
        """
        
        MODEL_ID = model_info["model_id"]
        # Extract any validation info from model_info
        VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
        
        def __init__(self, subject: int, selection: Dict, device: str = "auto", nest_dir: Optional[str] = None, **kwargs):
            """
            Initialize your model with the required parameters.
            
            Parameters
            ----------
            subject : int
                Subject ID for subject-specific models.
            selection : dict
                Specifies which outputs to include in the model responses
                (ROI, Time interval, ...)
            device : str
                Device to run the model on ('cpu', 'cuda', or 'auto').
            nest_dir : str, optional
                Path to the NEST directory.
            **kwargs
                Additional model-specific parameters.
            """
            self.subject = subject
            self.nest_dir = nest_dir
            self.model = None
            self.selection = selection
            self._validate_parameters()
            
            # Select device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            # Store any additional parameters
            # self.your_param = kwargs.get('your_param', default_value)

        def _validate_parameters(self):
            """
            Validate the input parameters against the model specs.
            """
            if self.subject not in self.VALID_SUBJECTS:
                raise InvalidParameterError(
                    f"Subject must be one of {self.VALID_SUBJECTS}, got {self.subject}"
                )

            # For selection Paramter if available
            if self.selection is not None:
                # Validate selection keys
                validate_selection_keys(self.selection, self.SELECTION_KEYS)

                # Individual validations (example of ROIs)
                if "roi" in self.selection:
                    self.roi = validate_roi(
                        self.selection["roi"], self.VALID_ROIS
                    )
            # Ensure selection is provided
            else:
                raise InvalidParameterError("Parameter 'selection' is required but was not provided")
            
            # Add any other parameter validation here


3.3: Loading the Model
----------------------

Next, implement the ``load_model()`` method, which handles loading model weights and preparing the model for inference.


This method:

1. Constructs the file path to your model weights using a consistent directory structure  
2. Loads the model architecture and weights (implementation will vary based on your model type)  
3. Moves the model to the appropriate device (CPU or GPU)  
4. Sets the model to evaluation mode  
5. Stores the loaded model in a class variable (e.g., ``self.model``) for use by other methods

.. code-block:: python

    def load_model(self) -> None:
        """
        Load model weights and prepare for inference.
        """
        try:
            # Build paths to model weights
            weights_path = os.path.join(
                self.nest_dir,
                'your_path')  # Adjust filename format as needed
            
            # Load your model here
            # Example with PyTorch:
            # self.model = YourModelArchitecture()
            # self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
            # self.model.to(self.device)
            # self.model.eval()
            
            print(f"Model loaded on {self.device} for subject {self.subject}")
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

3.4: Generating Responses
-------------------------

The ``generate_response()`` method is the core functionality that produces in silico neural responses from input stimuli.

This method:

1. Validates the input stimulus to ensure it meets requirements  
2. Preprocesses the stimulus if needed (e.g., normalization, resizing)  
3. Runs the model inference, typically in batches to manage memory usage  
4. Collects and formats the response data  
5. Returns the in silico neural responses as a NumPy array  

Customize this method based on your model's specific requirements and output format.

If you implement the `selection parameter` (`self.selection`) to select specific ROIs or timeintervals, make sure that given those parameters only those models are loaded to save memory and computation time!


.. code-block:: python

    def generate_response(
        self,
        stimulus: np.ndarray,
        **kwargs) -> np.ndarray:
        """
        Generate in silico neural responses for given stimuli.
        
        Parameters
        ----------
        stimulus : np.ndarray
            Input stimulus array. Typically has shape (batch_size, channels, height, width)
            for image stimuli, but requirements vary by model.
        **kwargs
            Additional model-specific parameters for in silico response generation.
        
        Returns
        -------
        np.ndarray
            Simulated in silico neural responses. Shape depends on your model's output.
        """
        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
            raise StimulusError(
                "Stimulus must be a 4D numpy array (batch, channels, height, width)"
            )
        
        # Preprocess stimulus if needed
        preprocessed_stimulus = preprocess(stimulus)
        
        # Generate in silico responses
        with torch.no_grad():
            batch_size = 100  # Adjust as needed
            responses = []
            
            for i in range(0, len(stimulus), batch_size):
                batch = torch.from_numpy(stimulus[i:i+batch_size]).to(self.device)
                output = self.model(batch)
                responses.append(output.cpu().numpy())
            
            all_responses = np.concatenate(responses, axis=0)
        
        return all_responses


3.5: Accessing Metadata
-----------------------

The ``get_metadata()`` method provides information about your encoding model and the shape or structure of its in silico responses.
This might include voxel indices, channel names, ROIs, timepoint definitions, or any other output-relevant detail.

To support metadata access *without having to load the full model*, NEST allows retrieving metadata in two ways:

- **During encoding**:
  ``_, metadata = nest_object.encode(model_id, stimuli, return_metadata=True)``

- **Directly through the NEST API** (without loading the model):
  ``metadata = nest_object.get_model_metadata(model_id, subject=..., roi=...)``

To support this flexibility, you must implement a ``@classmethod get_metadata()`` in your model class.
This method can extract metadata either from a provided model instance or from the input parameters alone.

Below is a template showing the recommended structure.
You can adapt it depending on whether your model uses ROIs, timepoints, or other selection parameters.

This is the most complicated function to implement but you should be able to "blindly" follow this template and just add your missing variables. Feel free to refer to existing models for concrete implementations:

- `fMRI model example <https://github.com/gifale95/NEST/blob/main/nest/models/fmri/nsd_fwrf.py>`_
- `EEG model example <https://github.com/gifale95/NEST/blob/main/nest/models/eeg/things_eeg.py>`_

.. code-block:: python

    @classmethod
    def get_metadata(cls, nest_dir=None, subject=None, model_instance=None, roi=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.

        Parameters
        ----------
        nest_dir : str
            Path to the NEST directory where metadata is stored.
        subject : int
            Subject number.
        model_instance : BaseModelInterface, optional
            If provided, parameters can be extracted directly from the model instance.
        roi : str, optional
            Region of interest (if applicable).
        **kwargs
            Additional model-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """
        
        # Extract parameters from instance if available
        if model_instance is not None:
            nest_dir = model_instance.nest_dir
            subject = model_instance.subject
            roi = getattr(model_instance, "roi", roi)

        # Also allow metadata retrieval from class instance
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            nest_dir = cls.nest_dir
            subject = cls.subject
            roi = getattr(cls, "roi", roi)

        # Validate required parameters
        missing = []
        if nest_dir is None: missing.append("nest_dir")
        if subject is None: missing.append("subject")
        if roi is None and "VALID_ROIS" in dir(cls): missing.append("roi")
        
        if missing:
            raise InvalidParameterError(f"Required parameters missing: {', '.join(missing)}")

        # Optional: validate against allowed values
        validate_subject(subject, cls.VALID_SUBJECTS)
        if roi is not None and hasattr(cls, "VALID_ROIS"):
            validate_roi(roi, cls.VALID_ROIS)

        # Build metadata path
        filename = os.path.join(
            nest_dir,
            "encoding_models",
            "modality-<your_modality>",               # e.g., modality-fmri
            "train_dataset-<your_dataset>",           # e.g., train_dataset-nsd
            "model-<your_model_id>",                  # e.g., model-vit_b_32
            "metadata",
            f"metadata_sub-{subject:02d}" + (f"_roi-{roi}" if roi else "") + ".npy"
        )

        # Load metadata
        if os.path.exists(filename):
            metadata = np.load(filename, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found at: {filename}")

3.6: Auxiliary Methods
----------------------

Finally, implement these required auxiliary methods:

.. code-block:: python

    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique identifier.
        
        Returns
        -------
        str
            Model ID string from the YAML config.
        """
        return cls.MODEL_ID

    def cleanup(self) -> None:
        """
        Release resources (e.g., GPU memory) when finished.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Free GPU memory if using CUDA
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            
            # Clear references
            self.model = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

3.7: Complete Model Implementation
----------------------------------

Here's the complete implementation of a model class.

For more detailed examples, you can refer to existing models:

- `fMRI model example <https://github.com/gifale95/NEST/blob/main/nest/models/fmri/nsd_fwrf.py>`_
- `EEG model example <https://github.com/gifale95/NEST/blob/main/nest/models/eeg/things_eeg.py>`_

.. code-block:: python

    import os
    import numpy as np
    import torch
    import yaml
    from typing import Dict, Any, Optional

    from nest.interfaces.base_model import BaseModelInterface
    from nest.core.model_registry import register_model
    from nest.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError

    # Load model info from YAML
    def load_model_info():
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
        with open(os.path.abspath(yaml_path), "r") as f:
            return yaml.safe_load(f)

    # Load model_info once at the top
    model_info = load_model_info()

    # Register this model with the registry using model_info
    register_model(
        model_id=model_info["model_id"],
        module_path="nest.models.your_modality.your_model_file",  # Replace with actual path
        class_name="YourModelClass",
        modality=model_info.get("modality", "your_modality"),
        training_dataset=model_info.get("training_dataset", "your_dataset"),
        yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
    )




    class YourModelClass(BaseModelInterface):
        """
        Your model description here. Explain what this model does, what
        in silico neural responses it generates, and any other important details.
        """

        MODEL_ID = model_info["model_id"]
        # Extract any validation info from model_info
        VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]

        def __init__(self, subject: int, selection: Dict, device: str = "auto", nest_dir: Optional[str] = None, **kwargs):
            """
            Initialize your model with the required parameters.

            Parameters
            ----------
            subject : int
                Subject ID for subject-specific models.
            selection : dict
                Specifies which outputs to include in the model responses
                (ROI, Time interval, ...)
            device : str
                Device to run the model on ('cpu', 'cuda', or 'auto').
            nest_dir : str, optional
                Path to the NEST directory.
            **kwargs
                Additional model-specific parameters.
            """
            self.subject = subject
            self.nest_dir = nest_dir
            self.model = None
            self.selection = selection
            self._validate_parameters()

            # Select device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            # Store any additional parameters
            # self.your_param = kwargs.get('your_param', default_value)

        def _validate_parameters(self):
            """
            Validate the input parameters against the model specs.
            """
            if self.subject not in self.VALID_SUBJECTS:
                raise InvalidParameterError(
                    f"Subject must be one of {self.VALID_SUBJECTS}, got {self.subject}"
                )

            # For selection Paramter if available
            if self.selection is not None:
                # Validate selection keys
                validate_selection_keys(self.selection, self.SELECTION_KEYS)

                # Individual validations (example of ROIs)
                if "roi" in self.selection:
                    self.roi = validate_roi(
                        self.selection["roi"], self.VALID_ROIS
                    )
            # Ensure selection is provided
            else:
                raise InvalidParameterError("Parameter 'selection' is required but was not provided")

            # Add any other parameter validation here

        def load_model(self) -> None:
            """
            Load model weights and prepare for inference.
            """
            try:
                # Build paths to model weights
                weights_path = os.path.join(
                    self.nest_dir,
                    'your_path') # Adjust filename format as needed

                # Load your model here
                # Example with PyTorch:
                # self.model = YourModelArchitecture()
                # self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
                # self.model.to(self.device)
                # self.model.eval()

                print(f"Model loaded on {self.device} for subject {self.subject}")

            except Exception as e:
                raise ModelLoadError(f"Failed to load model: {str(e)}")

        def generate_response(
            self,
            stimulus: np.ndarray,
            **kwargs) -> np.ndarray:
            """
            Generate in silico neural responses for given stimuli.

            Parameters
            ----------
            stimulus : np.ndarray
                Input stimulus array. Typically has shape (batch_size, channels, height, width)
                for image stimuli, but requirements vary by model.
            **kwargs
                Additional model-specific parameters for in silico response generation.

            Returns
            -------
            np.ndarray
                Simulated in silico neural responses. Shape depends on your model's output.
            """
            # Validate stimulus
            if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
                raise StimulusError(
                    "Stimulus must be a 4D numpy array (batch, channels, height, width)"
                )

            # Preprocess stimulus if needed
            preprocessed_stimulus = preprocess(stimulus)

            # Generate in silico responses
            with torch.no_grad():
                batch_size = 100  # Adjust as needed
                responses = []

                for i in range(0, len(stimulus), batch_size):
                    batch = torch.from_numpy(stimulus[i:i+batch_size]).to(self.device)
                    output = self.model(batch)
                    responses.append(output.cpu().numpy())

                all_responses = np.concatenate(responses, axis=0)

            return all_responses



        @classmethod
        def get_metadata(cls, nest_dir=None, subject=None, model_instance=None, roi=None, **kwargs) -> Dict[str, Any]:
            """
            Retrieve metadata for the model.

            Parameters
            ----------
            nest_dir : str
                Path to the NEST directory where metadata is stored.
            subject : int
                Subject number.
            model_instance : BaseModelInterface, optional
                If provided, parameters can be extracted directly from the model instance.
            roi : str, optional
                Region of interest (if applicable).
            **kwargs
                Additional model-specific parameters.

            Returns
            -------
            Dict[str, Any]
                Metadata dictionary.
            """

            # Extract parameters from instance if available
            if model_instance is not None:
                nest_dir = model_instance.nest_dir
                subject = model_instance.subject
                roi = getattr(model_instance, "roi", roi)

            # Also allow metadata retrieval from class instance
            elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
                nest_dir = cls.nest_dir
                subject = cls.subject
                roi = getattr(cls, "roi", roi)

            # Validate required parameters
            missing = []
            if nest_dir is None: missing.append("nest_dir")
            if subject is None: missing.append("subject")
            if roi is None and "VALID_ROIS" in dir(cls): missing.append("roi")

            if missing:
                raise InvalidParameterError(f"Required parameters missing: {', '.join(missing)}")

            # Optional: validate against allowed values
            validate_subject(subject, cls.VALID_SUBJECTS)
            if roi is not None and hasattr(cls, "VALID_ROIS"):
                validate_roi(roi, cls.VALID_ROIS)

            # Build metadata path
            filename = os.path.join(
                nest_dir,
                "encoding_models",
                "modality-<your_modality>",               # e.g., modality-fmri
                "train_dataset-<your_dataset>",           # e.g., train_dataset-nsd
                "model-<your_model_id>",                  # e.g., model-vit_b_32
                "metadata",
                f"metadata_sub-{subject:02d}" + (f"_roi-{roi}" if roi else "") + ".npy"
            )

            # Load metadata
            if os.path.exists(filename):
                metadata = np.load(filename, allow_pickle=True).item()
                return metadata
            else:
                raise FileNotFoundError(f"Metadata file not found at: {filename}")

        @classmethod
        def get_model_id(cls) -> str:
            """
            Return the model's unique identifier.

            Returns
            -------
            str
                Model ID string from the YAML config.
            """
            return cls.MODEL_ID

        def cleanup(self) -> None:
            """
            Release resources (e.g., GPU memory) when finished.
            """
            if hasattr(self, 'model') and self.model is not None:
                # Free GPU memory if using CUDA
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')

                # Clear references
                self.model = None

                # Force CUDA cache clear if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

Step 4: Adding a New Modality
=============================

To extend NEST with a new recording modality (e.g., MEG), follow these steps:

1. Create a Folder
------------------
Create a new directory under ``nest/models/``:

.. code-block:: text

    nest/models/your_modality/

2. Add Your Model Files
-----------------------
Inside the new folder, include:

- ``your_model.py`` — your model implementation.
- ``__init__.py`` — register your model by adding:

  .. code-block:: python

      import nest.models.your_modality.your_model

3. Add a Model Card
------------------
Create a YAML configuration file for your model and place it in:

.. code-block:: text

    nest/models/model_cards/your_model_id.yaml

4. Specify the Modality
----------------------
In both ``your_model.py`` and the YAML config file, define the modality name. For example:

.. code-block:: yaml

    modality: "your_modality"

5. Register the Modality
-----------------------
Finally, update ``nest/models/__init__.py`` to ensure your modality is loaded:

.. code-block:: python

    import nest.models.your_modality

Final Directory Structure
------------------------

.. code-block:: text

    nest/
    ├── models/
    │   ├── __init__.py
    │   ├── fmri/
    │   ├── eeg/
    │   ├── your_modality/
    │   │   ├── __init__.py
    │   │   └── your_model.py
    │   └── model_cards/
    │       └── your_model_id.yaml

Contributing to NEST
===================

We warmly welcome all contributions to the NEST toolbox and are happy for every addition that helps grow the community.

Code Quality
-----------
- Include clear **docstrings** for all public methods.
- Add **type hints** to improve code readability.
- Implement **robust error handling** with informative messages.
- Follow existing **NEST naming conventions**.
- Be thorough with your **YAML configuration** and include as much relevant information as possible.
- If available, feel free to add **performance details**.

Testing
-------
- Test your model with various **input shapes** and **data types**.
- Verify that **error handling** works as expected.
- Check **resource usage** during and after model execution.
- Ensure all required **metadata** is correctly provided.

How to Contribute
---------------

If you would like to contribute your model back to NEST:

1. **Fork** the NEST repository.
2. **Create a branch** from the ``development`` branch.
3. **Add your model** following this tutorial.
4. **Submit a pull request** with:
   - A clear description of your model.
   - Example code showing how to run your model.
   - Any relevant **citations** or **references**.

We look forward to your contributions and are excited to see the creative ways the community expands NEST!



.. _uploading_model_weights:

Uploading Your Model Weights
===========================

After implementing and testing your model, the final step is to upload the trained weights and associated metadata so that others can use your model through NEST.

Step 1: Follow the NEST Directory Structure
------------------------------------------

Please organize your files following the official NEST dataset structure, as described in the `NEST documentation <https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/data_storage.html#nest-dataset-structure>`_:

.. code-block:: text

    neural-encoding-simulation-toolkit/
    ├── encoding_models/
    │   ├── modality-{modality}/
    │   │   ├── train_dataset-{dataset}/
    │   │   │   └── model-{model}/
    │   │   │       ├── encoding_models_accuracy/
    │   │   │       ├── encoding_models_weights/
    │   │   │       └── metadata/
    └── nest_tutorials/

Replace ``{modality}``, ``{dataset}``, and ``{model}`` with the appropriate values (e.g., ``modality-fmri``, ``train_dataset-nsd``, ``model-fwrf``).

Each model directory must contain the following subfolders:

- ``encoding_models_weights/``: your model weights (e.g., ``.pth``, ``.npz``, etc.)
- ``encoding_models_accuracy/``: performance metrics or evaluation results
- ``metadata/``: precomputed metadata files returned by your ``get_metadata()`` method

Step 2: Create a Zip Archive
---------------------------

Once the directory is correctly structured, compress the entire ``neural-encoding-simulation-toolkit/`` folder into a ``.zip`` archive.

Step 3: Upload to a Cloud Service
--------------------------------

Upload the ``.zip`` file to a cloud storage provider that provides a **public direct download link**. Make sure that access permissions are set to **public or viewable by link**.

Step 4: Submit a Pull Request
----------------------------

Include the public link to your ``.zip`` archive in your pull request.
For detailed instructions on contributing to NEST, please refer to the official guide: `How to contribute <https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/contribution.html>`_



Citation
========

If you use the code and/or data from this tutorial, please cite:

    *Gifford AT, Bersch D, Roig G, Cichy RM. 2025. The Neural Encoding Simulation Toolkit. In preparation. https://github.com/gifale95/NEST*
