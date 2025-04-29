=======================
Adding Models to NEST
=======================

This guide walks you through the process of adding new models to the Neural Encoding Simulation Toolkit (NEST). You can also execute this `tutorial on Google Colab <https://colab.research.google.com/drive/1nBxEiJATzJdWwfzRPmyai2G76HkeBhAU>`_.


Overview
=========

Adding a new model to NEST requires creating two main files:

1. **A Python model file** — implements the model's logic and interface
2. **A YAML configuration file** — describes the model's parameters and behavior

These two files work together to register your model with NEST and make it accessible through the unified interface.

Quick Start: Model Implementation Template
===========================================

Here's a barebones version of a Python model file that shows only the required structure.  
We will expand on this template throughout the tutorial to build a fully functional model.

You can use this as a starting point and fill in the details specific to your model.

.. code-block:: python

    from nest.interfaces.base_model import BaseModelInterface
    from nest.core.model_registry import register_model

    # Load model info from YAML
    def load_model_info():
        path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    model_info = load_model_info()

    register_model(
        model_id=model_info["model_id"],
        module_path="nest.models.your_modality.your_model_file",
        class_name="YourModelClass",
        modality=model_info["modality"],
        dataset=model_info["dataset"],
        yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
    )

    class YourModelClass(BaseModelInterface):
        def __init__(self, subject, device="auto", nest_dir=None, **kwargs):
            # Set up model parameters and device
            pass

        def load_model(self):
            # Load model weights and prepare for inference
            pass

        def generate_response(self, stimulus, **kwargs):
            # Generate simulated neural responses
            pass

        def get_metadata(self):
            # Return model metadata
            pass

        @classmethod
        def get_model_id(cls):
            # Return the model ID string
            pass

        def cleanup(self):
            # Free up any resources (e.g., GPU memory)
            pass

Step 1: Determining Your Model's Location
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
       - What kind of neural responses it generates
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
      # First parameter (typically subject)
      param_name:
        type: param_type  # e.g., int, str, float
        required: true/false
        valid_values: list_of_valid_values  # or range, or omit if not applicable
        default: default_value  # include if there's a default value
        example: example_value
        description: "Description of what this parameter represents"
        function: "Which function uses this parameter: get_encoding_model, load_model, .."
      
      # Add more parameters as needed
      param_name:
        type: param_type
        required: true/false
        valid_values: list_of_valid_values  # or range, or omit if not applicable
        default: default_value  # include if there's a default value
        example: example_value
        description: "Description of what this parameter represents"
        function: "Which function uses this parameter"


      # Selection parameter to define specific outputs (ROI, channels, timepoints, etc.)
      selection:
        type: dict
        required: true
        description: |
        Specifies which outputs to include in the model responses.
        This parameter defines for which data the in silico responses should be generated 
        (e.g., specific ROI, timepoints, channels, etc.)
        function: get_encoding_model
        properties:
        key_name:  # Replace with model-specific keys, e.g., "roi", "channels", "timepoints"
            type: any
            description: "Description of Model-specific selection criterion."
            example: "V1"

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
        - "Citation for your model or dataset"

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
        dataset=model_info.get("dataset", "your_dataset"),
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

.. code-block:: python

    class YourModelClass(BaseModelInterface):
        """
        Your model description here. Explain what this model does, what
        neural responses it generates, and any other important details.
        """
        
        MODEL_ID = model_info["model_id"]
        # Extract any validation info from model_info
        VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
        
        def __init__(self, subject: int, device: str = "auto", nest_dir: Optional[str] = None, **kwargs):
            """
            Initialize your model with the required parameters.
            
            Parameters
            ----------
            subject : int
                Subject ID for subject-specific models.
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
            Additional model-specific parameters for response generation.
        
        Returns
        -------
        np.ndarray
            Simulated neural responses. Shape depends on your model's output.
        """
        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
            raise StimulusError(
                "Stimulus must be a 4D numpy array (batch, channels, height, width)"
            )
        
        # Preprocess stimulus if needed
        # preprocessed_stimulus = preprocess(stimulus)
        
        # Generate responses
        # with torch.no_grad():
        #     batch_size = 100  # Adjust as needed
        #     responses = []
        #     
        #     for i in range(0, len(stimulus), batch_size):
        #         batch = torch.from_numpy(stimulus[i:i+batch_size]).to(self.device)
        #         output = self.model(batch)
        #         responses.append(output.cpu().numpy())
        #     
        #     all_responses = np.concatenate(responses, axis=0)
        
        # For now, return dummy data with expected shape
        # Replace this with your actual model inference
        dummy_response = np.zeros((stimulus.shape[0], 100))  # Example shape
        
        return dummy_response

3.5: Accessing Metadata
-----------------------

The ``get_metadata()`` method provides information about the model and its outputs:

This method:

1. Attempts to load metadata from a predefined location  
2. Returns the metadata as a dictionary  
3. Provides basic information if no metadata file is found  

The metadata may include information about voxel indices, channel information, region details, or other model-specific information.


The ``get_metadata()`` method is a versatile function that provides information about your model and its outputs. This method is designed to be flexible, allowing it to be called in three different contexts:

1. **Class method with explicit parameters**: When users want metadata without initializing the model
2. **Instance method**: When called on an already initialized model
3. **During encoding**: When users request metadata alongside model responses

The metadata may include information about voxel indices, channel information, region details, or other model-specific information.

This function is a bit more complicated because it needs to handle **all three scenarios**.
However, we tried to make the code snippet below as understandable as possible so that you just need to fill in all the missing information and paste it into your function! Always feel free to check out the existing implementations for reference.


.. code-block:: python

    @classmethod
    def get_metadata(cls, nest_dir=None, subject=None, model_instance=None, 
                    # Add any model-specific parameters here (e.g., roi=None)
                    **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        nest_dir : str, optional
            Path to NEST directory.
        subject : int, optional
            Subject number.
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        # Document any model-specific parameters here
        **kwargs
            Additional parameters.
                
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """
        # STEP 1: Detect calling context and extract parameters
        # If model_instance is provided, extract parameters from it
        if model_instance is not None:
            nest_dir = model_instance.nest_dir
            subject = model_instance.subject
            # Extract any model-specific parameters you need
            # For example: roi = model_instance.roi
        
        # If this method is called on an instance (rather than the class)
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            # 'cls' is actually an instance in this case
            nest_dir = cls.nest_dir
            subject = cls.subject
            # Extract any model-specific attributes
            # For example: roi = cls.roi
        
        # STEP 2: Validate required parameters
        missing_params = []
        if nest_dir is None: missing_params.append('nest_dir')
        if subject is None: missing_params.append('subject')
        # Add checks for any other required parameters
        # For example: if roi is None: missing_params.append('roi')
        
        if missing_params:
            raise InvalidParameterError(f"Required parameters missing: {', '.join(missing_params)}")
        
        # STEP 3: Validate parameter values
        validate_subject(subject, cls.VALID_SUBJECTS)
        # Add validation for any other parameters
        # For example: validate_roi(roi, cls.VALID_ROIS)
        
        # STEP 4: Build metadata file path
        # CUSTOMIZE THIS PATH for your specific model
        file_name = os.path.join(nest_dir, 
                                'encoding_models', 
                                'modality-YOUR_MODALITY',  # Replace with your modality
                                'train_dataset-YOUR_DATASET',  # Replace with your dataset
                                'model-YOUR_MODEL_NAME',  # Replace with your model name
                                'metadata',
                                f'metadata_sub-{subject:02d}.npy')  # Customize filename format
        
        # STEP 5: Load and return metadata
        if os.path.exists(file_name):
            metadata = np.load(file_name, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found: {file_name}")

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

Here's the complete implementation of a model class:

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
        dataset=model_info.get("dataset", "your_dataset"),
        yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "your_model_id.yaml")
    )


    class YourModelClass(BaseModelInterface):
        """
        Your model description here. Explain what this model does, what
        neural responses it generates, and any other important details.
        """

        MODEL_ID = model_info["model_id"]
        # Extract any validation info from model_info
        VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]

        def __init__(self, subject: int, device: str = "auto", nest_dir: Optional[str] = None, **kwargs):
            """
            Initialize your model with the required parameters.

            Parameters
            ----------
            subject : int
                Subject ID for subject-specific models.
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
                Additional model-specific parameters for response generation.

            Returns
            -------
            np.ndarray
                Simulated neural responses. Shape depends on your model's output.
            """
            # Validate stimulus
            if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
                raise StimulusError(
                    "Stimulus must be a 4D numpy array (batch, channels, height, width)"
                )

            # Preprocess stimulus if needed
            # preprocessed_stimulus = preprocess(stimulus)

            # Generate responses
            # with torch.no_grad():
            #     batch_size = 100  # Adjust as needed
            #     responses = []
            #
            #     for i in range(0, len(stimulus), batch_size):
            #         batch = torch.from_numpy(stimulus[i:i+batch_size]).to(self.device)
            #         output = self.model(batch)
            #         responses.append(output.cpu().numpy())
            #
            #     all_responses = np.concatenate(responses, axis=0)

            # For now, return dummy data with expected shape
            # Replace this with your actual model inference
            dummy_response = np.zeros((stimulus.shape[0], 100))  # Example shape

            return dummy_response

        def get_metadata(self) -> Dict[str, Any]:
            """
            Return metadata about the model and its outputs.

            Returns
            -------
            Dict[str, Any]
                Dictionary containing model metadata.
            """
            # Load metadata file if available
            metadata_path = os.path.join(
                    self.nest_dir,
                    'your_path') # Adjust filename format as needed

            try:
                metadata = np.load(metadata_path, allow_pickle=True).item()
                return metadata
            except Exception as e:
                # If no metadata file exists, return basic info
                return {
                    "model_id": self.MODEL_ID,
                    "subject": self.subject,
                    # Add any other relevant metadata
                }

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

Citation
========

If you use the code and/or data from this tutorial, please cite:

    *Gifford AT, Bersch D, Roig G, Cichy RM. 2025. The Neural Encoding Simulation Toolkit. In preparation. https://github.com/gifale95/NEST*