
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import textwrap
from tqdm import tqdm
import os
import yaml
from nest.core.model_registry import MODEL_REGISTRY

class BaseModelInterface(ABC):
    @abstractmethod
    def load_model(self, device:str) -> None:
        """
        Load the model weights and prepare for inference
        Args:
            device: Target device ("cpu", "cuda")
        """
        pass
    
    
    @abstractmethod
    def generate_response(
        self, 
        stimulus: np.ndarray) -> np.ndarray:
        """
        Generate in silico neural responses for the given stimulus.
        
        Args:
            stimulus: Input stimulus array
        Returns:
            Neural responses as numpy array
        """
        pass
    
    
    def get_supported_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return information about supported parameters.
        
        Returns:
            Dict mapping parameter names to parameter metadata
        """
        model_id = self.get_model_id()
        
        # TODO: Make it version flexible
        yaml_path = MODEL_REGISTRY[model_id]["1.0.0"]["yaml_path"]

        # Load YAML metadata
        with open(os.path.abspath(yaml_path), "r") as f:
            metadata = yaml.safe_load(f)
            
        return metadata["supported_parameters"]
        
    
    @classmethod
    @abstractmethod
    def get_model_id(cls) -> str:
        """
        Return the unique identifier for this model.
        
        Returns:
            Model ID string
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (e.g., free GPU memory, close sessions)."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        "Retrieve Metadata from each of the models"
        pass
    
    @staticmethod
    def describe(model_id: str) -> Dict[str, Any]:
        """
        Return a detailed, human-readable description of the model
        using only the YAML metadata registered under model_id.
        
        Args:
            model_id: Unique model ID registered via register_model()
        
        Returns:
            Dict containing metadata and example usage
        """
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_id}' is not registered.")

        # TODO: Make it version flexible
        yaml_path = MODEL_REGISTRY[model_id]["1.0.0"]["yaml_path"]

        # Load YAML metadata
        with open(os.path.abspath(yaml_path), "r") as f:
            metadata = yaml.safe_load(f)
        
        parameters = metadata.get("supported_parameters", {})

        # Example parameters
        param_example_dict = {}
        for name, info in parameters.items():
            if "example" in info:
                param_example_dict[name] = info["example"]
            elif "valid_values" in info and info["valid_values"]:
                param_example_dict[name] = info["valid_values"][0]
            else:
                param_example_dict[name] = "..."

        param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_example_dict.items())

        example_code = textwrap.dedent(f"""\
            from nest import NEST

            nest = NEST("/path/to/nest_dir")
            model = nest.get_encoding_model("{model_id}", {param_str})

            # Generate responses (assuming stimulus is a numpy array)
            responses = model.generate_response(stimulus)
        """)

        # Pretty print
        print("=" * 60)
        print(f"🧠 Model: {model_id}")
        print("-" * 60)

        for key in ["modality", "dataset", "features", "repeats", "subject_level"]:
            if key in metadata:
                label = key.replace("_", " ").capitalize()
                print(f"{label}: {metadata[key]}")

        print("\n📌 Supported Parameters:")
        for name, info in parameters.items():
            desc = info.get("description", "")
            example = info.get("example", "...")
            valid = info.get("valid_values", None)
            required = info.get("required", True)

            print(f"\n• {name} ({info.get('type', 'unknown')}, {'required' if required else 'optional'})")
            if desc:
                print(f"  ↳ {desc}")
            if valid:
                print(f"  ↳ Valid values: {valid}")
            print(f"  ↳ Example: {example}")

        print("\n📦 Example Usage:\n")
        print(example_code)
        print("=" * 60)

        return {
            "model_id": model_id,
            "metadata": {k: metadata.get(k) for k in ["modality", "dataset", "features", "repeats", "subject_level"]},
            "supported_parameters": parameters,
            "example_usage": example_code.strip()
        }

            
    def __enter__(self):
        return self
        
    def __exit__(self):
        self.cleanup()