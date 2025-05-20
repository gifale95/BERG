
import os
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import yaml
from nest.core.model_registry import MODEL_REGISTRY


class BaseModelInterface(ABC):
    """
    Abstract base class for all NEST encoding models.
    Defines required methods for loading, running, and describing models.
    """
    
    @abstractmethod
    def load_model(self, device:str) -> None:
        """
        Load model weights and prepare for inference.
        
        Parameters
        ----------
        device : str
            Target device for computation ("cpu", "cuda", or "auto").
            If "auto", the system will use GPU acceleration if available,
            otherwise fall back to CPU.
        """
        pass
    
    def get_supported_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get supported input parameters defined in the model's YAML file.
    
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping parameter names to their model info.
            Each parameter entry contains information such as type,
            valid values, default values, and descriptions.
        """
        model_id = self.get_model_id()
        yaml_path = MODEL_REGISTRY[model_id]["yaml_path"]

        # Load YAML model_info
        with open(os.path.abspath(yaml_path), "r") as f:
            model_info = yaml.safe_load(f)
            
        return model_info["supported_parameters"]
    
    @classmethod
    @abstractmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique identifier.
        
        Returns
        -------
        str
            The unique string identifier for this model as registered
            in the model registry.
        """
        pass
        
        
    @classmethod
    @abstractmethod
    def get_metadata(cls, nest_dir=None, subject=None, model_instance=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        This method can be called in three ways:
        1. As a class method with explicit parameters: ModelClass.get_metadata(nest_dir, subject, ...)
        2. On a model instance: model.get_metadata()
        3. With a model_instance parameter: ModelClass.get_metadata(model_instance=model)
        
        Parameters
        ----------
        nest_dir : str, optional
            Path to NEST directory.
        subject : int, optional
            Subject number.
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        **kwargs
            Additional model-specific parameters.
                
        Returns
        -------
        Dict[str, Any]
            Model metadata dictionary.
        """
        pass
    
    
    @abstractmethod
    def generate_response(
        self, 
        stimulus: np.ndarray) -> np.ndarray:
        """
        Generate in silico neural responses for a given stimulus.
        
        Parameters
        ----------
        stimulus : np.ndarray
            Input stimulus array - requirements vary by model.
            See model's YAML model info for specific input constraints.
        
        Returns
        -------
        np.ndarray
            Simulated neural responses. Shape varies by modality
        """
        pass

    
    @staticmethod
    def describe_from_id(model_id: str) -> Dict[str, Any]:
        """
        Print and return a detailed description of a registered model.
        
        Parameters
        ----------
        model_id : str
            ID of the model as registered in the registry.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive model information
        """
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_id}' is not registered.")
            
        yaml_path = MODEL_REGISTRY[model_id]["yaml_path"]
        
        # Load YAML model Info
        with open(os.path.abspath(yaml_path), "r") as f:
            model_info = yaml.safe_load(f)
        
        # Get model parameters
        parameters = model_info.get("parameters", {})
        
        # Group parameters by function
        params_by_function = {}
        for name, info in parameters.items():
            func_name = info.get("function", "other")
            if func_name not in params_by_function:
                params_by_function[func_name] = {}
            params_by_function[func_name][name] = info
        
        # Generate example parameters for get_encoding_model
        init_example_dict = {}
        if "get_encoding_model" in params_by_function:
            for name, info in params_by_function["get_encoding_model"].items():
                if name == "selection":
                    # Build example selection block from its properties
                    selection_example = {}
                    for subname, subinfo in info.get("properties", {}).items():
                        if "example" in subinfo:
                            selection_example[subname] = subinfo["example"]
                        elif "valid_values" in subinfo and subinfo["valid_values"]:
                            selection_example[subname] = subinfo["valid_values"][0]
                        else:
                            selection_example[subname] = "..."
                    init_example_dict["selection"] = selection_example
                elif "example" in info:
                    init_example_dict[name] = info["example"]
                elif "valid_values" in info and info["valid_values"]:
                    init_example_dict[name] = info["valid_values"][0]
                elif "default" in info:
                    init_example_dict[name] = info["default"]
                else:
                    init_example_dict[name] = "..."
        
        init_param_str = ", ".join(f"{k}={repr(v)}" for k, v in init_example_dict.items())
        
        # Generate example code
        example_code = textwrap.dedent(f"""\
            from nest import NEST

            nest = NEST("path/to/neural_encoding_simulation_toolkit")

            # Initialize the model
            model = nest.get_encoding_model("{model_id}", {init_param_str})

            # Generate responses (assuming stimulus is a numpy array)
            responses = model.generate_response(stimulus)
        """)
        
        # Pretty print
        print("=" * 80)
        print(f"🧠 Model: {model_id}")
        print("=" * 80)
        print()
        
        # Print basic model_info
        for key in ["modality", "training_dataset", "model_architecture", "creator"]:
            if key in model_info:
                label = key.replace("_", " ").capitalize()
                print(f"{label}: {model_info[key]}")
        
        # Print description if available
        if "description" in model_info:
            print("\n📋 Description:")
            print(textwrap.fill(model_info["description"], width=80))
        
        # Print input information
        if "input" in model_info:
            print("\n📥 Input:")
            input_info = model_info["input"]
            print(f"  Type: {input_info.get('type', 'Not specified')}")
            print(f"  Shape: {input_info.get('shape', 'Not specified')}")
            if "description" in input_info:
                print(f"  Description: {input_info['description']}")
            if "constraints" in input_info:
                print(f"  Constraints:")
                for constraint in input_info["constraints"]:
                    print(f"    • {constraint}")
        
        # Print output information
        if "output" in model_info:
            print("\n📤 Output:")
            output_info = model_info["output"]
            print(f"  Type: {output_info.get('type', 'Not specified')}")
            print(f"  Shape: {output_info.get('shape', 'Not specified')}")
            if "description" in output_info:
                print(f"  Description: {output_info['description']}")
            if "dimensions" in output_info:
                print(f"  Dimensions:")
                for dim in output_info["dimensions"]:
                    print(f"    • {dim['name']}: {dim['description']}")
        
        # Print parameters by function
        for func_name, func_params in sorted(params_by_function.items()):
            if func_name == "other":
                print("\n📌 Other Parameters:")
            else:
                print(f"\n📌 Parameters for {func_name}():")
            
            for name, info in func_params.items():
                desc = info.get("description", "")
                example = info.get("example", info.get("default", "..."))
                valid = info.get("valid_values", None)
                required = info.get("required", True)
                default = info.get("default", None)
                param_type = info.get("type", "unknown")

                req_str = "required" if required else "optional"
                if not required and default is not None:
                    req_str = f"optional, default={repr(default)}"

                print(f"\n• {name} ({param_type}, {req_str})")
                if desc:
                    print(textwrap.fill(f"  ↳ {desc}", width=80, subsequent_indent="    "))
                if valid:
                    if isinstance(valid, list) and len(valid) > 10:
                        valid_str = str(valid)[1:]
                        print(f"  ↳ Valid values: {valid_str}")
                    else:
                        print(f"  ↳ Valid values: {valid}")

                # 📦 Handle nested properties for dict-type params (like 'selection')
                if param_type == "dict" and "properties" in info:
                    print("\n  ↪ Sub-parameters within 'selection':")
                    for subname, subinfo in info["properties"].items():
                        subdesc = subinfo.get("description", "")
                        subexample = subinfo.get("example", "...")
                        subvalid = subinfo.get("valid_values", None)
                        subtype = subinfo.get("type", "unknown")

                        print(f"\n    • {subname} ({subtype})")
                        if subdesc:
                            print(textwrap.fill(f"      ↳ {subdesc}", width=80, subsequent_indent="        "))
                        if subvalid:
                            if isinstance(subvalid, list) and len(subvalid) > 10:
                                subvalid_str = str(subvalid)[1:]
                                print(f"      ↳ Valid values: {subvalid_str}")
                            else:
                                print(f"      ↳ Valid values: {subvalid}")
                        print(f"      ↳ Example: {subexample}")
        
        # Print performance information if available
        if "performance" in model_info:
            perf = model_info["performance"]
            print("\n📊 Performance:")
            
            if "metrics" in perf:
                for metric in perf["metrics"]:
                    print(f"  • {metric['name']}: {metric['value']}")
                    if "description" in metric:
                        print(f"    ↳ {metric['description']}")
            
            if "accuracy_plots" in perf:
                print(f"\n  Performance plots at: {perf['accuracy_plots'][0]}")
                
        if "references" in model_info:
            ref = model_info["references"]
            print("\n📚 References:")
            
            for ref in model_info["references"]:
                print(textwrap.fill(f"    • {ref}", width=80, subsequent_indent="      "))
        
        # Print example usage
        print("\n📦 Example Usage:\n")
        print(example_code)
        print("=" * 80)
        
        # Return structured information
        return {
            "model_id": model_id,
            "model_info": {k: model_info.get(k) for k in ["modality", "training_dataset", "model_architecture", "creator"]},
            "description": model_info.get("description", ""),
            "input": model_info.get("input", {}),
            "output": model_info.get("output", {}),
            "parameters": parameters,
            "parameters_by_function": params_by_function,
            "performance": model_info.get("performance", {}),
            "example_usage": example_code.strip()
        }

        
    def describe(self) -> Dict[str, Any]:
        """
        Print and return a detailed description of this model instance.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive model information.
        """
        return self.__class__.describe_from_id(self.get_model_id())
    
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources, such as GPU memory or open sessions.
        """
        pass

    def __enter__(self):
        """
        Enable use of the model in a context manager (`with` statement).
        
        Returns:
            BaseModelInterface: The current model instance.
        """
        return self
        
    def __exit__(self):
        """
        Automatically clean up resources when leaving a context.
        """
        self.cleanup()