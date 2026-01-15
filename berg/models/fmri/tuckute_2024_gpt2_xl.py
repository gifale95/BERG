import os
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from berg.models.fmri.tuckute_2024.load_regr_weights_and_predict import ANNEncoder, BrainEncoder, Metric, Mapping
from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model
from berg.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from berg.core.parameter_validator import (
    validate_selection_keys,
    validate_roi,
)


def load_model_info():
    """Load model information from YAML file."""
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "fmri-tuckute_2024-GPT2_XL.yaml"
    )
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


model_info = load_model_info()

register_model(
    model_id=model_info["model_id"],
    module_path="berg.models.fmri.tuckute_2024_gpt2_xl",
    class_name="FMRITextEncodingModel",
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "tuckute_2024"),
    yaml_path=os.path.join(
        os.path.dirname(__file__),
        "..",
        "model_cards",
        "fmri-tuckute_2024-GPT2_XL.yaml"
    )
)


class FMRITextEncodingModel(BaseModelInterface):
    """
    fMRI text encoding model using GPT2-XL embeddings and ridge regression
    for predicting left-hemisphere language network responses.
    
    Based on Tuckute et al. (2024) - uses GPT2-XL layer 22 representations
    mapped to BOLD responses via linear ridge regression.
    """
    
    MODEL_ID = model_info["model_id"]
    SELECTION_KEYS = list(model_info["parameters"]["selection"]["properties"].keys())
    VALID_ROIS = model_info["parameters"]["selection"]["properties"]["roi"]["valid_values"]
    
    # Model configuration (hardcoded parameters specific to this pre-trained model)
    SENT_EMBED = 'last-tok'
    SOURCE_MODEL = 'gpt2-xl'
    SOURCE_LAYER = 22
    METRIC = 'pearsonr'
    MAPPING_CLASS = 'ridgeCV'
    
    # Weight file name
    WEIGHT_FILENAME = 'mapping-full_SOURCE-22_TARGET-20221214a-None_d-swr-5-0.05-bySessVoxZ_MAPPING-None-False-False.pkl'
    
    def __init__(
        self,
        selection: Optional[Dict] = None,
        device: str = "cpu",
        berg_dir: Optional[str] = None
    ):
        """
        Initialize the fMRI text encoding model.
        
        Parameters
        ----------
        selection : dict, optional
            Specifies which ROIs to include in the model responses.
            - roi: List of language network ROI names (e.g., ['lang_LH_IFG', 'lang_LH_netw'])
        device : str, default="cpu"
            Device parameter (included for API consistency, but this model always runs on CPU).
        berg_dir : str, optional
            Path to the BERG directory containing model weights and metadata files.
        """
        self.berg_dir = berg_dir
        self.ann_encoder = None
        self.brain_encoder = None
        self.mapping = None
        self.metadata = None
        self._warning_shown = False
        
        # Parameters from selection
        self.selection = selection
        self.roi_list = None
        
        # Validate parameters
        self._validate_parameters()
        
        # This model always runs on CPU
        self.device = "cpu"
        if device != "cpu":
            print(f"Note: This model runs on CPU only. Ignoring device='{device}'")
        
    def _validate_parameters(self):
        """Validate the selection parameters."""
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)
            
            # Validate ROI selection
            if "roi" in self.selection:
                roi_list = self.selection["roi"]
                if not isinstance(roi_list, list):
                    raise InvalidParameterError(
                        f"Parameter 'roi' must be a list, got {type(roi_list)}"
                    )
                for roi in roi_list:
                    validate_roi(roi, self.VALID_ROIS)
                self.roi_list = roi_list
    
    def load_model(self, device: str = "cpu") -> None:
        """
        Load the pre-trained model weights and initialize the encoders.
        
        Parameters
        ----------
        device : str, default="cpu"
            Device parameter (included for API consistency, always uses CPU).
        """
        if self.berg_dir is None:
            raise InvalidParameterError(
                "berg_dir must be provided to load model weights"
            )
        
        # Build weight file path
        weight_dir = os.path.join(
            self.berg_dir,
            'encoding_models',
            'modality-fmri',
            'train_dataset-tuckute_2024',
            'model-GPT2_XL',
            'encoding_models_weights'
        )
        
        weight_path = os.path.join(weight_dir, self.WEIGHT_FILENAME)
        
        if not os.path.exists(weight_path):
            raise ModelLoadError(
                f"Model weights not found at: {weight_path}"
            )
        
        # Initialize ANN Encoder
        self.ann_encoder = ANNEncoder(
            source_model=self.SOURCE_MODEL,
            sent_embed=self.SENT_EMBED,
            actv_cache_setting=None,
            actv_cache_path=None
        )
        
        # Initialize Brain Encoder
        self.brain_encoder = BrainEncoder()
        
        # Load metadata
        self.metadata = self.get_metadata(
            berg_dir=self.berg_dir,
            model_instance=self
        )
    
    def _validate_and_prepare_stimulus(
        self, 
        stimulus: Union[List[str], np.ndarray]
    ) -> pd.DataFrame:
        """
        Validate and prepare stimulus input for the model.
        
        Parameters
        ----------
        stimulus : list of str or np.ndarray
            List or array of natural language sentences.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with 'sentence' column and appropriate index.
        """
        # Convert to list if numpy array
        if isinstance(stimulus, np.ndarray):
            if stimulus.ndim != 1:
                raise StimulusError(
                    f"Stimulus array must be 1D, got shape {stimulus.shape}"
                )
            stimulus = stimulus.tolist()
        
        if not isinstance(stimulus, list):
            raise StimulusError(
                "Stimulus must be a list of strings or 1D numpy array"
            )
        
        if len(stimulus) == 0:
            raise StimulusError("Stimulus list cannot be empty")
        
        # Validate all elements are strings
        if not all(isinstance(s, str) for s in stimulus):
            raise StimulusError(
                "All stimulus elements must be strings"
            )
        
        # Check word count and warn if not 6 words (only once)
        if not self._warning_shown:
            non_six_word = [s for s in stimulus if len(s.split()) != 6]
            if non_six_word:
                print(
                    "WARNING: This model was trained on 6-word sentences. "
                    f"Found {len(non_six_word)} sentence(s) with different word counts. "
                    "Performance may vary for sentences of different lengths."
                )
                self._warning_shown = True
        
        # Create DataFrame with required format
        stimset = pd.DataFrame({"sentence": stimulus})
        
        # Add required index format: beta-neural-control-test.1, beta-neural-control-test.2, etc.
        stimset.index = [f"beta-neural-control-test.{i+1}" for i in range(len(stimset))]
        
        return stimset
    
    def generate_response(
        self,
        stimulus: Union[List[str], np.ndarray],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate in silico fMRI responses for text stimuli.
        
        Parameters
        ----------
        stimulus : list of str or np.ndarray
            List or array of natural language sentences to encode.
        show_progress : bool, default=True
            Whether to show progress (included for API consistency, not used by this model).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with shape (n_sentences, n_rois) containing predicted z-scored
            BOLD response magnitudes. Index contains sentence identifiers, columns
            contain ROI names. If ROI selection was specified, only selected ROIs
            are included as columns.
        """
        # Validate and prepare stimulus
        stimset = self._validate_and_prepare_stimulus(stimulus)
        
        # Encode with ANN
        self.ann_encoder.encode(
            stimset=stimset,
            cache_new_actv=False,
            case=None,
            stimsetid_suffix='',
            include_special_tokens=True,
            verbose=False
        )
        
        # Encode with Brain (mock encoding - no actual neural data needed)
        self.brain_encoder.encode(
            stimset=stimset,
            neural_data=None,
            specific_target=None
        )
        
        # Initialize mapping after encoders have stimset
        if self.mapping is None:
            metric = Metric(metric=self.METRIC)
            
            self.mapping = Mapping(
                ANNEncoder=self.ann_encoder,
                ann_layer=self.SOURCE_LAYER,
                BrainEncoder=self.brain_encoder,
                mapping_class=self.MAPPING_CLASS,
                metric=metric,
                Preprocessor=None,
                preprocess_X=False,
                preprocess_y=False,
            )
            
            # Load stored mapping weights with optional ROI selection
            weight_dir = os.path.join(
                self.berg_dir,
                'encoding_models',
                'modality-fmri',
                'train_dataset-tuckute_2024',
                'model-GPT2_XL',
                'encoding_models_weights'
            )
            self.mapping.load_full_mapping(
                WEIGHTDIR=weight_dir,
                mapping_result_identifier=self.WEIGHT_FILENAME,
                roi_selection=self.roi_list
            )
        
        # Generate predictions using pre-fitted mapping
        df_preds = self.mapping.predict_using_prefitted_mapping()
        
        # Return DataFrame with ROI columns
        return df_preds
    
    @classmethod
    def get_metadata(
        cls,
        berg_dir: Optional[str] = None,
        model_instance: Optional[BaseModelInterface] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.
        
        Parameters
        ----------
        berg_dir : str, optional
            Path to BERG directory containing metadata.
        model_instance : BaseModelInterface, optional
            If provided, extract parameters from this model instance.
        
        Returns
        -------
        dict
            Metadata dictionary containing fMRI and encoding model information.
        """
        # Extract parameters from model instance if provided
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir
        
        # Validate required parameters
        if berg_dir is None:
            raise InvalidParameterError(
                "Required parameter missing: berg_dir"
            )
        
        # Build metadata path
        metadata_path = os.path.join(
            berg_dir,
            'encoding_models',
            'modality-fmri',
            'train_dataset-tuckute_2024',
            'model-GPT2_XL',
            'metadata',
            'metadata.npy'
        )
        
        # Load metadata
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique string identifier.
        
        Returns
        -------
        str
            Model ID string.
        """
        return cls.MODEL_ID
    
    def cleanup(self) -> None:
        """
        Release memory and resources associated with the model.
        """
        if hasattr(self, 'ann_encoder') and self.ann_encoder is not None:
            self.ann_encoder = None
        
        if hasattr(self, 'brain_encoder') and self.brain_encoder is not None:
            self.brain_encoder = None
        
        if hasattr(self, 'mapping') and self.mapping is not None:
            self.mapping = None
        
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata = None
