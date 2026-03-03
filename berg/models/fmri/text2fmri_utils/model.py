import logging
from huggingface_hub import hf_hub_download
from x_transformers import Encoder

import torch
import torch.nn as nn
import os

from berg.models.fmri.text2fmri_utils.config import Text2fMRIConfig


class Text2fMRIModel(nn.Module):
    """
    Transformer-based encoding head that maps per-TR text features to fMRI brain activity.

    This model acts as the "mapping head" between the frozen LLM features and the brain space.
    It uses a bidirectional transformer encoder to mix information over time (TRs), allowing
    the predicted brain activity at time T to depend on context from neighboring timepoints.

    Attributes:
        config (Text2fMRIConfig): Configuration object.
        subj_emb (nn.Embedding): Learnable subject-specific embedding vectors.
        inp (nn.Linear): Projector from (LLM features + Subject Emb) to Transformer Dimension.
        encoder (x_transformers.Encoder): The main transformer backbone with RoPE.
        out (nn.Linear): Final projector to brain region space (e.g., 1000 ROIs).
        out_bias (nn.Embedding): Learnable subject-specific bias per output ROI.
    """

    def __init__(self, config: Text2fMRIConfig = Text2fMRIConfig(), device: str = "cpu", berg_dir: str = None):
        """
        Initialize the Text2fMRI model.

        Args:
            config (Text2fMRIConfig): Model hyperparameters.
            device (str): Device to initialize on ('cpu' or 'cuda').
            berg_dir (str, optional): Path to the BERG cache directory.
        """
        super().__init__()
        self.config = config
        self.berg_dir = berg_dir
        self.subj_emb = nn.Embedding(
            config.num_subjects, config.subject_embedding_dim)
        self.device = device

        # Input is [B, T, extractor_LLM_feature_size + sub_size] -> project to d_model
        self.inp = nn.Linear(config.extractor_LLM_feature_size +
                             config.subject_embedding_dim, config.transformer_dim)

        use_flash = (self.device == "cuda" and torch.cuda.is_available())
        # x-transformers Encoder:
        # - rotary_pos_emb=True enables RoPE (relative position in attention without
        #   overwriting your frozen feature vectors).
        # - layer_dropout adds stochastic depth regularization.
        self.encoder = Encoder(
            dim=config.transformer_dim,
            depth=config.num_transformer_layers,
            heads=config.num_transformer_heads,
            rotary_pos_emb=True,
            rotary_emb_dim=config.transformer_dim // config.num_transformer_heads,
            layer_dropout=0.1,  # may lower to 0.05 if your runs are short
            attn_dropout=0.1,
            ff_dropout=0.1,
            attn_dim_head=config.transformer_dim//config.num_transformer_heads,
            attn_flash=use_flash,
            use_scalenorm=True,
        )

        # Final projection to brain space (e.g., 1000 parcels).
        self.out = nn.Linear(config.transformer_dim,
                             config.num_rois, bias=False)

        # Subject-specific bias per output channel (per parcel)
        self.out_bias = nn.Embedding(config.num_subjects, config.num_rois)

        # Small, safe initialization: start biases at zero so early training
        # isn’t dominated by per-subject offsets.
        with torch.no_grad():
            nn.init.zeros_(self.out_bias.weight)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor, roi_indices=None):
        """
        Forward pass to predict brain activity from text features.

        Args:
            x (torch.Tensor): Input features of shape [B, T, extractor_LLM_feature_size].
            subject_ids (torch.Tensor): Subject indices of shape [B].
            roi_indices (numpy.ndarray, optional): Integer indices of ROIs to compute.
                If None, computes all ROIs. If provided, only computes selected ROIs
                for efficiency (mathematically equivalent to computing all and slicing).

        Returns:
            torch.Tensor: Predicted brain activity of shape [B, T, num_rois] or 
                [B, T, num_selected_rois] if roi_indices is provided.
        """
        # Minimal sanity checks that fail fast with a clear message
        assert x.dim(
        ) == 3, f"Expected x [1, T, extractor_LLM_feature_size]], got shape {tuple(x.shape)}"

        B, T, _ = x.shape

        # Subject conditioning: replicate subject embedding across time and concatenate
        s = self.subj_emb(subject_ids)          # [B, sub_size]
        s = s.unsqueeze(1).expand(B, T, -1)     # [B, T, sub_size]

        # [B, T, extractor_LLM_feature_size + sub_size]
        x = torch.cat([x, s], dim=-1)
        x = self.inp(x)                         # [B, T, d_model]

        # RoPE-enabled self-attention over time.
        # Note: this encoder is bidirectional (non-causal) by default.
        # That’s appropriate for offline encoding; if you ever need strictly
        # past-only context, you’d switch to a causal mask.
        h = self.encoder(x)          # [B, T, d_model]

        # Efficient ROI slicing: only compute selected ROIs if indices provided
        if roi_indices is not None:
            roi_indices_tensor = torch.as_tensor(roi_indices, device=self.device)
            # Slice output layer weights to only compute selected ROIs
            base = torch.nn.functional.linear(h, self.out.weight[roi_indices_tensor])  # [B, T, num_selected_rois]
            bias = self.out_bias.weight[subject_ids][:, roi_indices_tensor].unsqueeze(1)  # [B, 1, num_selected_rois]
        else:
            # Compute all ROIs
            base = self.out(h)                      # [B, T, num_rois]
            bias = self.out_bias(subject_ids).unsqueeze(1)  # [B, 1, num_rois]
        
        return base + bias

    def load_model_from_hub(self, repo_id: str):
        """
        Downloads and loads weights from the Hugging Face Hub if the config matches.

        Args:
            repo_id (str): HuggingFace repository ID to download weights from
                (e.g. "ShreyDixit/Text2fMRI-Qwen-2.5-0.5B").
        """
        # Construct cache directory path
        cache_dir = None
        if self.berg_dir is not None:
            cache_dir = os.path.join(
                self.berg_dir,
                "encoding_models",
                "modality-fmri",
                "train_dataset-cneuromod_algo2025",
                "model-text2fmri",
                "encoding_models_weights"
            )
        
        # Download the weights file (cached automatically)
        weights_path = hf_hub_download(
            repo_id=repo_id, 
            filename="model.pt",
            cache_dir=cache_dir
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)

    def load_model(self, PRETRAINED_CONFIGS):
        """
        Orchestrates model loading. Checks if current config exists in the registry.

        Args:
            pretrained_configs (Dict[Text2fMRIConfig, str]): Registry of valid models.
        """
        if self.config in PRETRAINED_CONFIGS:
            self.load_model_from_hub(PRETRAINED_CONFIGS[self.config])
            logging.info(
                f"Model loaded from {PRETRAINED_CONFIGS[self.config]}")

        else:
            logging.info(f"Model loaded with random weights")