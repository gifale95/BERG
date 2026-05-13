import os
import logging
from huggingface_hub import hf_hub_download
import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn as nn
from x_transformers import Encoder

from berg.models.fmri.vibe_utils.config import VIBEConfig


class ModalityFusionTransformer(nn.Module):
    """Project and fuse modality features together with subject conditioning."""

    def __init__(
        self,
        config: VIBEConfig
    ):
        super().__init__()

        self.config = config

        self._build_input_dims()

        self.projections = nn.ModuleDict({
            modality: self.build_projection(dim,
                                            config.modality_fusion_transformer_dim,
                                            config.modality_fusion_transformer_num_projection_layers)
            for modality, dim in self.input_dims.items()
        })

        self.subject_embeddings = nn.Embedding(self.config.num_subjects + 1,
                                               self.config.modality_fusion_transformer_dim)
        self.null_subject_index = self.config.num_subjects

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.modality_fusion_transformer_dim,
            nhead=self.config.modality_fusion_transformer_num_heads,
            dim_feedforward=self.config.modality_fusion_transformer_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=self.config.modality_fusion_transformer_num_layers)

    def build_projection(self, input_dim, output_dim, num_layers):
        """Build a simple MLP projection block for one modality stream."""
        layers = []
        dims = np.linspace(input_dim, output_dim, num_layers + 1, dtype=int)

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU())

        return nn.Sequential(*layers)

    def forward(self, inputs: dict, subject_ids):
        """Fuse modality sequences and append subject embeddings before encoding."""
        B, T, _ = next(iter(inputs.values())).shape

        projected_dict = {
            name: self.projections[name](inputs[name])
            for name in self.projections
        }

        projected = [projected_dict[name] for name in self.projections]
        x = torch.stack(projected, dim=2)

        subject_ids = torch.tensor(subject_ids, device=x.device, dtype=torch.long)

        subj_emb = self.subject_embeddings(subject_ids).unsqueeze(1).unsqueeze(2)
        subj_emb = subj_emb.expand(-1, T, 1, -1)

        x = torch.cat([x, subj_emb], dim=2)
        x = x.view(B * T, x.shape[2], -1)

        fused = self.transformer(x)

        if self.config.modality_fusion_transformer_fuse_mode == "concat":
            fused = fused.view(B * T, -1)
        elif self.config.modality_fusion_transformer_fuse_mode == "mean":
            fused = fused.mean(dim=1)

        fused = fused.view(B, T, -1)
        return fused

    def _build_input_dims(self):
        input_dims = {}
        input_dims["audio"] = self.config.audio_extractor_feature_size
        input_dims["text"] = self.config.text_extractor_feature_size
        input_dims["video"] = self.config.video_extractor_feature_size * \
            (self.config.video_extractor_pool_size**2)
        self.input_dims = input_dims


class VIBEModel(nn.Module):
    """End-to-end VIBE prediction model (fusion encoder + temporal predictor + ROI head)."""

    def __init__(
        self,
        config: VIBEConfig,
        device: str = "cpu",
        berg_dir: str = None,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.berg_dir = berg_dir
        self.encoder = ModalityFusionTransformer(config)

        fused_dim = (
            self.config.modality_fusion_transformer_dim * 4
            if self.config.modality_fusion_transformer_fuse_mode == "concat"
            else self.config.modality_fusion_transformer_dim
        )

        head_dim = fused_dim // self.config.predictor_transformer_num_heads

        use_flash = False  # self.device == "cuda" and torch.cuda.is_available())
        self.predictor = Encoder(
            dim=fused_dim,
            depth=self.config.predictor_transformer_num_layers,
            heads=self.config.predictor_transformer_num_heads,
            rotary_pos_emb=True,
            rotary_emb_dim=head_dim,
            attn_dim_head=head_dim,
            use_scalenorm=True,
            attn_flash=use_flash,
        )

        self.output_head = nn.Linear(fused_dim, self.config.num_rois, bias=False)
        self.register_buffer("pre_tokens", torch.empty(0, fused_dim))

    def forward(self, features, subject_ids):
        """Predict `[B, T, num_rois]` responses from multimodal feature sequences."""

        fused = self.encoder(features, subject_ids)

        preds = self.predictor(fused)
        preds = self.output_head(preds)

        return preds
    
    def load_model_from_hub(self, repo_id: str):
        """
        Downloads and loads weights from the Hugging Face Hub.

        Args:
            repo_id (str): HuggingFace repository ID to download weights from.
        """
        cache_dir = None
        if self.berg_dir is not None:
            cache_dir = os.path.join(
                self.berg_dir,
                "encoding_models",
                "modality-fmri",
                "train_dataset-cneuromod_algo2025",
                "model-vibe",
                "encoding_models_weights"
            )

        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=cache_dir
        )
        state_dict = load_file(weights_path, device="cpu")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict.pop("n_averaged", None)
        self.load_state_dict(state_dict, strict=True)
        self.to(self.device)

    def load_model(self, PRETRAINED_CONFIGS):
        """
        Load pretrained weights if current config is present in the registry.

        Args:
            pretrained_configs (Dict[VIBEConfig, str]): Registry of valid models.
        """
        if self.config in PRETRAINED_CONFIGS:
            self.load_model_from_hub(PRETRAINED_CONFIGS[self.config])
            logging.info(
                f"Model loaded from {PRETRAINED_CONFIGS[self.config]}")

        else:
            logging.info(f"Model loaded with random weights")
