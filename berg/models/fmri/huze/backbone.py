# %%
import copy
from functools import partial
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange
from filelock import FileLock

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from berg.models.fmri.huze.config import AutoConfig
from berg.models.fmri.huze.registry import Registry

import math

import torch.nn.functional as F


BACKBONES = Registry()


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0


class MonkeyLoRALinear(nn.Module):
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + self.lora_scale * self.fc_lora(
            hidden_states
        )
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias


class AdaLNZeroPatch(nn.Module):
    def __init__(self, embed_dim, d_c=64, adaln_scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_c = d_c
        self.adaln_scale = adaln_scale

        # for condition (behavior data)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.d_c, 6 * self.embed_dim, bias=False),
            nn.Tanh(),
        )

        nn.init.zeros_(self.adaLN_modulation[0].weight)

    def forward(self, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (
            self.adaLN_modulation(c) * self.adaln_scale
        ).chunk(6, dim=1)

        scale_msa = scale_msa + 1
        gate_msa = gate_msa + 1
        scale_mlp = scale_mlp + 1
        gate_mlp = gate_mlp + 1

        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

def maxavg_globalpool2d(x):
    out = torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], dim=1)
    out = out.squeeze(-1).squeeze(-1)
    return out



# from dinov2.models.vision_transformer import DinoVisionTransformer

# from dinov2.layers.attention import MemEffAttention, Attention
# from dinov2.layers.block import NestedTensorBlock, Block
# from dinov2.layers.block import drop_add_residual_stochastic_depth


class AdaLNDiNOBlock(nn.Module):
    def __init__(self, block, d_c=64, adaln_scale=1.0):
        super().__init__()
        self.block = block
        self.embed_dim = block.norm1.weight.shape[0]
        self.d_c = d_c

        self.adaLN = AdaLNZeroPatch(self.embed_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(self, x, c: Optional[torch.Tensor] = None):
        # conditioning can be None
        bsz = x.shape[0]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=x.device, dtype=x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.block.ls1(
                self.block.attn(
                    self.modulate(self.block.norm1(x), shift_msa, scale_msa)
                )
            ) * gate_msa.unsqueeze(1)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.block.ls2(
                self.block.mlp(self.modulate(self.block.norm2(x), shift_mlp, scale_mlp))
            ) * gate_mlp.unsqueeze(1)

        # if self.block.training and self.block.sample_drop_ratio > 0.1:
        #     # the overhead is compensated only for a drop path rate larger than 0.1
        #     x = drop_add_residual_stochastic_depth(
        #         x,
        #         residual_func=attn_residual_func,
        #         sample_drop_ratio=self.block.sample_drop_ratio,
        #     )
        #     x = drop_add_residual_stochastic_depth(
        #         x,
        #         residual_func=ffn_residual_func,
        #         sample_drop_ratio=self.block.sample_drop_ratio,
        #     )
        # elif self.block.training and self.block.sample_drop_ratio > 0.0:
        #     x = x + self.block.drop_path1(attn_residual_func(x))
        #     x = x + self.block.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        # else:
        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


@BACKBONES.register("adaln_lora_dinov2_vit")
class AdaLNLoRADiNOv2ViT(nn.Module):
    def __init__(
        self, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0, ver='dinov2_vitl14', **kwargs
    ) -> None:
        super().__init__()

        vision_model = torch.hub.load("facebookresearch/dinov2", ver)
        self.vision_model = vision_model
        self.vision_model.requires_grad_(False)
        
        self.lora_scale = lora_scale
        self.rank = rank
        self.d_c = d_c
        self.adaln_scale = adaln_scale
        
        self.init_lora()
        
    def init_lora(self):
        self.vision_model = self.inject_lora_and_adaln_dinov2(
            self.vision_model,
            lora_scale=self.lora_scale,
            rank=self.rank,
            d_c=self.d_c,
            adaln_scale=self.adaln_scale,
        )

    @staticmethod
    def inject_lora_and_adaln_dinov2(
        model, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0
    ):
        for _i in range(len(model.blocks)):
            block = model.blocks[_i]
            attn = block.attn
            block.attn.qkv = MonkeyLoRALinear(
                attn.qkv, rank=rank, lora_scale=lora_scale
            )
            block.attn.proj = MonkeyLoRALinear(
                attn.proj, rank=rank, lora_scale=lora_scale
            )
            block.mlp.fc1 = MonkeyLoRALinear(
                block.mlp.fc1, rank=rank, lora_scale=lora_scale
            )
            block.mlp.fc2 = MonkeyLoRALinear(
                block.mlp.fc2, rank=rank, lora_scale=lora_scale
            )
            model.blocks[_i] = AdaLNDiNOBlock(block, d_c=d_c, adaln_scale=adaln_scale)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_model(x)

    def get_intermediate_layers(
        self,
        x,
        n: List[str] = [0, 1, 2, 3],
        c: Optional[torch.Tensor] = None,
        reshape=True,
        masks=None,
    ):
        x = self.vision_model.prepare_tokens_with_masks(x, masks)

        output_dict = {}
        cls_dict = {}
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x, c=c)
            if i not in n:
                continue
            saved_x = x.clone()
            if reshape:
                saved_x = saved_x[:, 1:, :]  # remove cls token, [B, N, C]
                p = int(np.sqrt(saved_x.shape[1]))
                saved_x = rearrange(saved_x, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
            output_dict[str(i)] = saved_x
            if i == len(self.vision_model.blocks) - 1:
                cls_dict[str(i)] = x[:, 0, :]  # [B, C]
            else:
                cls_dict[str(i)] = maxavg_globalpool2d(saved_x)
        return output_dict, cls_dict

@BACKBONES.register("dinov2_vit_l")
def dinov2_vit_l(**kwargs):
    ver='dinov2_vitl14'
    return AdaLNLoRADiNOv2ViT(ver=ver, **kwargs)

@BACKBONES.register("dinov2_vit_b")
def dinov2_vit_b(**kwargs):
    ver='dinov2_vitb14'
    return AdaLNLoRADiNOv2ViT(ver=ver, **kwargs)

@BACKBONES.register("dinov2_vit_s")
def dinov2_vit_s(**kwargs):
    ver='dinov2_vits14'
    return AdaLNLoRADiNOv2ViT(ver=ver, **kwargs)

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if ".module." in k:
            k = k.replace(".module.", ".")
        new_state_dict[k] = v
    return new_state_dict


def build_backbone(cfg: AutoConfig):
    # home = os.path.expanduser("~")
    # lock_path = os.path.join(home, ".cache", "download.lock")
    # with FileLock(lock_path):
    return BACKBONES[cfg.MODEL.BACKBONE.NAME](
        lora_scale=cfg.MODEL.BACKBONE.LORA.SCALE,
        rank=cfg.MODEL.BACKBONE.LORA.RANK,
        d_c=cfg.MODEL.COND.DIM,
        adaln_scale=cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE,
    )
    
def build_backbone_prev(cfg: AutoConfig):
    return BACKBONES[cfg.MODEL.BACKBONE_SMALL.NAME](
        lora_scale=cfg.MODEL.BACKBONE_SMALL.LORA.SCALE,
        rank=cfg.MODEL.BACKBONE_SMALL.LORA.RANK,
        d_c=cfg.MODEL.COND.DIM,
        adaln_scale=cfg.MODEL.BACKBONE_SMALL.ADAPTIVE_LN.SCALE,
    )
    
class SubjectTimeEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Each subject is running at a different clock speed, so we need to a subject-layer
    """
    def __init__(self, hidden_size, subject_list, frequency_embedding_size=256):
        super().__init__()
        self.subject_list = subject_list
        self.subject_layers = nn.ModuleDict()
        self.frequency_embedding_size = frequency_embedding_size

        for subject in subject_list:
            self.subject_layers[subject] = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, subject):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.subject_layers[subject](t_freq)
        t_emb = self.mlp(t_emb)
        return t_emb

def build_time_emd(cfg: AutoConfig):
    return SubjectTimeEmbed(
        hidden_size=cfg.MODEL.BACKBONE_SMALL.T_DIM,
        subject_list=cfg.DATASET.SUBJECT_LIST,
    )
    

def get_shape(model, input_size, n=[5, 11]):
    model = BACKBONES[model]()
    model.eval()
    model = model.cuda()
    input = torch.randn(1, 3, input_size, input_size).cuda()
    out_dict, cls_dict = model.get_intermediate_layers(input, n)
    for k, v in out_dict.items():
        print(k, v.shape, cls_dict[k].shape)
    
    return model

BACKBONEC = {
    'clip_vit_l': (224, [5, 11, 17, 23], [1024, 1024, 1024, 1024], [2048, 2048, 2048, 1024]),
    'clip_vit_b': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
    'clip_vit_s': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
    'dinov2_vit_l': (224, [5, 11, 17, 23], [1024, 1024, 1024, 1024], [2048, 2048, 2048, 1024]),
    'dinov2_vit_b': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
    'dinov2_vit_s': (224, [2, 5, 8, 11], [384, 384, 384, 384], [768, 768, 768, 384]),
}