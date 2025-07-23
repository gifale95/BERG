#%%
from functools import partial
import logging
from einops import rearrange, repeat
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from berg.models.fmri.huze.config import AutoConfig

from berg.models.fmri.huze.backbone import (
    build_backbone,
    AdaLNLoRADiNOv2ViT,
)
from berg.models.fmri.huze.blocks import (
    build_conv_blocks,
    build_class_token_mlp,
    DictConvBlocks,
    ClassTokenMLPs,
)
from berg.models.fmri.huze.config_utils import load_from_yaml
from berg.models.fmri.huze.topyneck import (
    build_coords_mlp,
    CachedCoordsMLP,
    build_voxelouts_weight,
    CoordsMLPLinearWeight,
    VoxelNonShareLinearWeight,
)

import numpy as np

class BrainEncodingModel(nn.Module):
    def __init__(
        self,
        cfg: AutoConfig,
        n_voxel_dict = {'subj01': 327684},
    ):
        
        super().__init__()
        self.subject_list = list(n_voxel_dict.keys())
        assert len(self.subject_list) == 1, "Only one subject is supported"

        self.layers = cfg.MODEL.BACKBONE.LAYERS
        self.layers_small = cfg.MODEL.BACKBONE_SMALL.LAYERS
        self.n_layers = len(self.layers)
        r = cfg.MODEL.WIDTH_RATIO
        cfg.MODEL.CONV_HEAD.WIDTH = int(cfg.MODEL.CONV_HEAD.WIDTH * r)
        self.cfg = cfg

        self.backbone: AdaLNLoRADiNOv2ViT = build_backbone(cfg)
        self.conv_blocks: DictConvBlocks = build_conv_blocks(cfg)
        self.cls_blocks: ClassTokenMLPs = build_class_token_mlp(cfg)
        
        def build_each_subject(fn, subject_list):
            return nn.ModuleDict({subject: fn() for subject in subject_list})
                
        self.layer_selector: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=self.n_layers,
                act_fn=partial(nn.Softmax, dim=-1),
            ),
            self.subject_list,
        )
        self.retina_mapper: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=2,
                act_fn=nn.Tanh,
            ),
            self.subject_list,
        )
        self.mu_sigma = cfg.MODEL.RETINA_MAPPER.CONSTANT_SIGMA


        # voxel-wise output
        d_model = self.cfg.MODEL.CONV_HEAD.WIDTH
            
        self.n_voxel_dict = n_voxel_dict
        self.d_model = d_model
        self.voxel_outs_weight: Dict[
            str, Union[VoxelNonShareLinearWeight, CoordsMLPLinearWeight]
        ] = nn.ModuleDict(
            {
                subject: build_voxelouts_weight(cfg, self.n_voxel_dict[subject], self.d_model)
                for subject in self.subject_list
            }
        )
        
        self.coords : nn.Parameter = None

        
    def forward(
        self,
        x: Tensor,  # [B, C, H, W]
        voxel_indices: Optional[Tensor] = None,
        chunk_size=4096,
        **kwargs,
    ):
        coords = self.coords
        subject = self.subject_list[0]
        
        bsz = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        x_retina_grid, x_cls_dict = self.backbone.get_intermediate_layers(
            x, n=self.layers, c=None
        )
        x_retina_grid = self.conv_blocks(x_retina_grid)
        x_cls_dict = self.cls_blocks(x_cls_dict)
        x_cls = torch.stack(list(x_cls_dict.values()), dim=-1)  # [B, D, 4]


        #############################
        ### voxel-wise prediction ###
        #############################

        # divide voxels into chunks to avoid OOM
        n_voxels = coords.shape[0]
        if voxel_indices is None or voxel_indices == ...:
            voxel_indices = torch.arange(n_voxels, device=coords.device)
        voxel_indices_chunks = torch.split(voxel_indices, chunk_size)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in voxel_indices_chunks:
            out_y, reg_layer = self._forward_voxels(
                x_retina_grid,
                x_cls,
                subject,
                coords,
                voxel_indices_chunk,
                bsz,
                device,
                dtype
            )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)  # [B, N]
        reg_layer = torch.cat(reg_layers, dim=0).mean()  # [1]

        # if self.training:
            # return out_y, reg_layer
        # else:
        return out_y

    def _forward_voxels(
        self,
        x_retina_grid: Dict[str, Tensor],  # {layer: [B, D, H/k, W/k], ...}
        x_cls: Tensor,  # [B, D, 4]
        subject: str,
        coords: Tensor,
        voxel_indices: Tensor,
        bsz,
        device,
        dtype,
    ):
        N = len(voxel_indices)
        
        ## Layer Selector
        w_layer = self.layer_selector[subject](coords, voxel_indices)  # [N, 4]

        # regularization
        def entropy(x):
            return (x * x.log()).sum(dim=1)

        if self.training and next(self.layer_selector.parameters()).requires_grad:
            reg_layer = entropy(w_layer)  # [N]
        else:
            reg_layer = torch.zeros_like(w_layer[:, 0])  # [N]

        x_cls = repeat(x_cls, "b d l -> b n d l", n=1)
        _w_layer = repeat(w_layer, "n l -> b n d l", b=1, d=1)

        x_cls = (x_cls * _w_layer).sum(dim=-1)  # [B, N, D]


        ## Retina Mapper
        mu = self.retina_mapper[subject](coords, voxel_indices)  # [N, 2]
        mu = mu * (1 - self.mu_sigma)
        if self.training:
            norm = torch.normal(0, torch.ones_like(mu) * self.mu_sigma)
            mu = mu + norm
        bsz = x_cls.shape[0]
        mu = repeat(mu, "n d -> b n d", b=bsz)
        mu = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

        if self.cfg.EXPERIMENTAL.USE_LAYER_SELECTOR:
            _w_layer = repeat(w_layer, "n l -> b n l", b=1)
        x_retina = None  # [B, N, D]
        for i, layer in zip(range(self.n_layers), self.layers):
            x = x_retina_grid[str(layer)]
            _x_retina = F.grid_sample(
                x,
                mu,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B, C, N, D] (C=D_model, D=1, N=N_voxels)
            _x_retina = rearrange(_x_retina, "b c n d -> b n (c d)")
            if self.cfg.EXPERIMENTAL.USE_LAYER_SELECTOR:
                _x_retina = _x_retina * _w_layer[:, :, i : i + 1]
            if x_retina is None:
                x_retina = _x_retina
            else:
                x_retina += _x_retina
        # x_retina: [B, N, D]
        
        
        x_y = x_retina + x_cls  # [B, N, D]  # T=0
        w, b = self.voxel_outs_weight[subject](coords, voxel_indices)  # [N, DDD], [N]
         
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

        return out_y, reg_layer  # [B, N], [N]


        
def _load_one_model(model_path: str, subject: str='subj01', cfg_path: str=None):
    cfg = load_from_yaml(cfg_path)
    
    # load model weights
    sd = torch.load(model_path, map_location='cpu')
    n_voxels = sd[f'model.voxel_outs_weight.{subject}.weight'].shape[0]
    # create model
    model = BrainEncodingModel(cfg, {subject: n_voxels})
    
    # save voxel's coordinates to model
    coords = sd[f'coord_dict.{subject}']
    model.coords = nn.Parameter(coords)
    
    # load weights
    filtered_sd = {k: v for k, v in sd.items() if k.startswith('model')}
    filtered_sd = {k[6:]: v for k, v in filtered_sd.items() if k.startswith('model')}
    filtered_sd['coords'] = model.coords  # add coordinates of voxels
    model.load_state_dict(filtered_sd)
    
    model = model.eval()
    return model


class TowPartModel(nn.Module):
    def __init__(self, model_part1, model_part2, part1_voxel_indices):
        super().__init__()
        self.model_part1 = model_part1
        self.model_part2 = model_part2
        self.part1_voxel_indices = part1_voxel_indices
            
            
    def forward(self, x, voxel_indices=None):
        # x: [B, 3, 224, 224] # image after normalization
        if voxel_indices is None:
            out1 = self.model_part1(x)
            out2 = self.model_part2(x)
            out = out2
            out[:, self.part1_voxel_indices] = out1
            return out
        else:
            # Compute only the selected voxels
            # Convert numpy to torch
            target_device = voxel_indices.device if voxel_indices is not None else x.device

            if isinstance(self.part1_voxel_indices, np.ndarray):
                part1_voxel_indices_tensor = torch.from_numpy(self.part1_voxel_indices).to(target_device)
            else:
                part1_voxel_indices_tensor = self.part1_voxel_indices.to(target_device)
            
            # Sort part1_voxel_indices for searchsorted to work correctly
            part1_sorted, sort_permutation = torch.sort(part1_voxel_indices_tensor)
            
            # Find which of the requested voxel_indices are in part1
            part1_mask = torch.isin(voxel_indices, part1_voxel_indices_tensor)
            part1_global_indices = voxel_indices[part1_mask]
            part2_global_indices = voxel_indices[~part1_mask]
            
            # Initialize output tensor
            out = torch.zeros(x.shape[0], len(voxel_indices), device=x.device)
            
            # Process part1 voxels
            if len(part1_global_indices) > 0:
                # Use sorted indices for searchsorted, then map back to original order
                part1_local_indices_sorted = torch.searchsorted(part1_sorted, part1_global_indices)
                part1_local_indices = sort_permutation[part1_local_indices_sorted]
                
                out1 = self.model_part1(x, voxel_indices=part1_local_indices)
                out[:, part1_mask] = out1
            
            # Process part2 voxels
            if len(part2_global_indices) > 0:
                # Create part2's global indices (all indices not in part1)
                total_voxels = self.model_part1.coords.shape[0] + self.model_part2.coords.shape[0]
                all_indices = torch.arange(total_voxels, device=part1_voxel_indices_tensor.device)
                part2_global_map = all_indices[~torch.isin(all_indices, part1_voxel_indices_tensor)]
                part2_global_map_sorted, part2_sort_perm = torch.sort(part2_global_map)
                
                # Map part2's global indices to local indices
                part2_local_indices_sorted = torch.searchsorted(part2_global_map_sorted, part2_global_indices)
                part2_local_indices = part2_sort_perm[part2_local_indices_sorted]
                
                out2 = self.model_part2(x, voxel_indices=part2_local_indices)
                out[:, ~part1_mask] = out2
            
            return out
        
