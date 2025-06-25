#%%
from functools import partial
import logging
from einops import rearrange, repeat
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from berg.models.fmri.mem.config import AutoConfig

from berg.models.fmri.mem.backbone import (
    build_backbone,
    AdaLNLoRADiNOv2ViT,
)
from berg.models.fmri.mem.blocks import (
    build_conv_blocks,
    build_class_token_mlp,
    DictConvBlocks,
    ClassTokenMLPs,
)
from berg.models.fmri.mem.config_utils import load_from_yaml
from berg.models.fmri.mem.topyneck import (
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
        
    def forward(self, x):
        # x: [B, 3, 224, 224]   # image after normalization
        out1 = self.model_part1(x)
        out2 = self.model_part2(x)
        out = out2
        out[:, self.part1_voxel_indices] = out1
        return out
    



# %%
if __name__ == '__main__':
    # model_path = "/nfscc/alg23/xalex_distill2/high/t826c6_00016_DATASET.SUBJECT_LIST=subj01,LOSS.DARK.MAX_EPOCH=90,/soup.pth"
    subject = 'subj01'
    cfg_path = "/workspace/model_packed2/config.yaml"
    model_path1 = f"/workspace/model_packed2/ckpts/{subject}_part1.pth"
    model_path2 = f"/workspace/model_packed2/ckpts/{subject}_part2.pth"
    model1 = _load_one_model(model_path1, subject, cfg_path)
    model2 = _load_one_model(model_path2, subject, cfg_path)
    voxel_indices_path = "/workspace/model_packed2/ckpts/part1_voxel_indices.pt"
    voxel_indices = torch.load(voxel_indices_path)[subject]
    model = TowPartModel(model1, model2, voxel_indices)

    x = torch.randn(1, 3, 224, 224)
    x = x.cuda()
    model = model.cuda()
    out = model(x)
    print(out.shape)
