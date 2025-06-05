import sys
import os
import struct
import h5py
from scipy.stats import pearsonr
from tqdm import tqdm
import math

import numpy as np
import src_new.numpy_utility as pnu

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim
import copy


def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    _x.data.copy_(T.from_numpy(x))
    
    
def downsampling(fm_s, rf_s):
    if fm_s==rf_s:
        return 0, None
    elif fm_s<rf_s:
        # downsample rf, transpose to apply from righthand side
        return -1, pnu.create_downsampling_vector(old_dim=rf_s, new_dim=fm_s, preserve_norm=True).T
    else:
        # downsample fmap, transpose to apply from righthand side
        return 1,  pnu.create_downsampling_vector(old_dim=fm_s, new_dim=rf_s, preserve_norm=False).T        
        
def apply_downsampling(tensor4d, sampling_matrix_1d, start_dim=2):
    return T.tensordot(T.tensordot(tensor4d, sampling_matrix_1d, dims=[[start_dim],[0]]),
                       sampling_matrix_1d, dims=[[start_dim],[0]])

    
def iterate_block(length, blocksize):
    res = length%blocksize
    for r in np.arange(0, length - res, blocksize):
        yield np.arange(r,r+blocksize)
    if res>0: 
        yield np.arange(length-blocksize,length)

def iterate_voxel_batch(voxels, voxel_batch_size, voxel_axis=1):
    for s,v in voxels.items():
        for idx in iterate_block(v.shape[voxel_axis], voxel_batch_size):
            yield s, idx, np.take(v, indices=idx, axis=voxel_axis)



class Torch_FWRF(L.Module):
    def __init__(self, fmaps, rf_rez=1, nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_FWRF, self).__init__()
        self.fmaps_shapes = [list(f.size()) for f in fmaps]
        self.nf = np.sum([s[1] for s in self.fmaps_shapes])
        self.pre_nl  = pre_nl
        self.post_nl = post_nl
        self.nv = nv
        self.ns = np.square(rf_rez)

        self.rfs = [L.Parameter(T.tensor(np.ones(shape=(self.nv, rf_rez, rf_rez), dtype=dtype), requires_grad=True)),]
        self.register_parameter('rf0', self.rfs[0])
        self.sm = L.Softmax(dim=1)
        #self.m  = T.tensor(np.ones(shape=(self.nv, self.nf), dtype=dtype), requires_grad=False).to(device)
        self.w  = L.Parameter(T.tensor(np.random.normal(0, 0.001, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        self.b  = L.Parameter(T.tensor(np.full(fill_value=0.0, shape=(self.nv,), dtype=dtype), requires_grad=True))
        self.dl = []
        self.ul = []
        print (self.nf, rf_rez, rf_rez)
        for k,fm_rez in enumerate(self.fmaps_shapes):
            if fm_rez[2]>1:
                d, u = downsampling(fm_rez[2], rf_rez)
                self.dl += [d,]
                if u is not None:
#                    self.ul += [T.tensor(u.astype(dtype), requires_grad=False).to(device),]
                    self.ul += [T.tensor(u.astype(dtype), requires_grad=False),]
                    self.register_buffer('sc%d'%k, self.ul[-1], persistent=False)
                    if d<0: # downsample rf
                        print ('rescale from', [rf_rez, rf_rez], 'to', fm_rez[2:4])
                    elif d>0: # downsample fm
                        print ('rescale from', fm_rez[2:4], 'to', [rf_rez, rf_rez])
                else:
                    self.ul += [None,]
                    print ('native', fm_rez[2:4])
            else:
                self.dl += [None,]
                self.ul += [None,]
                print ('singleton fmaps')
              
    def forward(self, fmaps):
        phi = []
        for k,(fm,d,u) in enumerate(zip(fmaps, self.dl, self.ul)):
            if self.pre_nl is not None:
                f = self.pre_nl(fm)
            else:
                f = fm
            if d is not None:
                g = T.reshape(self.sm(T.flatten(self.rfs[0], start_dim=1)), self.rfs[0].size())
                if d<0: # downsample rf
                    g = apply_downsampling(g, getattr(self, 'sc%d'%k), start_dim=1)
                    # for some weird reason, torch doesn't transfer the buffer to gpu correectly when refered via the member variable, but correctly when refering to the attribute name. There is a new interface in torch v1.9
                    
                elif d>0: # downsample fm               
                    f = apply_downsampling(f, getattr(self, 'sc%d'%k), start_dim=2)
                # fmaps : [batch, features, space]
                # v     : [nv, space]            
                phi += [T.tensordot(g,f, dims=[[1,2], [2,3]]),] # apply pooling field and add to list.
            else:
                phi += [f[None,:,:,0,0].repeat(self.nv, 1, 1)]
            # phi : [nv, batch, features] 
        Phi = T.cat(phi, dim=2)
        if self.post_nl is not None:
            Phi = self.post_nl(Phi)
        vr = T.squeeze(T.bmm(Phi, T.unsqueeze(self.w,2))).t() + T.unsqueeze(self.b,0)
        # vr : [batch, nv]
        return vr
    
class Torch_LayerwiseFWRF(L.Module):
    def __init__(self, fmaps, nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_LayerwiseFWRF, self).__init__()
        self.fmaps_shapes = [list(f.size()) for f in fmaps]
        self.nf = np.sum([s[1] for s in self.fmaps_shapes])
        self.pre_nl  = pre_nl
        self.post_nl = post_nl
        self.nv = nv
        ##
        self.rfs = []
        self.sm = L.Softmax(dim=1)
        for k,fm_rez in enumerate(self.fmaps_shapes):
            rf = L.Parameter(T.tensor(np.ones(shape=(self.nv, fm_rez[2], fm_rez[2]), dtype=dtype), requires_grad=True))
            self.register_parameter('rf%d'%k, rf)
            self.rfs += [rf,]
        #self.w  = L.Parameter(T.tensor(np.random.normal(0, 0.001, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        #self.b  = L.Parameter(T.tensor(np.full(fill_value=0.0, shape=(self.nv,), dtype=dtype), requires_grad=True))
        self.w  = L.Parameter(T.tensor(np.random.normal(0, 0.01, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        self.b  = L.Parameter(T.tensor(np.random.normal(0, 0.01, size=(self.nv,)).astype(dtype=dtype), requires_grad=True))
        
    def forward(self, fmaps):
        phi = []
        for fm,rf in zip(fmaps, self.rfs): #, self.scales):
            g = self.sm(T.flatten(rf, start_dim=1))
            f = T.flatten(fm, start_dim=2)  # *s
            if self.pre_nl is not None:          
                f = self.pre_nl(f)
            # fmaps : [batch, features, space]
            # v     : [nv, space]
            phi += [T.tensordot(g, f, dims=[[1],[2]]),] # apply pooling field and add to list.
            # phi : [nv, batch, features] 
        Phi = T.cat(phi, dim=2)
        if self.post_nl is not None:
            Phi = self.post_nl(Phi)
        vr = T.squeeze(T.bmm(Phi, T.unsqueeze(self.w,2))).t() + T.unsqueeze(self.b,0)
        return vr
    
    
    
    

            
class Torch_FWRF_Block(L.Module):
    def __init__(self, fmaps, rf_rez=1, block_nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_FWRF_Block, self).__init__()
        #self.bns = nn.ModuleList([torch.nn.BatchNorm2d(fm.size()[1], eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)
        #                         for fm in fmaps])
        self.fwrf = Torch_FWRF(fmaps, rf_rez=rf_rez, nv=block_nv, pre_nl=pre_nl, post_nl=post_nl, dtype=dtype)
        # save initial param values
        self.weight_init_value = copy.deepcopy(self.fwrf.state_dict())
        
    def forward(self, fmaps):
        #bn_fmaps = [bn(fm) for bn,fm in zip(self.bns, fmaps)]
        return self.fwrf(fmaps)
        
    def get_optimizer(self, lr):   
        # reload initial param values
        self.fwrf.load_state_dict(self.weight_init_value)
        return optim.Adam([{'params': self.parameters()}], lr=lr, betas=(0.9, 0.999), eps=1e-08)
    
class Torch_LayerwiseFWRF_Block(L.Module):
    def __init__(self, fmaps, block_nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_LayerwiseFWRF_Block, self).__init__()
        #self.bns = nn.ModuleList([torch.nn.BatchNorm2d(fm.size()[1], eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)
        #                         for fm in fmaps])
        self.fwrf = Torch_LayerwiseFWRF(fmaps, nv=block_nv, pre_nl=pre_nl, post_nl=post_nl, dtype=dtype)
        # save initial param values
        self.weight_init_value = copy.deepcopy(self.fwrf.state_dict())
        
    def forward(self, fmaps):
        #bn_fmaps = [bn(fm) for bn,fm in zip(self.bns, fmaps)]
        return self.fwrf(fmaps)
        
    def get_optimizer(self, lr):   
        # reload initial param values
        self.fwrf.load_state_dict(self.weight_init_value)
        return optim.Adam([{'params': self.parameters()}], lr=lr, betas=(0.9, 0.999), eps=1e-08)
