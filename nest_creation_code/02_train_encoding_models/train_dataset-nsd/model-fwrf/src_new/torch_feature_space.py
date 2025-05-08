import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math

import torch
import torch.nn as nn

def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        

class Torch_filter_fmaps(nn.Module):
    def __init__(self, _fmaps, lmask, fmask):
        super(Torch_filter_fmaps, self).__init__()
        device = next(_fmaps.parameters()).device
        self.fmaps = _fmaps
        self.lmask = lmask
        self.fmask = [nn.Parameter(torch.from_numpy(fm).to(device), requires_grad=False) for fm in fmask]
        for k,fm in enumerate(self.fmask):
             self.register_parameter('fm%d'%k, fm)

    def forward(self, _x):
        _fmaps = self.fmaps(_x)
        return [torch.index_select(torch.cat([_fmaps[l] for l in lm], axis=1), dim=1, index=fm) for lm,fm in zip(self.lmask, self.fmask)]



def get_tuning_masks(layer_rlist, fmaps_count):
    tuning_masks = []
    for rl in layer_rlist:
        tm = np.zeros(shape=(fmaps_count,), dtype=bool)
        tm[rl] = True
        tuning_masks += [tm,]
    return tuning_masks
    
    
def filter_dnn_feature_maps(data, _fmaps_fn, batch_size, fmap_max=1024, concatenate=True, trn_size=None):
    '''Runs over the image set and keep the fmap_max features with the most variance withing each layer of the network.
    Return an updated torch function and a list of binary mask that match the new feature space to identify the layer provenance of the feature'''
    device = next(_fmaps_fn.parameters()).device
    size = trn_size if trn_size is not None else len(data)
#    _fmaps = _fmaps_fn(_to_torch(data[:batch_size], device=device))
    fmaps_fn = lambda x: [np.copy(_fm.data.cpu().numpy()) for _fm in _fmaps_fn(_to_torch(x, device=device))]
    fmaps = fmaps_fn(data[:batch_size])
    run_avg = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps]
    run_sqr = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps] 
    for rr,rl in tqdm(iterate_range(0, size, batch_size)):
        fb = fmaps_fn(data[rr])
        for k,f in enumerate(fb):
            if f.shape[1]>fmap_max: # only need the average if we're going to use them to reduce the number of feature maps
                run_avg[k] += np.sum(np.mean(f.astype(np.float64), axis=(2,3)), axis=0)
                run_sqr[k] += np.sum(np.mean(np.square(f.astype(np.float64)), axis=(2,3)), axis=0)
    for k in range(len(fb)):
        run_avg[k] /= size
        run_sqr[k] /= size
    ###
    fmask = [np.zeros(shape=(fm.shape[1]), dtype=bool) for fm in fmaps]
    fmap_var = [np.zeros(shape=(fm.shape[1]), dtype=np.float32) for fm in fmaps]
    for k,fm in enumerate(fmaps):  
        if fm.shape[1]>fmap_max:
            #select the feature map with the most variance to the dataset
            fmap_var[k] = (run_sqr[k] - np.square(run_avg[k])).astype(np.float32)
            most_var = fmap_var[k].argsort()[-fmap_max:] #the feature indices with the top-fmap_max variance
            fmaps[k] = fm[:,np.sort(most_var),:,:]
            fmask[k][most_var] = True
        else:
            fmask[k][:] = True
        print ("layer: %s, shape=%s" % (k, (fmaps[k].shape)))    
        sys.stdout.flush()

    # ORIGINAL PARTITIONING OF LAYERS
    fmaps_sizes = [fm.shape for fm in fmaps]
    fmaps_count = sum([fm[1] for fm in fmaps_sizes])   
    partitions = [0,]
    for r in fmaps_sizes:
        partitions += [partitions[-1]+r[1],]
    layer_rlist = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
    # concatenate fmaps of identical dimension to speed up rf application
    clmask, cfmask, cfmaps = [],[],[]
    print ("")
    sys.stdout.flush()
    # I would need to make sure about the order and contiguousness of the fmaps to preserve the inital order.
    # It isn't done right now but since the original feature maps are monotonically decreasing in resultion in
    # the examples I treated, the previous issue doesn't arise.
    if concatenate:
        for k,us in enumerate(np.unique([np.prod(fs[2:4]) for fs in fmaps_sizes])[::-1]): ## they appear sorted from small to large, so I reversed the order
            mask = np.array([np.prod(fs[2:4])==us for fs in fmaps_sizes]) # mask over layers that have that spatial size
            lmask = np.arange(len(fmaps_sizes))[mask] # list of index for layers that have that size
            bfmask = np.concatenate([fmask[l] for l in lmask], axis=0)
            clmask += [lmask,]
            cfmask += [np.arange(len(bfmask))[bfmask],]
            cfmaps += [np.concatenate([fmaps[l] for l in lmask], axis=1),]
            print ("fmaps: %s, shape=%s" % (k, (cfmaps[-1].shape)))
            sys.stdout.flush()
        fmaps_sizes = [fm.shape for fm in cfmaps]
    else:
        for k,fm in enumerate(fmask):
            clmask += [np.array([k]),]
            cfmask += [np.arange(len(fm))[fm],]
    ###
    tuning_masks = get_tuning_masks(layer_rlist, fmaps_count)
    assert np.sum(sum(tuning_masks))==fmaps_count, "%d != %d" % (np.sum(sum(tuning_masks)), fmaps_count)    
    #layer_rlist, fmaps_sizes, fmaps_count, clmask, cfmask #, scaling
    return Torch_filter_fmaps(_fmaps_fn, clmask, cfmask), clmask, cfmask, tuning_masks


