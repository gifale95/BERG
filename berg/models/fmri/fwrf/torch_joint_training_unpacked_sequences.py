import sys
import os
import struct
import time
import h5py
from scipy.stats import pearsonr
from tqdm import tqdm
import math

import numpy as np
import berg.models.fmri.fwrf.numpy_utility as pnu

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim


def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    _x.data.copy_(T.from_numpy(x))
    
    
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual
        
def iterate_minibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    seq = ['|','/','--','\\']
    assert len(inputs) == len(targets)
    n = np.ceil(float(len(inputs)) / batchsize)
    for i,start_idx in enumerate(range(0, len(inputs), batchsize)):
        sys.stdout.write('\r%-2s: %.1f %%'%(seq[i%4], float(i+1)*100/n))
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]  
    

################################################################  
def iterate_voxels(batch_params, voxel_params):
    seq = ['|','/','--','\\']
    batchsize = batch_params[0].size()[0]
    totalsize = voxel_params[0].shape[0]
    index = np.arange(batchsize)
    if batchsize==totalsize:
        for _p, p in zip(batch_params, voxel_params):
            set_value(_p, p)
        yield index
    else:
        n = np.ceil(float(totalsize)/batchsize)
        for i,startindex in enumerate(range(0, totalsize, batchsize)):    
            shifted_index = (index + startindex) % totalsize
            sys.stdout.write('\r%-2s: %.1f %%: voxels [%6d:%-6d] of %d' % (seq[i%4], float(i+1)*100/n, shifted_index[0], shifted_index[-1], totalsize))
            for _p, p in zip(batch_params, voxel_params):
                set_value(_p, p[shifted_index])
            yield shifted_index
         
        
################################################################        
def iterate_slice(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield slice(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield slice(start+batch_count*batchsize,start+length),residual    
        

######################################################
def iterate_subject_ordering_minibatches(inputs, targets, image_ordering, batchsize, shuffle=False):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    seq = ['|','/','--','\\']   
    subs, islices, vslices = [], [], []
    for s,d in targets.items():
        for rb,lb in iterate_slice(0, len(d), batchsize):
            subs += [s,]
            islices += [image_ordering[s][rb],]
            vslices += [rb,]
    n = len(subs)
    ordering = np.arange(n)
    if shuffle:
        np.random.shuffle(ordering)
        subs   = np.array(subs)[ordering]
        islices = np.array(islices)[ordering]
        vslices = np.array(vslices)[ordering]
    else:
        subs   = np.array(subs)#, dtype=int)
        islices = np.array(islices)#, dtype=int)
        vslices = np.array(vslices)#, dtype=int)
        
    for i,idx in enumerate(ordering):
        sys.stdout.write('\r%-2s: %.1f %%'%(seq[i%4], float(i+1)*100/n))
        s = subs[idx]
        yield s, inputs[s][islices[idx]], targets[s][vslices[idx]]      
        
        
##################################################################
def subject_training_pass(_trn_fn, _ext, _cons, _ops, x, v, ordering, batch_size):
    trn_err = float(0)
    for s, xb, vb in iterate_subject_ordering_minibatches(x, v, ordering, batch_size, shuffle=True):
        trn_err += get_value(T.mean(_trn_fn(_ext, _cons[s], _ops[s], xb, vb))) 
    return trn_err / sum(len(vv) for s,vv in v.items())

#################################################
def subject_holdout_pass(_hld_fn, _ext, _cons, x, v, ordering, batch_size):
    #val_err = np.zeros(shape=(v.shape[1]), dtype=v.dtype)
    val_err = float(0)
    for s, xb, vb in iterate_subject_ordering_minibatches(x, v, ordering, batch_size):
        val_err += get_value(T.mean(_hld_fn(_ext, _cons[s], xb, vb)))
    return val_err / sum(len(vv) for s,vv in v.items())

#################################################
#def subject_pred_pass(_pred_fn, _ext, _con, x, batch_size): # Backup of original!
#    pred = _pred_fn(_ext, _con, x[:batch_size]) # this is just to get the shape
#    pred = np.zeros(shape=(len(x), pred.shape[1]), dtype=np.float32) # allocate
#    for rb,_ in iterate_range(0, len(x), batch_size):
#        pred[rb] = get_value(_pred_fn(_ext, _con, x[rb]))
#    return pred
def subject_pred_pass(_pred_fn, _ext, _con, x, batch_size):
    n_batches = int(np.ceil(len(x) / batch_size))
    progress_bar = tqdm(range(n_batches), desc='Encoding fMRI responses')
    idx = 0
    for b in progress_bar:
        # Update the progress bar
        encoded_images = batch_size * idx
        progress_bar.set_postfix(
            {'Encoded images': encoded_images, 'Total images': len(x)})
        # Image batch indices
        idx_start = b * batch_size
        idx_end = idx_start + batch_size
        # Generate the in silico fMRI responses
        pred = get_value(_pred_fn(_ext, _con, x[idx_start:idx_end]))
        # Store the in silico fMRI responses
        if b == 0:
            pred_fmri = pred
        else:
            pred_fmri = np.append(pred_fmri, pred, 0)
        del pred
        # Update the progress bar
        idx += 1
        encoded_images = batch_size * idx
        progress_bar.set_postfix(
            {'Encoded images': encoded_images, 'Total images': len(x)})
    pred_fmri = pred_fmri.astype(np.float32)
    return pred_fmri

def subject_validation_pass(_pred_fn, _ext, _con, x, v, ordering, batch_size):
    val_cc  = np.zeros(shape=(v.shape[1]), dtype=v.dtype)
    val_pred = subject_pred_pass(_pred_fn, _ext, _con, x, batch_size)[ordering]
    for i in range(v.shape[1]):
        val_cc[i] = np.corrcoef(v[:,i], val_pred[:, i])[0,1]     ###############             
    return val_cc

#################################################
def random_split(stim, voxel, subselect, trn_size, holdout_frac, random=False):
    n = len(stim)
    holdout_size = int(np.ceil(n * holdout_frac))
    if random:
        idx = np.arange(n)
        np.random.shuffle(idx)  
        idx = idx[:trn_size]
        return stim[idx[:-holdout_size]], voxel[:,subselect][idx[:-holdout_size]], \
               stim[idx[-holdout_size:]], voxel[:,subselect][idx[-holdout_size:]]
    else:
        return stim[:trn_size-holdout_size], voxel[:,subselect][:trn_size-holdout_size], \
               stim[-holdout_size:], voxel[:,subselect][-holdout_size:]

def learn_params_(writer, _trn_fn, _hld_fn, _pred_fn, _ext, _cons, _opts, stims, trn_voxels, trn_stim_ordering, hld_voxels, hld_stim_ordering, num_epochs, batch_size, holdout_frac, trn_size=None, masks=None, randomize=False):
    '''assumes shared_model and subject_fwrfs in global scope
    
       voxelwise model fit is performed for one subject at a time.
    
    '''
    import copy
    #trn_stim_ordering, trn_voxels, hld_stim_ordering, hld_voxels = {},{},{},{}

    for s,v in trn_voxels.items():
        # if masks is None:
        #     mask = np.ones(shape=(v.shape[1]), dtype=bool) 
        # else:
        #     mask = masks[s]
            
        # trn_stim_ordering[s], trn_voxels[s], hld_stim_ordering[s], hld_voxels[s] = \
        #     random_split(ordering[s], v, mask, trn_size=trn_size if trn_size is not None else len(ordering[s]), \
        #     holdout_frac=holdout_frac, random=randomize)
        #print ('subject', s, 'masked', trn_voxels[s].shape[1], 'of', v.shape[1])
        print ('subject', s, 'training/holdout', len(trn_stim_ordering[s]), len(hld_stim_ordering[s]))
    
    
    
    hold_hist, trn_hist = [], []
    hold_cc_hist = {s: [] for s in trn_voxels.keys()}
    best_joint_cc_score = float(0)
    best_params = {}
    best_epoch = 0
    
    for epoch in range(num_epochs):
        ##
        ## Training pass for this subject
        ##
        start_time = time.time()
        _ext.train()
        for s,_c in _cons.items():
            _c.train()
        trn_err = subject_training_pass(_trn_fn, _ext, _cons, _opts, stims, trn_voxels, trn_stim_ordering, batch_size)
        trn_hist += [trn_err,]
        ##
        _ext.eval()
        for s,_c in _cons.items():
            _c.eval()
        hold_err = subject_holdout_pass(_hld_fn, _ext, _cons, stims, hld_voxels, hld_stim_ordering, batch_size)
        hold_hist += [hold_err,]
        ##    
        ## Do a validation pass to monitor the evolution
        ##
        print("\n  Epoch {} of {} took       {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:               {:.6f}".format(trn_err))
        print("  holdout loss (batch):        {:.6f}".format(hold_err))       
        ##
        joint_cc = []
        for s,v in hld_voxels.items():
            hold_cc  = np.nan_to_num( subject_validation_pass(_pred_fn, _ext, _cons[s], stims[s], v, hld_stim_ordering[s], batch_size) )
            joint_cc += [np.copy(hold_cc),]
            hold_cc_hist[s] += [np.copy(hold_cc),]
            print("  Subject {}: median (max) validation accuracy = {:.3f} ({:.3f})".format(s, np.median(np.nan_to_num(hold_cc)), np.max(np.nan_to_num(hold_cc))))
        ##
        ## Save parameter snapshot
        ##
        joint_cc_score = np.median(np.nan_to_num(np.concatenate(joint_cc)))
        if joint_cc_score>best_joint_cc_score:
            print ("** Saving params with joint score = {:.3f} **".format(joint_cc_score))
            best_joint_cc_score = joint_cc_score
            best_epoch = epoch
            best_params = {
                'enc': copy.deepcopy(_ext.state_dict()),
                'fwrfs': {s: copy.deepcopy(_c.state_dict()) for s,_c in _cons.items()}
                }
        print ("")
        sys.stdout.flush()
        writer.add_scalars('Loss', {'train': trn_err, 'val': hold_err}, epoch+1)
        writer.flush()
        writer.add_scalars('Acc', {'val': joint_cc_score}, epoch+1)
        writer.flush()
    ###
    writer.close()
    final_params = {
        'enc': copy.deepcopy(_ext.state_dict()),
        'fwrfs': {s: copy.deepcopy(_c.state_dict()) for s,_c in _cons.items()}
        }    
    
    return best_params, final_params, hold_cc_hist, hold_hist, trn_hist, best_epoch, best_joint_cc_score        



def validation_(_pred_fn, _ext, _cons, stims, voxels, ordering, batch_size, masks=None):
    val_ccs = {}
    _ext.eval()
    for s,v in voxels.items():
        if masks is None:
            mask = np.ones(shape=(v.shape[1]), dtype=bool) 
        else:
            mask = masks[s]
        _ext.eval()
        _cons[s].eval()   
        val_ccs[s] = np.nan_to_num(subject_validation_pass(_pred_fn, _ext, _cons[s], stims[s], v[:,mask], ordering[s], batch_size))
    return val_ccs


