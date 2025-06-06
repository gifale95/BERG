import sys
import os
import numpy as np
import h5py
from scipy.io import loadmat


def load_beta_file(filename, voxel_mask=None, zscore=True):
    from src_new.file_utility import load_mask_from_nii
    if ".mat" in filename:
        beta_data_set = h5py.File(filename, 'r')
        values = np.copy(beta_data_set['betas'])
        print (values.dtype, np.min(values), np.max(values), values.shape)
        if voxel_mask is None:
            beta = values.reshape((len(values), -1), order='F').astype(np.float32) / 300.
        else:
            beta = values.reshape((len(values), -1), order='F')[:,voxel_mask.flatten()].astype(np.float32) / 300.
        beta_data_set.close()
    elif ".nii" in filename:
        values = load_mask_from_nii(filename).transpose((3,0,1,2))
        print (values.dtype, np.min(values), np.max(values), values.shape)
        if voxel_mask is None:
            beta = values.reshape((len(values), -1)).astype(np.float32) / 300.
        else:
            beta = values.reshape((len(values), -1))[:,voxel_mask.flatten()].astype(np.float32) / 300.               
    elif ".h5" in filename:
        print (".h5 not yet implemented")
        return None
    else:
        print ("Unknown file format")
        return None
    ###
    if zscore: 
        mb = np.mean(beta, axis=0, keepdims=True)
        sb = np.std(beta, axis=0, keepdims=True)
        beta = np.nan_to_num((beta - mb) / (sb + 1e-6))
        print ("<beta> = %.3f, <sigma> = %.3f" % (np.mean(mb), np.mean(sb)))
    return beta
     


def load_betas(folder_name, zscore=False, voxel_mask=None, up_to=0, load_ext='.mat'):
    '''load beta value in the structure of the NSD experiemnt'''
    from src_new.file_utility import list_files
    matfiles, betas = [], []
    k = 0
    for filename in list_files(folder_name):
        filename_no_path = filename.split('/')[-1]
        if 'betas' in filename_no_path and load_ext in filename_no_path:
            k += 1
            if up_to>0 and k>up_to:
                break
            print (filename) 
            matfiles += [filename,]  
            betas += [ load_beta_file(filename, voxel_mask=voxel_mask, zscore=zscore), ]       
    return np.concatenate(betas, axis=0), matfiles
    
    
def image_feature_fn(image):
    '''take uint8 image and return floating point (0,1), either color or bw'''
    return image.astype(np.float32) / 255

def image_uncolorize_fn(image):
    data = image.astype(np.float32) / 255
    return (0.2126*data[:,0:1]+ 0.7152*data[:,1:2]+ 0.0722*data[:,2:3])
    
    
    
def ordering_split(voxel, ordering, val_idx, combine_trial=False):
    data_size, nv = voxel.shape 
    print ("Total number of voxels = %d" % nv)
    ordering_data = ordering[:data_size]
    shared_mask = ordering_data<1000  # the first 1000 indices are the shared indices
    val_mask = np.isin(ordering_data, val_idx)
    hld_mask = (shared_mask.astype(np.int64) - val_mask.astype(np.int64)).astype(np.bool)
    if combine_trial:        
        idx, idx_count = np.unique(ordering_data, return_counts=True)
        idx_list = [ordering_data==i for i in idx]
        voxel_avg_data = np.zeros(shape=(len(idx), nv), dtype=np.float32)
        for i,m in enumerate(idx_list):
            voxel_avg_data[i] = np.mean(voxel[m], axis=0)
        shared_mask_mt = idx<1000

        val_voxel_data = voxel_avg_data[shared_mask_mt] 
        val_stim_ordering = idx[shared_mask_mt]   

        trn_voxel_data = voxel_avg_data[~shared_mask_mt]
        trn_stim_ordering = idx[~shared_mask_mt]              
        
    else:
        val_voxel_data = voxel[val_mask]    
        val_stim_ordering  = ordering_data[val_mask]
        hld_voxel_data = voxel[hld_mask]    
        hld_stim_ordering  = ordering_data[hld_mask]
        trn_voxel_data = voxel[~shared_mask]
        trn_stim_ordering  = ordering_data[~shared_mask]
        
    return trn_stim_ordering, trn_voxel_data, hld_stim_ordering, hld_voxel_data, val_stim_ordering, val_voxel_data



def data_split(stim, voxel, ordering, imagewise=True):
    data_size, nv = voxel.shape 
    print ("Total number of voxels = %d" % nv)
    ordering_data = ordering[:data_size]
    shared_mask = ordering_data<1000  # the first 1000 indices are the shared indices

    val_voxel_st = voxel[shared_mask]    
    val_stim_st  = stim[ordering_data[shared_mask]]
 
    idx, idx_count = np.unique(ordering_data, return_counts=True)
    idx_list = [ordering_data==i for i in idx]
    voxel_avg_data = np.zeros(shape=(len(idx), nv), dtype=np.float32)
    for i,m in enumerate(idx_list):
        voxel_avg_data[i] = np.mean(voxel[m], axis=0)
    shared_mask_mt = idx<1000

    val_voxel_mt = voxel_avg_data[shared_mask_mt]  
    val_stim_mt  = stim[idx][shared_mask_mt]        
    
    if imagewise:
        trn_voxel = voxel_avg_data[~shared_mask_mt]
        trn_stim  = stim[idx][~shared_mask_mt] 
        return trn_stim, trn_voxel, val_stim_st, val_voxel_st, val_stim_mt, val_voxel_mt
    else:
        trn_voxel = voxel[~shared_mask]
        trn_stim = stim[ordering_data[~shared_mask]]
        return trn_stim, trn_voxel, val_stim_st, val_voxel_st, val_stim_mt, val_voxel_mt
