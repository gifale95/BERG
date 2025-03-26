import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math

import nest.models.fwrf.numpy_utility as pnu

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim
from nest.models.fwrf.numpy_utility import iterate_range

def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)        

class Torch_fwRF_voxel_block(nn.Module):
    '''
    This is a variant of the fwRF model as a module for a voxel block (we can't have it all at once)
    '''

    def __init__(self, _fmaps_fn, params, _nonlinearity=None, input_shape=(1,3,227,227), aperture=1.0):
        super(Torch_fwRF_voxel_block, self).__init__()
        
        self.aperture = aperture
        models, weights, bias, mstmt, mstst = params
        device = next(_fmaps_fn.parameters()).device
        _x =torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
        
        self.pfs = []
        for k,n_pix in enumerate(self.fmaps_rez):
            pf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            self.pfs += [nn.Parameter(torch.from_numpy(pf).to(device), requires_grad=False),]
            self.register_parameter('pf%d'%k, self.pfs[-1])
            
        self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
            
        self.mstm = None
        self.msts = None
        if mstmt is not None:
            self.mstm = nn.Parameter(torch.from_numpy(mstmt.T).to(device), requires_grad=False)
        if mstst is not None:
            self.msts = nn.Parameter(torch.from_numpy(mstst.T).to(device), requires_grad=False)
        self._nl = _nonlinearity
              
    def load_voxel_block(self, *params):
        models = params[0]
        for _pf,n_pix in zip(self.pfs, self.fmaps_rez):
            pf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=self.aperture, dtype=np.float32)[2]
            if len(pf)<_pf.size()[0]:
                pp = np.zeros(shape=_pf.size(), dtype=pf.dtype)
                pp[:len(pf)] = pf
                set_value(_pf, pp)
            else:
                set_value(_pf, pf)
        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp)
                else:
                    set_value(_p, p)
        for _p,p in zip([self.mstm, self.msts], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp.T)
                else:
                    set_value(_p, p.T)
 
    def forward(self, _fmaps):
        _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [1,2]]) for _fm,_pf in zip(_fmaps, self.pfs)], dim=1) # [#samples, #features, #voxels] 
        if self._nl is not None:
            _mst = self._nl(_mst)
        if self.mstm is not None:              
            _mst -= self.mstm[None]
        if self.msts is not None:
            _mst /= self.msts[None]
        _mst = torch.transpose(torch.transpose(_mst, 0, 2), 1, 2) # [#voxels, #samples, features]
        _r = torch.squeeze(torch.bmm(_mst, torch.unsqueeze(self.weights, 2))).t() # [#samples, #voxels]
        if self.bias is not None:
            _r += torch.unsqueeze(self.bias, 0)
        return _r



def learn_params_ridge_regression(data, voxels, _fmaps_fn, models, lambdas, aperture=1.0, _nonlinearity=None, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False):
    """
    Learn the parameters of the fwRF model

    Parameters
    ----------
    data : ndarray, shape (#samples, #channels, x, y)
        Input image block.
    voxels: ndarray, shape (#samples, #voxels)
        Input voxel activities.
    _fmaps_fn: Torch module
        Torch module that returns a list of torch tensors.
    models: ndarray, shape (#candidateRF, 3)
        The (x, y, sigma) of all candidate RFs for gridsearch.
    lambdas: ndarray, shape (#candidateRegression)
        The rigde parameter candidates.
    aperture (default: 1.0): scalar
        The span of the stimulus in the unit used for the RF models.
    _nonlinearity (default: None)
        A nonlinearity expressed with torch's functions.
    zscore (default: False)
        Whether to zscore the feature maps or not.
    sample_batch_size (default: 100)
        The sample batch size (used where appropriate)
    voxel_batch_size (default: 100) 
        The voxel batch size (used where appropriate)
    holdout_size (default: 100) 
        The holdout size for model and hyperparameter selection
    shuffle (default: True)
        Whether to shuffle the training set or not.
    add_bias (default: False)
        Whether to add a bias term to the rigde regression or not.

    Returns
    -------
    losses : ndarray, shape (#voxels)
        The final loss for each voxel.
    lambdas : ndarray, shape (#voxels)
        The regression regularization index for each voxel.
    models : ndarray, shape (#voxels, 3)
        The RF model (x, y, sigma) associated with each voxel.
    params : list of ndarray, shape (#voxels, #features)
        Can contain a bias parameter of shape (#voxels) if add_bias is True.
    mst_mean : ndarray, shape (#voxels, #feature)
        None if zscore is False. Otherwise returns zscoring average per feature.
    mst_std : ndarray, shape (#voxels, #feature)
        None if zscore is False. Otherwise returns zscoring std.dev. per feature.
    """
    def _cofactor_fn(_x, lambdas):
        '''input matrix [#samples, #features], a list of lambda values'''
        _f = torch.stack([(torch.mm(torch.t(_x), _x) + torch.eye(_x.size()[1], device=device) * l).inverse() for l in lambdas], axis=0) # [#lambdas, #feature, #feature]       
        return torch.tensordot(_f, _x, dims=[[2],[1]]) # [#lambdas, #feature, #sample]
    
    def _loss_fn(_cofactor, _vtrn, _xout, _vout):
        '''input '''
        _beta = torch.tensordot(_cofactor, _vtrn, dims=[[2], [0]]) # [#lambdas, #feature, #voxel]
        _pred = torch.tensordot(_xout, _beta, dims=[[1],[1]]) # [#samples, #lambdas, #voxels]
        _loss = torch.sum(torch.pow(_vout[:,None,:] - _pred, 2), dim=0) # [#lambdas, #voxels]
        return _beta, _loss
    
    #############################################################################
    dtype = data.dtype.type
    device = next(_fmaps_fn.parameters()).device
    trn_size = len(voxels) - holdout_size
    assert trn_size>0, 'Training size needs to be greater than zero'
    print ('trn_size = %d (%.1f%%)' % (trn_size, float(trn_size)*100/len(voxels)))
    print ('dtype = %s' % dtype)
    print ('device = %s' % device)
    print ('---------------------------------------')
    # shuffle
    nt = len(data)
    nm = len(models)
    nv = voxels.shape[1]
    order = np.arange(len(voxels), dtype=int)
    if shuffle:
        np.random.shuffle(order)
    data   = data[order]
    voxels = voxels[order]  
    trn_voxels = voxels[:trn_size]
    out_voxels = voxels[trn_size:]
    ### Calculate total feature count
    nf = 0
    _fmaps = _fmaps_fn(_to_torch(data[:sample_batch_size], device=device))

    fmaps_rez = []
    for k,_fm in enumerate(_fmaps):
        nf += _fm.size()[1]
        assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
        fmaps_rez += [_fm[k].size()[2],]
        print (_fm.size())    
    print ('---------------------------------------')
    #############################################################################        
    ### Create full model value buffers    
    best_models = np.full(shape=(nv,), fill_value=-1, dtype=np.int)   
    best_lambdas = np.full(shape=(nv,), fill_value=-1, dtype=np.int)
    best_losses = np.full(fill_value=np.inf, shape=(nv), dtype=dtype)
    best_w_params = np.zeros(shape=(nv, nf), dtype=dtype)
    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    mst_mean = None
    mst_std = None
    if zscore:
        mst_mean = np.zeros(shape=(nv, nf), dtype=dtype)
        mst_std  = np.zeros(shape=(nv, nf), dtype=dtype)
    
    start_time = time.time()
    vox_loop_time = 0
    print ('')
    with torch.no_grad():
        for m,(x,y,sigma) in enumerate(models):
            mst = np.zeros(shape=(nt, nf), dtype=dtype)
            _pfs = [_to_torch(pnu.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, dtype=dtype)[2], device=device) for n_pix in fmaps_rez]
            for rt,rl in iterate_range(0, nt, sample_batch_size):
                _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [0,1]]) for _fm,_pf in zip(_fmaps_fn(_to_torch(data[rt], device=device)), _pfs)], dim=1) # [#samples, #features]
                if _nonlinearity is not None:
                    _mst = _nonlinearity(_mst)
                mst[rt] = get_value(_mst)
            if zscore:  
                mstm = np.mean(mst, axis=0, keepdims=True) #[:trn_size]
                msts = np.std(mst, axis=0, keepdims=True) + 1e-6          
                mst -= mstm
                mst /= msts    
            if add_bias:
                mst = np.concatenate([mst, np.ones(shape=(len(mst), 1), dtype=dtype)], axis=1)
            trn_mst = mst[:trn_size]
            out_mst = mst[trn_size:]   

            _xtrn = _to_torch(trn_mst, device=device)
            _xout = _to_torch(out_mst, device=device)           
            _cof = _cofactor_fn(_xtrn, lambdas)
            ###    
            vox_start = time.time()
            for rv,lv in iterate_range(0, nv, voxel_batch_size):
                sys.stdout.write('\rmodel %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, nm, rv[0], rv[-1], nv))

                _vtrn = _to_torch(trn_voxels[:,rv], device=device)
                _vout = _to_torch(out_voxels[:,rv], device=device)

                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                _values, _select = torch.min(_loss, dim=0)

                betas = get_value(_betas)
                values, select = get_value(_values), get_value(_select)
                imp = values<best_losses[rv]
                if np.sum(imp)>0:
                    arv = np.array(rv)[imp]
                    li = select[imp]
                    best_lambdas[arv] = li
                    best_losses[arv] = values[imp]
                    best_models[arv] = m
                    if zscore:
                        mst_mean[arv] = mstm # broadcast over updated voxels
                        mst_std[arv]  = msts
                    best_w_params[arv,:] = pnu.select_along_axis(betas[:,:,imp], li, run_axis=2, choice_axis=0).T
            vox_loop_time += (time.time() - vox_start)
        
    #############################################################################   
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,:nf],]
    if add_bias:
        return_params += [best_w_params[:,-1],]
    else: 
        return_params += [None,]
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / nv))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / nv))
    print ('setup throughput = %fs/model' % (inv_time / nm))
    sys.stdout.flush()
    return best_losses, best_lambdas, [models[best_models],]+return_params+[mst_mean, mst_std]


    
def get_predictions(data, _fmaps_fn, _fwrf_fn, params, sample_batch_size=100):
    """
    The predictive fwRF model for arbitrary input image.

    Parameters
    ----------
    data : ndarray, shape (#samples, #channels, x, y)
        Input image block.
    _fmaps_fn: Torch module
        Torch module that returns a list of torch tensors.
    _fwrf_fn: Torch module
    Torch module that compute the fwrf model for one batch of voxels
    params: list including all of the following:
    [
        models : ndarray, shape (#voxels, 3)
            The RF model (x, y, sigma) associated with each voxel.
        weights : ndarray, shape (#voxels, #features)
            Tuning weights
        bias: Can contain a bias parameter of shape (#voxels) if add_bias is True.
           Tuning biases: None if there are no bias
        mst_mean (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring average per feature.
        mst_std (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring std.dev. per feature.
    ]
    sample_batch_size (default: 100)
        The sample batch size (used where appropriate)

    Returns
    -------
    pred : ndarray, shape (#samples, #voxels)
        The prediction of voxel activities for each voxels associated with the input data.
    """
    dtype = data.dtype.type
    device = next(_fmaps_fn.parameters()).device
    _params = [_p for _p in _fwrf_fn.parameters()]
    voxel_batch_size = _params[0].size()[0]    
    nt, nv = len(data), len(params[0])
    #print ('val_size = %d' % nt)
    pred = np.full(fill_value=0, shape=(nt, nv), dtype=dtype)
    start_time = time.time()
    with torch.no_grad():
        for rv, lv in iterate_range(0, nv, voxel_batch_size):
            _fwrf_fn.load_voxel_block(*[p[rv] if p is not None else None for p in params])
            pred_block = np.full(fill_value=0, shape=(nt, voxel_batch_size), dtype=dtype)
            for rt, lt in iterate_range(0, nt, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], nt, rv[0], rv[-1], nv))
                pred_block[rt] = get_value(_fwrf_fn(_fmaps_fn(_to_torch(data[rt], device)))) 
            pred[:,rv] = pred_block[:,:lv]
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / nt))
    print ('voxel throughput = %fs/voxel' % (total_time / nv))
    sys.stdout.flush()
    return pred

