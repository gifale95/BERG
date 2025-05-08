import sys
import os
import struct
import numpy as np
from tqdm import tqdm

import torch
import torch as T
from src_new.file_utility import zip_dict
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF

def sample_with_replacement(indices):
    return indices[np.random.randint(len(indices), size=len(indices))]
def cc_resampling_with_replacement(_pred_fn, _ext, _con, x, v, ordering, batch_size, n_resample=1):
    pred = subject_pred_pass(_pred_fn, _ext, _con, x, batch_size)[ordering]
    cc = np.zeros(shape=(v.shape[1]), dtype=v.dtype)
    ccs = []
    for rs in tqdm(range(n_resample)):
        res = sample_with_replacement(np.arange(len(pred)))
        data_res = v[res]
        pred_res = pred[res]
        for i in range(v.shape[1]):
            cc[i] = np.corrcoef(data_res[:,i], pred_res[:,i])[0,1]  
        ccs += [np.nan_to_num(cc)]
    return np.array([np.mean(np.array(ccs),axis=0), np.std(np.array(ccs),axis=0)]).T


def gnet8j_predictions(image_data, _pred_fn, checkpoint, batch_size, device=torch.device("cuda:0")):
    
    subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()}    
    # allocate
    subject_image_pred = {s: np.zeros(shape=(len(image_data[s]), nv), dtype=np.float32) for s,nv in subject_nv.items()}
    _log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)
     
    best_params = checkpoint['best_params']
    shared_model = Encoder(np.array(checkpoint['input_mean']).astype(np.float32), trunk_width=64).to(device)
    shared_model.load_state_dict(best_params['enc'])
    shared_model.eval() 

    # example fmaps
    rec, fmaps, h = shared_model(T.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))                                                                                                                     
    for s,param in best_params['fwrfs'].items():
        sd = Torch_LayerwiseFWRF(fmaps, nv=subject_nv[s], pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
        sd.load_state_dict(param)
        sd.eval() 
        
        subject_image_pred[s] = subject_pred_pass(_pred_fn, shared_model, sd, image_data[s], batch_size)

    return subject_image_pred


def subjectwise_gnet8r_predictions(image_data, _pred_fn, checkpoint, subj, batch_size, device=torch.device("cuda:0")):
    '''The reason for the code duplication is for ressouce management. We only define the shared feature extractor once whether we need a single subject or all of them. This could be improved.'''
    masks = checkpoint[list(checkpoint.keys())[0]]['group_mask']
    subject_nv = len(masks[subj])    
    # allocate
    subject_image_pred = np.zeros(shape=(len(image_data), subject_nv), dtype=np.float32)
    _log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)
        
    for roi, cp in checkpoint.items(): 
        group_masks = cp['group_mask']
        best_params = cp['best_params']

        shared_model = Encoder(np.array(cp['input_mean']).astype(np.float32), trunk_width=64).to(device)
        shared_model.load_state_dict(best_params['enc'])
        shared_model.eval() 

        # example fmaps
        rec, fmaps, h = shared_model(T.from_numpy(image_data[:20]).to(device))
                                                                                                                                     
        param = best_params['fwrfs'][subj]
        sd = Torch_LayerwiseFWRF(fmaps, nv=np.sum(group_masks[subj]), pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
        sd.load_state_dict(param)
        sd.eval() 
        
        subject_image_pred[:, group_masks[subj]] = subject_pred_pass(_pred_fn, shared_model, sd, image_data, batch_size)
#    return the combined prediction for all subjects
    return subject_image_pred


def gnet8r_predictions(image_data, _pred_fn, checkpoint, batch_size, device=torch.device("cuda:0")):
    
    masks = checkpoint[list(checkpoint.keys())[0]]['group_mask']
    subject_nv = {s: len(v) for s,v in masks.items()}    
    # allocate
    subject_image_pred = {s: np.zeros(shape=(len(image_data[s]), nv), dtype=np.float32) for s,nv in subject_nv.items()}
    _log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)
        
    for roi, cp in checkpoint.items(): 
        print ('predicting %s voxels'%roi)
        group_masks = cp['group_mask']
        best_params = cp['best_params']

        shared_model = Encoder(np.array(cp['input_mean']).astype(np.float32), trunk_width=64).to(device)
        shared_model.load_state_dict(best_params['enc'])
        shared_model.eval() 
        # example fmaps
        rec, fmaps, h = shared_model(T.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))                                 
        for s,param in best_params['fwrfs'].items():
            sd = Torch_LayerwiseFWRF(fmaps, nv=np.sum(group_masks[s]), pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
            sd.load_state_dict(param)
            sd.eval() 

            subject_image_pred[s][:, group_masks[s]] = subject_pred_pass(_pred_fn, shared_model, sd, image_data[s], batch_size)
#    return the combined prediction for all subjects
    return subject_image_pred




def gnet8j_tuning_analysis(image_data, voxel_data, stim_ordering, _pred_fn, checkpoint, tuning_masks, batch_size, n_resample=1, device=torch.device("cuda:0")):
    subject_total_val_cc          = {}
    subject_partition_incl_val_cc = {}
    subject_partition_excl_val_cc = {}

    best_params = checkpoint['best_params']
    
    _log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)
    shared_model = Encoder(np.array(checkpoint['input_mean']).astype(np.float32), trunk_width=64).to(device)
    shared_model.load_state_dict(best_params['enc'])
    shared_model.eval()     
    
    rec, fmaps, h = shared_model(T.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))
                 
    for s,v,p in zip_dict(voxel_data, best_params['fwrfs']):
    
        nv = v.shape[1]
        sd = Torch_LayerwiseFWRF(fmaps, nv=nv, pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
        sd.load_state_dict(p)
        sd.eval() 
          
        total_val_cc = cc_resampling_with_replacement(\
            _pred_fn, shared_model, sd, image_data[s], v, stim_ordering[s], batch_size, n_resample=n_resample)    
            
        weights = get_value(p['w'])
        partition_incl_val_cc   = np.ndarray(shape=(len(tuning_masks), nv, 2), dtype=v.dtype)
        partition_excl_val_cc   = np.ndarray(shape=(len(tuning_masks), nv, 2), dtype=v.dtype)      
        for l,rl in tqdm(enumerate(tuning_masks)):
            ##### Inclusion #####            
            partition_w = np.zeros_like(weights)
            partition_w[:, rl] = weights[:, rl]
            set_value(sd.w, partition_w)
            sd.eval() 
            
            partition_incl_val_cc[l,:] = cc_resampling_with_replacement(\
                _pred_fn, shared_model, sd, image_data[s], v, stim_ordering[s], batch_size, n_resample=n_resample)
            ##### Exlusion ####            
            partition_w = np.copy(weights)
            partition_w[:, rl] = 0
            set_value(sd.w, partition_w)
            sd.eval() 
            
            partition_excl_val_cc[l,:] = cc_resampling_with_replacement(\
                _pred_fn, shared_model, sd, image_data[s], v, stim_ordering[s], batch_size, n_resample=n_resample)       
         
        subject_total_val_cc[s]          = total_val_cc 
        subject_partition_incl_val_cc[s] = partition_incl_val_cc 
        subject_partition_excl_val_cc[s] = partition_excl_val_cc 

    
    return subject_total_val_cc, subject_partition_incl_val_cc, subject_partition_excl_val_cc
    
    
    
def gnet8r_tuning_analysis(image_data, voxel_data, stim_ordering, _pred_fn, checkpoint, tuning_masks, batch_size, n_resample=1, device=torch.device("cuda:0")):
        
    masks = checkpoint[list(checkpoint.keys())[0]]['group_mask']
    # allocate    
    subject_total_val_cc          = {s: np.zeros(shape=(len(m), 2), dtype=np.float32) for s,m in masks.items()}
    subject_partition_incl_val_cc = {s: np.zeros(shape=(len(tuning_masks), len(m), 2), dtype=np.float32) for s,m in masks.items()}
    subject_partition_excl_val_cc = {s: np.zeros(shape=(len(tuning_masks), len(m), 2), dtype=np.float32) for s,m in masks.items()}
    
    _log_act_fn = lambda _x: T.log(1 + T.abs(_x))*T.tanh(_x)  
    #######
    for group, cp in checkpoint.items(): 
        
        group_masks = cp['group_mask']
        best_params = cp['best_params']
        #######
        shared_model = Encoder(np.array(cp['input_mean']).astype(np.float32), trunk_width=64).to(device)
        shared_model.load_state_dict(best_params['enc'])
        shared_model.eval()     

        rec, fmaps, h = shared_model(T.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))

        for s,v,m,p in zip_dict(voxel_data, group_masks, best_params['fwrfs']):

            nv = np.sum(m) # v.shape[1]
            voxels = v[:,m]
            sd = Torch_LayerwiseFWRF(fmaps, nv=nv, pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
            sd.load_state_dict(p)
            sd.eval() 

            total_val_cc = cc_resampling_with_replacement(\
                _pred_fn, shared_model, sd, image_data[s], voxels, stim_ordering[s], batch_size, n_resample=n_resample)            
            
            weights = get_value(p['w'])
            partition_incl_val_cc   = np.ndarray(shape=(len(tuning_masks), nv, 2), dtype=v.dtype)
            partition_excl_val_cc   = np.ndarray(shape=(len(tuning_masks), nv, 2), dtype=v.dtype)    
            for l,rl in tqdm(enumerate(tuning_masks)):
                ##### Inclusion #####            
                partition_w = np.zeros_like(weights)
                partition_w[:, rl] = weights[:, rl]
                set_value(sd.w, partition_w)
                sd.eval() 
                
                partition_incl_val_cc[l,:] = cc_resampling_with_replacement(\
                    _pred_fn, shared_model, sd, image_data[s], voxels, stim_ordering[s], batch_size, n_resample=n_resample) 

                ##### Exlusion ####            
                partition_w = np.copy(weights)
                partition_w[:, rl] = 0
                set_value(sd.w, partition_w)
                sd.eval() 
                
                partition_excl_val_cc[l,:] = cc_resampling_with_replacement(\
                    _pred_fn, shared_model, sd, image_data[s], voxels, stim_ordering[s], batch_size, n_resample=n_resample)        

            subject_total_val_cc[s][m]             = total_val_cc
            subject_partition_incl_val_cc[s][:, m] = partition_incl_val_cc 
            subject_partition_excl_val_cc[s][:, m] = partition_excl_val_cc 

    
    return subject_total_val_cc, subject_partition_incl_val_cc, subject_partition_excl_val_cc
   


def tuning_and_spread(subject_val_cc, subject_incl_val_cc, subject_excl_val_cc):
    
    subject_incl_tunings, subject_incl_tunings_err, subject_incl_tuning_argmax, subject_incl_tuning_spread = {},{},{},{}
    subject_excl_tunings, subject_excl_tunings_err, subject_excl_tuning_argmax, subject_excl_tuning_spread = {},{},{},{}
    
    for s, val_ccs, incl_ccs, excl_ccs in zip_dict(subject_val_cc, subject_incl_val_cc, subject_excl_val_cc):
        
        val_cc , val_err  = val_ccs[:,0],  val_ccs[:,1]
        incl_cc, incl_err = incl_ccs[:,:,0], incl_ccs[:,:,1]
        excl_cc, excl_err = excl_ccs[:,:,0], excl_ccs[:,:,1]
        
        partition_excl_variance = np.square(val_cc) - np.square(np.nan_to_num(excl_cc))
        #partition_excl_var_err  = np.sqrt((2*val_cc*val_err)**2 + (2*excl_cc*excl_err)**2)
        
        excl_var_min = np.min(partition_excl_variance, axis=0, keepdims=True)
        excl_var_max = np.max(partition_excl_variance, axis=0, keepdims=True)
        excl_tuning_scores = np.nan_to_num(np.sort((partition_excl_variance - excl_var_min) / (excl_var_max - excl_var_min), axis=0))
        ###
        excl_tuning = np.nan_to_num(partition_excl_variance / np.sum(partition_excl_variance, axis=0, keepdims=True))
        excl_tuning_argmax = np.argmax(partition_excl_variance, axis=0)
        excl_tuning_spread = np.zeros_like(val_cc)
        for v in tqdm(range(len(val_cc))):
            excl_tuning_spread[v] = np.interp(0.5, excl_tuning_scores[:,v], np.linspace(0.,1.,len(excl_tuning_scores), endpoint=True)[::-1])
            
        partition_incl_variance = np.square(np.nan_to_num(incl_cc))
        partition_incl_var_err  = 2*incl_cc*incl_err
        
        incl_var_min = np.min(partition_incl_variance, axis=0, keepdims=True)
        incl_var_max = np.max(partition_incl_variance, axis=0, keepdims=True)
        incl_tuning_scores = np.nan_to_num(np.sort((partition_incl_variance - incl_var_min) / (incl_var_max - incl_var_min), axis=0))
        ###
        incl_tuning = np.nan_to_num(partition_incl_variance / np.sum(partition_incl_variance, axis=0, keepdims=True))
        incl_tuning_argmax = np.argmax(partition_incl_variance, axis=0)
        incl_tuning_spread = np.zeros_like(val_cc)
        for v in tqdm(range(len(val_cc))):  
            incl_tuning_spread[v] = np.interp(0.5, incl_tuning_scores[:,v], np.linspace(0.,1.,len(incl_tuning_scores),endpoint=True)[::-1])
            
            
        subject_incl_tunings[s] = partition_incl_variance * 100. / np.square(val_cc)
        subject_incl_tunings_err[s] =  100. * (2 * incl_cc**2/val_cc**2) * np.sqrt((val_err/val_cc)**2 + (incl_err/incl_cc)**2)
        #100. / np.square(val_cc) * np.sqrt(partition_incl_var_err**2 + (2*incl_cc*val_err/val_cc)**2)
        subject_incl_tuning_argmax[s] = incl_tuning_argmax
        subject_incl_tuning_spread[s] = incl_tuning_spread
        
        subject_excl_tunings[s] = partition_excl_variance * 100. / np.square(val_cc)
        subject_excl_tunings_err[s] = 100. * (2 * excl_cc**2/val_cc**2) * np.sqrt((val_err/val_cc)**2 + (excl_err/excl_cc)**2)
        subject_excl_tuning_argmax[s] = excl_tuning_argmax
        subject_excl_tuning_spread[s] = excl_tuning_spread        
           
    return subject_incl_tunings, subject_incl_tunings_err, subject_incl_tuning_argmax, subject_incl_tuning_spread,\
           subject_excl_tunings, subject_excl_tunings_err, subject_excl_tuning_argmax, subject_excl_tuning_spread





from matplotlib import cm
def pooling_fn(x):
    return np.exp(x) / np.sum(np.exp(x), axis=(1,2), keepdims=True)

from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx

def twoD_Gaussian(xy, amplitude, xo, yo, sx, sy, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    sigma_x = np.exp(sx)
    sigma_y = np.exp(sy)
    A = np.exp(amplitude)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = np.sin(2*theta) * (1/(2*sigma_y**2) - 1/(2*sigma_x**2))
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + A*np.exp( - (a*((xy[0]-xo)**2) + b*(xy[0]-xo)*(xy[1]-yo) + c*((xy[1]-yo)**2)))
    return g.ravel()

def batched_2D_Gaussian_fits(Zs):
    import scipy.optimize as opt
    rf_params, rf_params_err, rf_R = [], [], []
    X, Y = np.meshgrid(np.linspace(-.5,.5,Zs.shape[1]), np.linspace(-.5,.5,Zs.shape[2])[::-1])
    for k,Z in enumerate(Zs):
        w = np.maximum(Z - np.mean(Z), 0)
        # calculate center of mass
        x0,y0 = np.sum(X*w)/np.sum(w), np.sum(Y*w)/np.sum(w)
        xs,ys = np.sqrt(np.sum((X-x0)**2*w)/np.sum(w)), np.sqrt(np.sum((Y-y0)**2*w)/np.sum(w))
        initial_guess = (np.max(Z), x0, y0, np.log(xs), np.log(ys), 0., 0.)    
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, (X.flatten(),Y.flatten()), Z.flatten(), p0=initial_guess)   
        except (RuntimeError, opt.OptimizeWarning):
            rf_params     += [(0., 0., 0., 1., 1., 0., 0.),]
            rf_params_err += [(0., 0., 0., 1., 1., 0., 0.),]
            rf_R += [-1,]
            continue

        data_fitted = twoD_Gaussian((X.flatten(), Y.flatten()), *popt)
        rf_R          += [np.sqrt(np.sum(np.square(Z.flatten() - data_fitted))),]        
        rf_params     += [popt,]
        rf_params_err += [np.sqrt(np.diag(pcov)),]   
    return np.array(rf_params), np.array(rf_params_err), np.array(rf_R)

def calc_angle(x, y):
    '''return the min angle between 2 angles between -pi..pi'''
    t = y - x
    tp = 2*np.pi + y - x
    tm = -2*np.pi + y - x
    idx = np.argmin(np.abs(np.concatenate([t[np.newaxis], tp[np.newaxis], tm[np.newaxis]], axis=0)), axis=0)
    return t*(idx==0) + tp*(idx==1) + tm*(idx==2)

def calc_phi(rf_params):
    is_ymaj = rf_params[:,3]<rf_params[:,4]
    theta = np.arctan2(rf_params[:,2], rf_params[:,1])
    phi = -(rf_params[:,5] + is_ymaj * np.pi/2) % (2*np.pi) - np.pi
    phi_op = (phi - np.pi) * (phi>0) + (phi + np.pi) * (phi<=0)

    delta_1 = calc_angle(theta, phi)
    delta_2 = calc_angle(theta, phi_op)
    delta   = (delta_1 * (np.abs(delta_1)<np.abs(delta_2)) + delta_2 * (np.abs(delta_1)>=np.abs(delta_2)))
    return delta
