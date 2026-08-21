"""Compute the DNN layerwise RSA scores as a function of t-fMRI time points,
the best DNN layer as a function of t-fMRI time points, and the corresponding
confidence intervals.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
roi : list
    List of used ROIs.
dnn : str
    Name of the used DNN. Possible values are 'dinov2l'.
images : str
    If 'things_eeg_2_vivo', use the in vivo EEG responses for the 200 THINGS
    EEG2 test images.
    If 'things_eeg_2_silico', use the in silico EEG responses for the 200
    THINGS EEG2 test images.
    If 'nsd_515_shared', use the in silico EEG responses for the 515 NSD shared
    images.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import linregress
from berg import BERG
from tqdm import tqdm
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--rois', default=['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA', 'PPA'], type=list)
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the DNN layerwise RSA results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution',
    'dnn_layerwise_rsa')

# Loop across ROIs
dnn_layerwise_rsa = {}
for r, roi in enumerate(args.rois):

    # Loop across fMRI subjects
    dnn_layerwise_rsa[roi] = []
    for s, sub in enumerate(args.fmri_subjects):

        # Load the results
        file_name = (f'dnn_layerwise_rsa_sub-{sub:02d}_roi-{roi}_'
            f'images-{args.images}.npy')
        dnn_layerwise_rsa_sub = np.load(os.path.join(data_dir, file_name))

        # Sum the results across fMRI subjects
        dnn_layerwise_rsa[roi].append(dnn_layerwise_rsa_sub)
        del dnn_layerwise_rsa_sub

    # Format the results to numpy arrays
    dnn_layerwise_rsa[roi] = np.array(dnn_layerwise_rsa[roi])


# =============================================================================
# Select EEG time points between 60-400ms (where the EEG responses are most
# reliable)
# =============================================================================
# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = np.round(metadata_eeg['eeg']['times'], 3)

# EEG time point selection
idx_times = (times >= 0.06) & (times <= 0.4)
times = times[idx_times]
for key, val in dnn_layerwise_rsa.items():
    dnn_layerwise_rsa[key] = val[:,:,idx_times]


# =============================================================================
# Get the best DNN layer for each t-fMRI time point
# =============================================================================
# Get the best DNN layer for each t-fMRI time point (based on the average
# of the top-5 DNN layers to get a more robust estimate)
best_dnn_layer = {}
for key, val in dnn_layerwise_rsa.items():
    best_dnn_layer[key] = []
    for s in range(len(val)):
        idx_best = np.mean(np.argsort(val[s], 0)[-5:], 0)
        best_dnn_layer[key].append(idx_best)
        del idx_best
    best_dnn_layer[key] = np.array(best_dnn_layer[key])

# Compute the correlation between best DNN layers (averaged across subjects)
# and the t-fMRI time points
corr_dnn_layer_tfmri_times = {}
for key, val in best_dnn_layer.items():
    corr_dnn_layer_tfmri_times[key] = spearmanr(times, np.mean(val, 0))

# Fit a regression line between best DNN layers (averaged across subjects) and
# the t-fMRI time points
reg_best_dnn_layer_tfmri_times = {}
for key, val in best_dnn_layer.items():
    reg_best_dnn_layer_tfmri_times[key] = linregress(times, np.mean(val, 0))


# =============================================================================
# Compute the confidence intervals (CIs)
# =============================================================================
ci_dnn_layerwise_rsa = {}
ci_best_dnn_layer = {}
n_times = len(times)
n_layers = dnn_layerwise_rsa[list(dnn_layerwise_rsa.keys())[0]].shape[1]

for key in dnn_layerwise_rsa.keys():

    ci_dnn_layerwise_rsa[key] = np.zeros((2, n_layers, n_times))
    ci_best_dnn_layer[key] = np.zeros((2, n_times))
    rsa_dist = np.zeros((args.n_iter, n_layers, n_times))
    best_dnn_layer_dist = np.zeros((args.n_iter, n_times))

    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.fmri_subjects)))
        rsa_dist[i] = np.mean(dnn_layerwise_rsa[key][idx], 0)
        best_dnn_layer_dist[i] = np.mean(best_dnn_layer[key][idx], 0)
    ci_dnn_layerwise_rsa[key][0] = np.percentile(rsa_dist, 2.5, axis=0)
    ci_dnn_layerwise_rsa[key][1] = np.percentile(rsa_dist, 97.5, axis=0)
    ci_best_dnn_layer[key][0] = np.percentile(best_dnn_layer_dist, 2.5, axis=0)
    ci_best_dnn_layer[key][1] = np.percentile(best_dnn_layer_dist, 97.5, axis=0)


# =============================================================================
# Save the stats
# =============================================================================
results = {
    'times': times,
    'dnn_layerwise_rsa': dnn_layerwise_rsa,
    'best_dnn_layer': best_dnn_layer,
    'corr_dnn_layer_tfmri_times': corr_dnn_layer_tfmri_times,
    'reg_best_dnn_layer_tfmri_times': reg_best_dnn_layer_tfmri_times,
    'ci_dnn_layerwise_rsa': ci_dnn_layerwise_rsa,
    'ci_best_dnn_layer': ci_best_dnn_layer
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_dnn-{args.dnn}_images-{args.images}.npy'

np.save(os.path.join(save_dir, file_name), results)