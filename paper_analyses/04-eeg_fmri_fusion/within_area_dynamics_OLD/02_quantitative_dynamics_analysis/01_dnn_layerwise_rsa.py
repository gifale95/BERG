"""Perform RSA between t-fMRI time point RSMs, and the DNN layerwise
activation RSMs.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
roi : str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest used to find the
    baseline and controlling images.
use_time_bins: int
    If '1', average the t-fMRI responses into four time bins (50-100ms,
    100-150ms, 150-200ms, 200-250ms). If '0', do not average the t-fMRI
    responses into time bins.
dnn : str
    Name of the used DNN. Possible values are 'dinov2l'.
correlation_measure: str
    Whether to use 'pearson' or 'spearman' correlation.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.05-0.10__0.20-0.25', type=str)
parser.add_argument('--use_time_bins', default=1, type=int)
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--correlation_measure', default='pearson', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> DNN layerwise RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the t-fMRI RSMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'tfmri_rsms')

file_name = (f'tfmri_rsms_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}_'
    f'use_time_bins-{args.use_time_bins}.npy')

tfmri_rsms = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()


# =============================================================================
# Load the DNN RSMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'dnn_rsms')

file_name = (f'dnn_rsms_dnn-{args.dnn}_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}.npy')

dnn_rsms = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()


# =============================================================================
# Perform the DNN layerwise RSA
# =============================================================================
dnn_layerwise_rsa = {} 

for key in tqdm(tfmri_rsms.keys()):

    n_times = tfmri_rsms[key].shape[2]
    n_layers = dnn_rsms[key].shape[2]
    n_img = tfmri_rsms[key].shape[0]
    dnn_layerwise_rsa[key] = np.ones((n_layers, n_times))
    idx_tril = np.tril_indices(n_img, k=-1)

    for t in range(n_times):
        for l in range(n_layers):
            if args.correlation_measure == 'pearson':
                dnn_layerwise_rsa[key][l,t] = pearsonr(
                    dnn_rsms[key][:,:,l][idx_tril],
                    tfmri_rsms[key][:,:,t][idx_tril])[0]
            elif args.correlation_measure == 'spearman':
                dnn_layerwise_rsa[key][l,t] = spearmanr(
                    dnn_rsms[key][:,:,l][idx_tril],
                    tfmri_rsms[key][:,:,t][idx_tril])[0]


# =============================================================================
# Save the RSA results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'dnn_layerwise_rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'dnn_layerwise_rsa_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}_'
    f'use_time_bins-{args.use_time_bins}_dnn-{args.dnn}_'
    f'corr-{args.correlation_measure}.npy')

np.save(os.path.join(save_dir, file_name), dnn_layerwise_rsa)