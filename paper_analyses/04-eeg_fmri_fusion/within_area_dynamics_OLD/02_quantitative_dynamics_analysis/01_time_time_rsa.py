"""Perform time-time RSA on the t-fMRI response RSMs.

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
parser.add_argument('--correlation_measure', default='pearson', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Time-time RSA <<<')
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
# Perform the time-time RSA
# =============================================================================
time_time_rsa = {} 

for key, val in tqdm(tfmri_rsms.items()):

    n_times = val.shape[2]
    time_time_rsa[key] = np.ones((n_times, n_times))
    idx_tril = np.tril_indices(val.shape[0], k=-1)

    for t1 in range(n_times):
        for t2 in range(t1):
            if args.correlation_measure == 'pearson':
                time_time_rsa[key][t1,t2] = pearsonr(
                    val[:,:,t1][idx_tril], val[:,:,t2][idx_tril])[0]
            elif args.correlation_measure == 'spearman':
                time_time_rsa[key][t1,t2] = spearmanr(
                    val[:,:,t1][idx_tril], val[:,:,t2][idx_tril])[0]
            time_time_rsa[key][t2,t1] = time_time_rsa[key][t1,t2]


# =============================================================================
# Save the RSA results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'tfmri_time_time_rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'tfmri_time_time_rsa_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}_'
    f'use_time_bins-{args.use_time_bins}_corr-{args.correlation_measure}.npy')

np.save(os.path.join(save_dir, file_name), time_time_rsa)