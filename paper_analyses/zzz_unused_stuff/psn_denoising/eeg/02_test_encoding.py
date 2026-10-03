"""Test the THINGS EEG2 encoding models while optionally denoising the neural
responses with PSN.

PSN GitHub: https://github.com/jacob-prince/PSN

Parameters
----------
subjects : list
    List of subject identifier sfor the THINGS EEG2 data. Valid subject
    identifiers are integers from 1 to 10.
psn_mode : int
    PSN mode, randing from 1 to 5.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--psn_mode', default=1, type=int)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the EEG channel names and time points
# =============================================================================
metadata_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'eeg_test_responses', f'psn_mode-{args.psn_mode}', 'eeg_metadata.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()

ch_names = metadata['ch_names']
times = metadata['times']


# =============================================================================
# Loop across subjects
# =============================================================================
ncsnr = {}
noise_ceiling = {}
correlation = {}
n_chan = len(ch_names)
n_time = len(times)

data_types = [
    'invivo_eeg_vte-0',
    'invivo_eeg_vte-1',
    'insilico_eeg_vtr-0_ste-0',
    'insilico_eeg_vtr-1_ste-0',
    'insilico_eeg_vtr-0_ste-1',
    'insilico_eeg_vtr-1_ste-1'
]

for s, sub in enumerate(tqdm(args.subjects)):


# =============================================================================
# Load the EEG test responses
# =============================================================================
    eeg_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'eeg_test_responses', f'psn_mode-{args.psn_mode}',
    'eeg_test_subject-'+format(sub, '02')+'.npy')
    eeg = np.load(eeg_dir, allow_pickle=True).item()


# =============================================================================
# Compute the NCSNR and noise ceiling
# =============================================================================
    # Loop across test EEG response instances
    for key, val in eeg.items():

        # Reshape the EEG responses to: (n_cond, n_trial, n_units)
        val = np.reshape(val, (val.shape[0], val.shape[1], -1))

        # Empty result variables
        if s == 0:
            ncsnr[key] = []
            noise_ceiling[key] = []

        # Estimate the noise standard deviation (calculate the variance of the
        # responses across the 30 presentations of each test image).
        var = np.nanvar(val, axis=1, ddof=1)

        # Average the variance across images and compute the square root of the
        # result
        sigma_noise = np.sqrt(np.nanmean(var, 0))

        # Estimate the signal standard deviation (total variance - noise variance)
        tot_var_data = np.nanvar(np.reshape(val, (-1, val.shape[2])), axis=0,
            ddof=1)
        sigma_signal = tot_var_data - (sigma_noise ** 2)
        sigma_signal[sigma_signal<0] = 0
        sigma_signal = np.sqrt(sigma_signal)

        # Compute the ncsnr
        ncsnr_sub = sigma_signal / sigma_noise

        # Convert the ncsnr to noise ceiling (the noise ceiling is in r²
        # explained variance units)
        n_trial = val.shape[1]
        noise_ceiling_sub = 100 * (ncsnr_sub ** 2) / ((ncsnr_sub ** 2) + (1 / n_trial))

        # Store the results
        ncsnr[key].append(ncsnr_sub)
        noise_ceiling[key].append(noise_ceiling_sub)
        del ncsnr_sub, noise_ceiling_sub


# =============================================================================
# Compute the encoding accuracy (Pearson's r)
# =============================================================================
    for i1 in range(len(data_types)):
        for i2 in range(i1):

            # Get the data
            key_1 = data_types[i1]
            key_2 = data_types[i2]
            val_1 = eeg[key_1]
            val_2 = eeg[key_2]

            # Empty result variables
            if s == 0:
                correlation[key_1+'_vs_'+key_2] = []

            # Average the EEG test responses across repeats
            val_1 = np.mean(val_1, 1)
            val_2 = np.mean(val_2, 1)

            # Reshape the EEG responses to: (n_cond, n_units)
            val_1 = np.reshape(val_1, (val_1.shape[0], -1))
            val_2 = np.reshape(val_2, (val_2.shape[0], -1))

            # Compute the encoding accuracy
            corr_sub = np.zeros(val_1.shape[1], dtype=np.float32)
            for u in range(val_1.shape[1]):
                corr_sub[u] = pearsonr(val_1[:,u], val_2[:,u])[0]

            # Store the results
            correlation[key_1+'_vs_'+key_2].append(corr_sub)
            del corr_sub


# =============================================================================
# Reformat the results to numpy arrays
# =============================================================================
# NCSNR
for key, val in ncsnr.items():
    ncsnr[key] = np.asarray(val)

# Noise ceiling
for key, val in noise_ceiling.items():
    noise_ceiling[key] = np.asarray(val)

# Correlation
for key, val in correlation.items():
    correlation[key] = np.asarray(val)


# =============================================================================
# Confidence intervals
# =============================================================================
# NCSNR and noise ceiling
ci_ncsnr = {}
ci_noise_ceiling = {}
for key in ncsnr.keys():
    ci_ncsnr[key] = np.zeros((2, n_chan*n_time))
    ci_noise_ceiling[key] = np.zeros((2, n_chan*n_time))
    ncsnr_dist = np.zeros((args.n_iter, n_chan*n_time))
    nc_dist = np.zeros((args.n_iter, n_chan*n_time))
    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.subjects)))
        ncsnr_dist[i] = np.mean(ncsnr[key][idx], 0)
        nc_dist[i] = np.mean(noise_ceiling[key][idx], 0)
    ci_ncsnr[key][0] = np.percentile(ncsnr_dist, 2.5, axis=0)
    ci_ncsnr[key][1] = np.percentile(ncsnr_dist, 97.5, axis=0)
    ci_noise_ceiling[key][0] = np.percentile(nc_dist, 2.5, axis=0)
    ci_noise_ceiling[key][1] = np.percentile(nc_dist, 97.5, axis=0)

# Encoding accuracy
ci_correlation = {}
for key, val in correlation.items():
    ci_correlation[key] = np.zeros((2, n_chan*n_time))
    corr_dist = np.zeros((args.n_iter, n_chan*n_time))
    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.subjects)))
        corr_dist[i] = np.mean(val[idx], 0)
    ci_correlation[key][0] = np.percentile(corr_dist, 2.5, axis=0)
    ci_correlation[key][1] = np.percentile(corr_dist, 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'ncsnr': ncsnr,
    'noise_ceiling': noise_ceiling,
    'correlation': correlation,
    'ci_ncsnr': ci_ncsnr,
    'ci_noise_ceiling': ci_noise_ceiling,
    'ci_correlation': ci_correlation,
    'ch_names': ch_names,
    'times': times
    }

save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg', 'test_encoding',
    f'psn_mode-{args.psn_mode}')
os.makedirs(save_dir, exist_ok=True)

file_name = 'test_encoding_stats.npy'

np.save(os.path.join(save_dir, file_name), results)