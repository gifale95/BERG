"""Compute the stats on the results of the RSA analysis between in silico EEG
responses and behavioral embeddings. The stats consist of bootstrapped 95%
confidence intervals and significance estimates.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
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
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the RSA results
# =============================================================================
rsa = []

for sub in args.subjects:

    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'behavioral_modeling_correlation_rdms', 'rsa', args.encoding_model, 'rsa_sub-'+
        format(sub,'02')+'_channels-'+'-'.join(args.channels)+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    rsa.append(results['rsa'])
    times = results['metadata']['eeg']['times']

# Convert to numpy arrays
rsa = np.asarray(rsa)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_rsa = np.zeros((2, len(times)))
ci_peak_latency_ci_rsa = np.zeros((2))

rsa_dist = np.zeros((args.n_iter, len(times)))
peak_lat_rsa_dist = np.zeros((args.n_iter))

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    rsa_dist[i] = np.mean(rsa[idx], 0)
    peak_lat_rsa_dist[i] = times[np.argmax(np.mean(rsa[idx], 0))]

ci_rsa[0] = np.percentile(rsa_dist, 2.5, axis=0)
ci_rsa[1] = np.percentile(rsa_dist, 97.5, axis=0)
ci_peak_latency_ci_rsa[0] = np.percentile(peak_lat_rsa_dist, 2.5, axis=0)
ci_peak_latency_ci_rsa[1] = np.percentile(peak_lat_rsa_dist, 97.5, axis=0)


# =============================================================================
# Compute the significance
# =============================================================================
# Compute the p-values with t-tests
pval_rsa = ttest_1samp(rsa, 0, axis=0, alternative='greater')[1]

# Correct for multiple comparisons
sig_rsa, pval_rsa_corrected, _, _ = multipletests(pval_rsa, 0.05, 'fdr_bh')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'ci_rsa': ci_rsa,
    'ci_peak_latency_ci_rsa': ci_peak_latency_ci_rsa,
    'pval_rsa': pval_rsa,
    'sig_rsa': sig_rsa,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'behavioral_modeling_correlation_rdms', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_' + 'channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results)