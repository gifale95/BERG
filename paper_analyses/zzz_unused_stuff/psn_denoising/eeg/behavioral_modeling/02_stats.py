"""Compute the stats on the results of the RSA analysis between in silico EEG
responses and behavioral embeddings. The stats consist of bootstrapped 95%
confidence intervals and significance estimates.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
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
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--psn_mode', default=1, type=int)
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
# Load the decoding and RSA results
# =============================================================================
decoding = {}
rsa = {}

for s, sub in enumerate(args.subjects):

    data_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
        'behavioral_modeling', 'rsa', 'rsa_sub-'+format(sub, '02')+
        '_psn_mode-'+str(args.psn_mode)+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    # Get the decoding results
    for key, val in results['eeg_rdm'].items():
        if s == 0:
            decoding[key] = []
            idx_tril = np.tril_indices(len(val), -1)
        decoding[key].append(np.mean(val[idx_tril], 0))

    # Get the RSA results
    for key, val in results['rsa'].items():
        if s == 0:
            rsa[key] = []
        rsa[key].append(val)

    # EEG metadata
    times = results['metadata']['times']

# Convert to numpy arrays
for key, val in decoding.items():
    decoding[key] = np.asarray(val) * 100
for key, val in rsa.items():
    rsa[key] = np.asarray(val)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_decoding = {}
ci_rsa = {}

for key in rsa.keys():

    ci_decoding[key] = np.zeros((2, len(times))) # type: ignore
    ci_rsa[key] = np.zeros((2, len(times))) # type: ignore

    decoding_dist = np.zeros((args.n_iter, len(times))) # type: ignore
    rsa_dist = np.zeros((args.n_iter, len(times))) # type: ignore

    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.subjects)))
        decoding_dist[i] = np.mean(decoding[key][idx], 0)
        rsa_dist[i] = np.mean(rsa[key][idx], 0)

    ci_decoding[key][0] = np.percentile(decoding_dist, 2.5, axis=0)
    ci_decoding[key][1] = np.percentile(decoding_dist, 97.5, axis=0)
    ci_rsa[key][0] = np.percentile(rsa_dist, 2.5, axis=0)
    ci_rsa[key][1] = np.percentile(rsa_dist, 97.5, axis=0)


# =============================================================================
# Compute the significance
# =============================================================================
sig_decoding = {}
sig_rsa = {}

for key in rsa.keys():

    # Compute the p-values with t-tests
    pval_decoding = ttest_1samp(decoding[key], 50, axis=0, alternative='greater')[1]
    pval_rsa = ttest_1samp(rsa[key], 0, axis=0, alternative='greater')[1]

    # Correct for multiple comparisons
    sig_decoding[key] = multipletests(pval_decoding, 0.05, 'fdr_bh')[0]
    sig_rsa[key] = multipletests(pval_rsa, 0.05, 'fdr_bh')[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'decoding': decoding,
    'rsa': rsa,
    'ci_decoding': ci_decoding,
    'ci_rsa': ci_rsa,
    'sig_decoding': sig_decoding,
    'sig_rsa': sig_rsa,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
        'behavioral_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_psn_mode-' + str(args.psn_mode) + '.npy'

np.save(os.path.join(save_dir, file_name), results)