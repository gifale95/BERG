"""Compute the confidence intervals and statistical significance for the
enocoding accuracy and noise analysis results.

Parameters
----------
encoding_models : list
    The names of BERG's encoding models used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list of int
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : list
    List containing the EEG channel type(s) across which the encoding
    accuracies are averaged for the analyses. Possible values are: 'O'
    (occipital), 'P' (posterior), 'T' (temporal), 'C' (central), 'F' (frontal).
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
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['eeg-things_eeg_2-vit_b_32', 'eeg-things_eeg_2-alexnet', 'eeg-things_eeg_2-alexnet_untrained'])
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--channels', default=['O', 'P', 'T', 'C', 'F'], type=list)
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
# Load the encoding accuracy and noise analysis results
# =============================================================================
# Load the results
correlation_single_chan = {}
corr_iv_is_single_chan = {}
corr_iv_iv_single_chan = {}
for model in args.encoding_models:
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'encoding_accuracy', 'encoding_accuracy', model,
        'encoding_accuracy.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    metadata = results['metadata']
    correlation_single_chan[model[17:]] = np.array(results['correlation'])
    corr_iv_is_single_chan[model[17:]] = np.array(results['corr_iv_is'])
    corr_iv_iv_single_chan[model[17:]] = np.array(results['corr_iv_iv'])

# Average the encoding accuracy results across channels from the same channel
# group
correlation = {}
for model in correlation_single_chan.keys():
    correlation[model] = []
    for chan in args.channels:
        idx_chan = []
        for i, ch in enumerate(metadata[0]['eeg']['ch_names']):
            if chan in ch:
                idx_chan.append(i)
        idx_chan = np.array(idx_chan)
        correlation[model].append(np.mean(
            correlation_single_chan[model][:,idx_chan], 1))
    correlation[model] = np.swapaxes(np.array(correlation[model]), 0, 1)

# Average the noise analysis results across occipital and parietal channels,
# and across time points from 60ms after stimulus onset
corr_iv_iv = {}
corr_iv_is = {}
for model in corr_iv_is_single_chan.keys():
    idx_chan = []
    for i, ch in enumerate(metadata[0]['eeg']['ch_names']):
        if 'O' in ch or 'P' in ch:
            idx_chan.append(i)
    idx_chan = np.array(idx_chan)
    c_iv_is = corr_iv_is_single_chan[model][:,idx_chan]
    c_iv_iv = corr_iv_iv_single_chan[model][:,:,idx_chan]
    idx_time = np.where(metadata[0]['eeg']['times'] >= 0.06)[0]
    c_iv_is = c_iv_is[:,:,idx_time]
    c_iv_iv = c_iv_iv[:,:,:,idx_time]
    corr_iv_is[model] = np.mean(c_iv_is, (1,2))
    corr_iv_iv[model] = np.mean(c_iv_iv, (2,3))


# =============================================================================
# Compute the difference in encoding accuracy between encoding models
# =============================================================================
diff_correlation = {}

# "vit_b_32" minus "alexnet"
diff_correlation['vit_b_32_minus_alexnet'] = correlation['vit_b_32'] - \
    correlation['alexnet']

# "vit_b_32" minus "alexnet_untrained"
diff_correlation['vit_b_32_minus_alexnet_untrained'] = \
    correlation['vit_b_32'] - correlation['alexnet_untrained']

# "alexnet" minus "alexnet_untrained"
diff_correlation['alexnet_minus_alexnet_untrained'] = \
    correlation['alexnet'] - correlation['alexnet_untrained']


# =============================================================================
# Encoding accuracy confidence intervals
# =============================================================================
# Encoding accuracy
ci_correlation = {}
for model in correlation.keys():
    ci_correlation[model] = np.zeros((2, correlation[model].shape[1],
        correlation[model].shape[2]))
    dist_correlation = np.zeros((args.n_iter, correlation[model].shape[1],
        correlation[model].shape[2]))
    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.subjects)))
        dist_correlation[i] = np.mean(correlation[model][idx], 0)
    ci_correlation[model][0] = np.percentile(dist_correlation, 2.5, axis=0)
    ci_correlation[model][1] = np.percentile(dist_correlation, 97.5, axis=0)

# Encoding accuracy difference
ci_diff_correlation = {}
for model in diff_correlation.keys():
    ci_diff_correlation[model] = np.zeros((2, diff_correlation[model].shape[1],
        diff_correlation[model].shape[2]))
    dist_diff_correlation = np.zeros((args.n_iter,
        diff_correlation[model].shape[1], diff_correlation[model].shape[2]))
    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.subjects)))
        dist_diff_correlation[i] = np.mean(diff_correlation[model][idx], 0)
    ci_diff_correlation[model][0] = np.percentile(dist_diff_correlation, 2.5,
        axis=0)
    ci_diff_correlation[model][1] = np.percentile(dist_diff_correlation, 97.5,
        axis=0)


# =============================================================================
# Encoding accuracy significance testing
# =============================================================================
# Encoding accuracy
p_val_correlation = {}
sig_correlation = {}
for model in correlation.keys():
    # Compute the p-values for the correlation scores being significantly
    # greater than 0
    p_val = ttest_1samp(correlation[model], 0, axis=0,
        alternative='greater')[1]
    # Correct for multiple comparisons
    shape = p_val.shape
    sig = multipletests(p_val.flatten(), 0.05, 'fdr_bh')[0]
    # Store the p-values and significance
    p_val_correlation[model] = p_val
    sig_correlation[model] = np.reshape(sig, shape)

# Encoding accuracy differences
p_val_diff_correlation = {}
sig_diff_correlation = {}
for model in diff_correlation.keys():
    # Compute the p-values for the correlation scores being significantly
    # greater or smaller than 0
    p_val = ttest_1samp(diff_correlation[model], 0, axis=0,
        alternative='two-sided')[1]
    # Correct for multiple comparisons
    shape = p_val.shape
    sig = multipletests(p_val.flatten(), 0.05, 'fdr_bh')[0]
    # Store the p-values and significance
    p_val_diff_correlation[model] = p_val
    sig_diff_correlation[model] = np.reshape(sig, shape)


# =============================================================================
# Noise analysis confidence intervals and significance testing
# =============================================================================
# Confidence intervals
ci_corr_iv_is = {}
ci_corr_iv_iv = {}
for model in corr_iv_is.keys():
    ci_corr_iv_is[model] = np.zeros((2))
    ci_corr_iv_iv[model] = np.zeros((2, corr_iv_iv[model].shape[1]))
    dist_corr_iv_is = np.zeros((args.n_iter))
    dist_corr_iv_iv = np.zeros((args.n_iter, corr_iv_iv[model].shape[1]))
    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.subjects)))
        dist_corr_iv_is[i] = np.mean(corr_iv_is[model][idx])
        dist_corr_iv_iv[i] = np.mean(corr_iv_iv[model][idx], axis=0)
    ci_corr_iv_is[model][0] = np.percentile(dist_corr_iv_is, 2.5)
    ci_corr_iv_is[model][1] = np.percentile(dist_corr_iv_is, 97.5)
    ci_corr_iv_iv[model][0] = np.percentile(dist_corr_iv_iv, 2.5, axis=0)
    ci_corr_iv_iv[model][1] = np.percentile(dist_corr_iv_iv, 97.5, axis=0)

# Significance testing
p_val_less = {}
p_val_greater = {}
sig_less = {}
sig_greater = {}
for model in corr_iv_is.keys():
    p_less = ttest_rel(corr_iv_iv[model], np.repeat(np.reshape(
        corr_iv_is[model], (len(corr_iv_is[model]), 1)),
        corr_iv_iv[model].shape[1], axis=1), axis=0, alternative='less')[1]
    p_greater = ttest_rel(corr_iv_iv[model], np.repeat(np.reshape(
        corr_iv_is[model], (len(corr_iv_is[model]), 1)),
        corr_iv_iv[model].shape[1], axis=1), axis=0, alternative='greater')[1]
    sig_less[model] = multipletests(p_less, 0.05, 'fdr_bh')[0]
    sig_greater[model] = multipletests(p_greater, 0.05, 'fdr_bh')[0]
    p_val_less[model] = p_less
    p_val_greater[model] = p_greater


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,

    'correlation': correlation,
    'ci_correlation': ci_correlation,
    'p_val_correlation': p_val_correlation,
    'sig_correlation': sig_correlation,

    'diff_correlation': diff_correlation,
    'ci_diff_correlation': ci_diff_correlation,
    'p_val_diff_correlation': p_val_diff_correlation,
    'sig_diff_correlation': sig_diff_correlation,

    'corr_iv_iv': corr_iv_iv,
    'corr_iv_is': corr_iv_is,
    'ci_corr_iv_is': ci_corr_iv_is,
    'ci_corr_iv_iv': ci_corr_iv_iv,
    'p_val_less': p_val_less,
    'p_val_greater': p_val_greater,
    'sig_less': sig_less,
    'sig_greater': sig_greater
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'encoding_accuracy', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)