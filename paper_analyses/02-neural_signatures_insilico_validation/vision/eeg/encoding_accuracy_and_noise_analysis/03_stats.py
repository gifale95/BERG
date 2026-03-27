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
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['eeg-things_eeg_2-vit_b_32', 'eeg-things_eeg_2-alexnet', 'eeg-things_eeg_2-alexnet_untrained'])
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
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
correlation = {}
corr_iv_iv = {}
corr_iv_is = {}

for model in args.encoding_models:

    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'encoding_accuracy', 'encoding_accuracy', model,
        'encoding_accuracy.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    correlation[model[17:]] = np.array(results['correlation'])
    corr_iv_iv[model[17:]] = np.array(results['corr_iv_iv'])
    corr_iv_is[model[17:]] = np.array(results['corr_iv_is'])
    metadata = results['metadata']


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


# =============================================================================
# Encoding accuracy confidence intervals # !!!
# =============================================================================
# Encoding accuracy
p_val_correlation_nsdcore = {}
sig_correlation_nsdcore = {}
for key in correlation_nsdcore.keys():
    p_val_correlation_nsdcore[key] = {}
    p_val_correlation_nsdsynthetic[key] = {}
    sig_correlation_nsdcore[key] = {}
    sig_correlation_nsdsynthetic[key] = {}
    # Append the correlation scores across hemispheres
    nsdcore = np.append(correlation_nsdcore[key]['lh'],
        correlation_nsdcore[key]['rh'], 1)
    nsdsynthetic = np.append(correlation_nsdsynthetic[key]['lh'],
        correlation_nsdsynthetic[key]['rh'], 1)
    # Compute the p-values for the correlation scores being significantly
    # greater than 0
    p_val_nsdcore = ttest_1samp(nsdcore, 0, axis=0, alternative='greater')[1]
    p_val_nsdsynthetic = ttest_1samp(nsdsynthetic, 0, axis=0,
        alternative='greater')[1]
    # Correct for multiple comparisons
    sig_nsd_core = multipletests(p_val_nsdcore, 0.05, 'fdr_bh')[0]
    sig_nsdsynthetic = multipletests(p_val_nsdsynthetic, 0.05, 'fdr_bh')[0]
    # Store the p-values and significance
    p_val_correlation_nsdcore[key]['lh'] = p_val_nsdcore[:163842]
    p_val_correlation_nsdcore[key]['rh'] = p_val_nsdcore[163842:]
    p_val_correlation_nsdsynthetic[key]['lh'] = p_val_nsdsynthetic[:163842]
    p_val_correlation_nsdsynthetic[key]['rh'] = p_val_nsdsynthetic[163842:]
    sig_correlation_nsdcore[key]['lh'] = sig_nsd_core[:163842]
    sig_correlation_nsdcore[key]['rh'] = sig_nsd_core[163842:]
    sig_correlation_nsdsynthetic[key]['lh'] = sig_nsdsynthetic[:163842]
    sig_correlation_nsdsynthetic[key]['rh'] = sig_nsdsynthetic[163842:]

# Encoding accuracy differences
p_val_diff_correlation_nsdcore = {}
p_val_diff_correlation_nsdsynthetic = {}
sig_diff_correlation_nsdcore = {}
sig_diff_correlation_nsdsynthetic = {}
for key in diff_correlation_nsdcore.keys():
    p_val_diff_correlation_nsdcore[key] = {}
    p_val_diff_correlation_nsdsynthetic[key] = {}
    sig_diff_correlation_nsdcore[key] = {}
    sig_diff_correlation_nsdsynthetic[key] = {}
    # Append the correlation scores across hemispheres
    diff_nsdcore = np.append(diff_correlation_nsdcore[key]['lh'],
        diff_correlation_nsdcore[key]['rh'], 1)
    diff_nsdsynthetic = np.append(diff_correlation_nsdsynthetic[key]['lh'],
        diff_correlation_nsdsynthetic[key]['rh'], 1)
    # Compute the p-values for the correlation score differences being
    # significantly greater or smaller than 0
    p_val_nsdcore = ttest_1samp(diff_nsdcore, 0, axis=0,
        alternative='two-sided')[1]
    p_val_nsdsynthetic = ttest_1samp(diff_nsdsynthetic, 0,
        axis=0, alternative='two-sided')[1]
    # Correct for multiple comparisons
    sig_nsdcore = multipletests(p_val_nsdcore, 0.05, 'fdr_bh')[0]
    sig_nsdsynthetic = multipletests(p_val_nsdsynthetic, 0.05, 'fdr_bh')[0]
    # Store the p-values
    p_val_diff_correlation_nsdcore[key]['lh'] = p_val_nsdcore[:163842]
    p_val_diff_correlation_nsdcore[key]['rh'] = p_val_nsdcore[163842:]
    p_val_diff_correlation_nsdsynthetic[key]['lh'] = p_val_nsdsynthetic[:163842]
    p_val_diff_correlation_nsdsynthetic[key]['rh'] = p_val_nsdsynthetic[163842:]
    sig_diff_correlation_nsdcore[key]['lh'] = sig_nsdcore[:163842]
    sig_diff_correlation_nsdcore[key]['rh'] = sig_nsdcore[163842:]
    sig_diff_correlation_nsdsynthetic[key]['lh'] = sig_nsdsynthetic[:163842]
    sig_diff_correlation_nsdsynthetic[key]['rh'] = sig_nsdsynthetic[163842:]


# =============================================================================
# Encoding accuracy significance testing
# =============================================================================
# Encoding accuracy
p_val_correlation = {}
sig_correlation = {}
for key in correlation.keys():
    # Compute the p-values for the correlation scores being significantly
    # greater than 0
    p_val = ttest_1samp(correlation[model], 0, axis=0,
        alternative='greater')[1]
    # Correct for multiple comparisons
    shape = p_val.shape
    sig = multipletests(p_val.flatten(), 0.05, 'fdr_bh')[0]
    # Store the p-values and significance
    p_val_correlation[key] = p_val
    sig_correlation[key] = sig.reshape(sig, shape)

# Encoding accuracy differences
p_val_diff_correlation = {}
sig_diff_correlation = {}
for key in diff_correlation.keys():
    # Compute the p-values for the correlation scores being significantly
    # greater or smaller than 0
    p_val = ttest_1samp(diff_correlation[model], 0, axis=0,
        alternative='two-sided')[1]
    # Correct for multiple comparisons
    shape = p_val.shape
    sig = multipletests(p_val.flatten(), 0.05, 'fdr_bh')[0]
    # Store the p-values and significance
    p_val_diff_correlation[key] = p_val
    sig_diff_correlation[key] = sig.reshape(sig, shape)


# =============================================================================
# Noise analysis confidence intervals and significance testing # !!!
# =============================================================================
# Bootstrap the confidence intervals
ci_corr_iv1tr_is = {}
ci_corr_iv1tr_iv1tr = {}
ci_corr_iv1tr_iv2tr = {}
for model in corr_iv1tr_is_avg.keys():
    ci_corr_iv1tr_is[model] = np.zeros((2))
    ci_corr_iv1tr_iv1tr[model] = np.zeros((2))
    ci_corr_iv1tr_iv2tr[model] = np.zeros((2))
    dist_corr_iv1tr_is = np.zeros((args.n_iter))
    dist_corr_iv1tr_iv1tr = np.zeros((args.n_iter))
    dist_corr_iv1tr_iv2tr = np.zeros((args.n_iter))
    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.subjects)))
        dist_corr_iv1tr_is[i] = np.mean(corr_iv1tr_is_avg[model][idx])
        dist_corr_iv1tr_iv1tr[i] = np.mean(corr_iv1tr_iv1tr_avg[model][idx])
        dist_corr_iv1tr_iv2tr[i] = np.mean(corr_iv1tr_iv2tr_avg[model][idx])
    ci_corr_iv1tr_is[model][0] = np.percentile(dist_corr_iv1tr_is, 2.5)
    ci_corr_iv1tr_is[model][1] = np.percentile(dist_corr_iv1tr_is, 97.5)
    ci_corr_iv1tr_iv1tr[model][0] = np.percentile(dist_corr_iv1tr_iv1tr, 2.5)
    ci_corr_iv1tr_iv1tr[model][1] = np.percentile(dist_corr_iv1tr_iv1tr, 97.5)
    ci_corr_iv1tr_iv2tr[model][0] = np.percentile(dist_corr_iv1tr_iv2tr, 2.5)
    ci_corr_iv1tr_iv2tr[model][1] = np.percentile(dist_corr_iv1tr_iv2tr, 97.5)

# Significance testing
p_val_1 = {}
p_val_2 = {}
for model in corr_iv1tr_is_avg.keys():

    p_1 = ttest_rel(corr_iv1tr_iv2tr_avg[model], corr_iv1tr_iv1tr_avg[model],
        alternative='greater')[1]
    p_2 = ttest_rel(corr_iv1tr_is_avg[model], corr_iv1tr_iv2tr_avg[model],
        alternative='greater')[1]
    pval = np.append(p_1, p_2)
    pval_corrected = multipletests(pval, 0.05, 'fdr_bh')[1]
    p_val_1[model] = pval_corrected[0]
    p_val_2[model] = pval_corrected[1]


# =============================================================================
# Save the results # !!!
# =============================================================================
results = {
    'metadata': metadata,

    'correlation': correlation_nsdcore,
    'p_val_correlation': p_val_correlation,
    'sig_correlation': sig_correlation,

    'diff_correlation': diff_correlation_nsdcore,
    'p_val_diff_correlation': p_val_diff_correlation,
    'sig_diff_correlation': sig_diff_correlation,

    'corr_iv_iv': corr_iv_iv,
    'corr_iv_is': corr_iv_is,
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'encoding_accuracy', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)