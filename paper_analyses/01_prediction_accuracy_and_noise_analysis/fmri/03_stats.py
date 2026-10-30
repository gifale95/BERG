"""Compute the confidence intervals and statistical significance for the
enocoding accuracy and noise analysis results.

Parameters
----------
encoding_models : list
    The names of BERG's encoding models used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list of int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemispheres : list of str
    List of strings containing the hemispheres used for the analyses.
    Possible values  are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from scipy.stats import binom
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['fmri-nsd_fsaverage-huze', 'fmri-nsd_fsaverage-alexnet', 'fmri-nsd_fsaverage-alexnet_untrained'])
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
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
rng = np.random.default_rng(seed=seed)


# =============================================================================
# Load and format the prediction accuracy and noise analysis results
# =============================================================================
correlation_nsdcore = {}
correlation_nsdsynthetic = {}
metadata_berg = {}
corr_iv1tr_is_avg = {}
corr_iv1tr_iv2tr_avg = {}
corr_iv1tr_iv1tr_avg = {}
corr_iv1tr_is = {}
corr_iv1tr_iv2tr = {}
corr_iv1tr_iv1tr = {}

for model in args.encoding_models:

    results_dir = os.path.join(args.berg_dir,
        'prediction_accuracy_noise_analysis', 'fmri', 'prediction_accuracy',
        model, 'prediction_accuracy.npy')
    results = np.load(results_dir, allow_pickle=True).item()

    metadata = results['metadata']
    correlation_nsdcore[model[19:]] = results['correlation_nsdcore']
    correlation_nsdsynthetic[model[19:]] = results['correlation_nsdsynthetic']

    corr_iv1tr_is_avg[model[19:]] = results['corr_iv1tr_is_avg']
    corr_iv1tr_iv2tr_avg[model[19:]] = results['corr_iv1tr_iv2tr_avg']
    corr_iv1tr_iv1tr_avg[model[19:]] = results['corr_iv1tr_iv1tr_avg']

    corr_iv1tr_is[model[19:]] = []
    corr_iv1tr_iv2tr[model[19:]] = []
    corr_iv1tr_iv1tr[model[19:]] = []
    for s in range(len(args.subjects)):
        for h, hemi in enumerate(args.hemispheres):
            idx_non_nan = ~np.isnan(results['corr_iv1tr_is'][hemi][s])
            if h == 0:
                corr_iv1tr_is_hemi = results['corr_iv1tr_is'][hemi][s]
                corr_iv1tr_iv2tr_hemi = results['corr_iv1tr_iv2tr'][hemi][s]
                corr_iv1tr_iv1tr_hemi = results['corr_iv1tr_iv1tr'][hemi][s]
            else:
                corr_iv1tr_is_hemi = np.append(corr_iv1tr_is_hemi,
                    results['corr_iv1tr_is'][hemi][s], 0)
                corr_iv1tr_iv2tr_hemi = np.append(corr_iv1tr_iv2tr_hemi,
                    results['corr_iv1tr_iv2tr'][hemi][s], 0)
                corr_iv1tr_iv1tr_hemi = np.append(corr_iv1tr_iv2tr_hemi,
                    results['corr_iv1tr_iv1tr'][hemi][s], 0)
        corr_iv1tr_is[model[19:]].append(corr_iv1tr_is_hemi)
        corr_iv1tr_iv2tr[model[19:]].append(corr_iv1tr_iv2tr_hemi)
        corr_iv1tr_iv1tr[model[19:]].append(corr_iv1tr_iv1tr_hemi)
    corr_iv1tr_is[model[19:]] = np.array(corr_iv1tr_is[model[19:]])
    corr_iv1tr_iv2tr[model[19:]] = np.array(corr_iv1tr_iv2tr[model[19:]])
    corr_iv1tr_iv1tr[model[19:]] = np.array(corr_iv1tr_iv1tr[model[19:]])


# =============================================================================
# Noise analysis confidence intervals
# =============================================================================
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


# =============================================================================
# Noise analysis significance testing (only for the Huze model)
# =============================================================================
# Get the actual difference between the noise analysis predictors
model = 'huze'
diff_1 = corr_iv1tr_iv2tr_avg[model] - corr_iv1tr_iv1tr_avg[model]
diff_2 = corr_iv1tr_is_avg[model] - corr_iv1tr_iv2tr_avg[model]

# Create the difference null distribution using permuted t-fMRI
# responses
diff_1_null_dist = np.zeros((len(args.subjects), args.n_iter),
    dtype=np.float32)
diff_2_null_dist = np.zeros((len(args.subjects), args.n_iter),
    dtype=np.float32)
appended_data = np.append(corr_iv1tr_iv2tr[model], corr_iv1tr_iv1tr[model], 1)
appended_data = np.append(appended_data, corr_iv1tr_is[model], 1)
for i in tqdm(range(args.n_iter)):
    n_vertex = len(corr_iv1tr_iv2tr[model][0])
    idx_perm = rng.choice(n_vertex*3, size=n_vertex*3, replace=False)
    for s in range(len(args.subjects)):
        diff_1_null_dist[s,i] = \
            np.nanmean(appended_data[s,idx_perm[:n_vertex]]) - \
            np.nanmean(appended_data[s,idx_perm[n_vertex:2*n_vertex]])
        diff_2_null_dist[s,i] = \
            np.nanmean(appended_data[s,idx_perm[2*n_vertex:]]) - \
            np.nanmean(appended_data[s,idx_perm[:n_vertex]])

# Compute the within-subject significance
within_subj_pval_1 = np.zeros((len(args.subjects)))
within_subj_pval_2 = np.zeros((len(args.subjects)))
for s in range(len(args.subjects)):
    within_subj_pval_1[s] = (np.sum(diff_1_null_dist[s] >= diff_1[s]) + 1) / \
        (args.n_iter + 1)
    within_subj_pval_2[s] = (np.sum(diff_2_null_dist[s] >= diff_2[s]) + 1) / \
        (args.n_iter + 1)
within_subj_sig_1 = within_subj_pval_1 < 0.01
within_subj_sig_2 = within_subj_pval_2 < 0.01

# Compute the population-level significance based on population
# prevalence: this is the probability of observing k or more
# significant results by chance, based on the CDF of the binomial
# distribution of within-subject significances.
n = len(args.subjects) # Total number of subjects
p = 0.05 # probability of success in each trial
k_1 = sum(within_subj_sig_1) # Number of significant subjects
k_2 = sum(within_subj_sig_2) # Number of significant subjects
# We use "k-1" because otherwise we would get the probability of observing
# k+1 or more significant results by chance
between_subj_pval_1 = 1 - binom.cdf(k_1-1, n, p)
between_subj_pval_2 = 1 - binom.cdf(k_2-1, n, p)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,

    'correlation_nsdcore': correlation_nsdcore,
    'correlation_nsdsynthetic': correlation_nsdsynthetic,

    'corr_iv1tr_is_avg': corr_iv1tr_is_avg,
    'corr_iv1tr_iv2tr_avg': corr_iv1tr_iv2tr_avg,
    'corr_iv1tr_iv1tr_avg': corr_iv1tr_iv1tr_avg,
    'ci_corr_iv1tr_is': ci_corr_iv1tr_is,
    'ci_corr_iv1tr_iv2tr': ci_corr_iv1tr_iv2tr,
    'ci_corr_iv1tr_iv1tr': ci_corr_iv1tr_iv1tr,
    'between_subj_pval_1': between_subj_pval_1,
    'between_subj_pval_2': between_subj_pval_2
}

save_dir = os.path.join(args.berg_dir, 'prediction_accuracy_noise_analysis',
    'fmri', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)