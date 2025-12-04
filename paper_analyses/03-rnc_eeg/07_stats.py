"""This code tests whether the controlling images found using the in silico
EEG responses of the train subjects generalize to the in silico EEG responses
for the left-out subject. Stats include confidence intervals and significance.

Parameters
----------
all_subjects : list
    List with the subject identifiers of the 10 THINGS EEG2 subjects.
cv : int
    If '1' multivariate RNC leaves the data of one subject out for
    cross-validation, if '0' multivariate RNC uses the data of all subjects.
time_pair : str
    Used pairwise time point combination.
control_conditions : list
    List of used control conditions.
n_iter : int
    Amount of iterations for the permutation stats.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from copy import copy
import random
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

from utils import load_rsms
from utils import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--time_pair', type=str, default='0.1-0.2')
parser.add_argument('--control_conditions', type=list, default=['align', 'disentangle'])
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Multivariate RNC stats <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Time point names
# =============================================================================
idx = args.time_pair.find('-')
time_1 = args.time_pair[:idx]
time_2 = args.time_pair[idx+1:]


# =============================================================================
# Load the mutivariate RNC results
# =============================================================================
best_generation_scores_train = {}
best_generation_image_batches = {}

if args.cv == 0:
    for c in args.control_conditions:
        data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
            'best_image_batches', 'cv-'+format(args.cv), args.time_pair,
            'control_condition-'+c, 'best_image_batches.npy')
        results = np.load(data_dir, allow_pickle=True).item()
        best_generation_scores_train[c] = \
            results['image_batches_scores']
        best_generation_image_batches[c] = \
            results['best_generation_image_batches']

elif args.cv == 1:
    for c in args.control_conditions:
        best_gen_scores_train = []
        best_gen_chromosomes = []
        for sub in args.all_subjects:
            data_dir = os.path.join(args.berg_dir, 'rnc_eeg',
                'multivariate_rnc', 'best_image_batches',
                'cv-'+format(args.cv), args.time_pair, 'control_condition-'+c,
                'best_image_batches_subject-'+format(sub, '02')+'.npy')
            results = np.load(data_dir, allow_pickle=True).item()
            best_gen_scores_train.append(copy(
                results['image_batches_scores']))
            best_gen_chromosomes.append(copy(
                results['best_generation_image_batches']))
        best_generation_scores_train[c] = np.asarray(best_gen_scores_train)
        best_generation_image_batches[c] = np.asarray(best_gen_chromosomes)


# =============================================================================
# Validate the neural control conditions across subjects
# =============================================================================
# Get the test subjects RSA scores for the controlling images from the two
# neural control conditions.

if args.cv == 1:

    # Cross-validate
    best_generation_scores_test = {}
    for c in args.control_conditions:
        scores_test = np.zeros((len(args.all_subjects),
            best_generation_image_batches[c].shape[1]))
        for s, sub in enumerate(args.all_subjects):
            rsm_1, rsm_2 = load_rsms(args, sub, 'test')
            for g in range(best_generation_image_batches[c].shape[1]):
                image_batch = best_generation_image_batches[c][s,g]
                image_batch = np.reshape(image_batch, (1, len(image_batch)))
                scores_test[s,g] = evaluate(image_batch, rsm_1, rsm_2)[0]
            del rsm_1, rsm_2
        best_generation_scores_test[c] = copy(scores_test)


# =============================================================================
# Load the multivariate RNC baseline scores
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'baseline', 'cv-'+format(args.cv), args.time_pair)

if args.cv == 0:
    file_name = 'baseline.npy'
    results = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    baseline_images = results['baseline_images']
    baseline_images_score = results['baseline_images_score']

elif args.cv == 1:
    baseline_images = []
    baseline_images_score_test = []
    baseline_images_score_train = []
    null_distribution_test = []
    for s in args.all_subjects:
        file_name = 'baseline_cv_subject-' + format(s, '02') + '.npy'
        results = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        baseline_images.append(results['baseline_images'])
        baseline_images_score_test.append(
            results['baseline_images_score_test'])
        baseline_images_score_train.append(
            results['baseline_images_score_train'])
        null_distribution_test.append(
            results['null_distribution_test'])
    baseline_images = np.asarray(baseline_images)
    baseline_images_score_test = np.asarray(baseline_images_score_test)
    baseline_images_score_train = np.asarray(baseline_images_score_train)
    null_distribution_test = np.asarray(null_distribution_test)
    # Compute the neural control differences from baseline
    # Alignment minus baseline
    rsa_alignment_minus_baseline = \
        best_generation_scores_test['align'][:,-1] - baseline_images_score_test
    # Disentanglement minus baseline
    rsa_disentanglement_minus_baseline = \
        best_generation_scores_test['disentangle'][:,-1] - \
        baseline_images_score_test


# =============================================================================
# Compute the 95% confidence intervals (only for cv==1)
# =============================================================================
# Compute the confidence intervals of the cross-validated RSA scores for the
# controlling images, across the 10 THINGS EEG2 subjects.

if args.cv == 1:

    # CI arrays of shape: (CI percentiles)
    ci_align = np.zeros((2))
    ci_disentangle = np.zeros(ci_align.shape)
    ci_baseline = np.zeros(ci_align.shape)

    # Empty CI distribution arrays
    align_dist = np.zeros((args.n_iter))
    disentangle_dist = np.zeros((args.n_iter))
    baseline_dist = np.zeros((args.n_iter))

    # Compute the CI distributions
    for i in tqdm(range(args.n_iter), leave=False):
        idx_resample = resample(np.arange(len(args.all_subjects)))
        align_dist[i] = np.mean(
            best_generation_scores_test['align'][idx_resample,-1])
        disentangle_dist[i] = np.mean(
            best_generation_scores_test['disentangle'][idx_resample,-1])
        baseline_dist[i] = np.mean(baseline_images_score_test[idx_resample])

    # Get the 5th and 95th CI distributions percentiles
    ci_align[0] = np.percentile(align_dist, 2.5)
    ci_align[1] = np.percentile(align_dist, 97.5)
    ci_disentangle[0] = np.percentile(disentangle_dist, 2.5)
    ci_disentangle[1] = np.percentile(disentangle_dist, 97.5)
    ci_baseline[0] = np.percentile(baseline_dist, 2.5)
    ci_baseline[1] = np.percentile(baseline_dist, 97.5)


# =============================================================================
# Compute the within-subject significance (only for cv==1)
# =============================================================================
if args.cv == 1:

    # To perform the within-subject significance we will use the RSA null
    # distributions created while computing the multivariate RNC baseline.
    # These null distributions contain 1 million scores, each consisting of the
    # RSA scores obtained when correlating the RSMs of two time points based on
    # a different set of 50 image conditions.

    # Store the difference between alignment and baseline null distribution
    # scores. The first N null distribution scores will be used as the baseline
    # scores, and the second N null distribution scores will be used as the
    # alignment scores (where N is equal to args.n_iter).
    idx_baseline_start = args.n_iter * 0
    idx_baseline_end = idx_baseline_start + args.n_iter
    idx_align_start = args.n_iter * 1
    idx_align_end = idx_align_start + args.n_iter
    rsa_alignment_minus_baseline_null_dist = \
        null_distribution_test[:,idx_align_start:idx_align_end] - \
        null_distribution_test[:,idx_baseline_start:idx_baseline_end]
    # Store the difference between disentanglement and baseline null
    # distribution scores. The first N null distribution scores will be used as
    # the baseline scores, and the third N null distribution scores will be used
    # as the disentanglement scores (where N is equal to args.n_iter).
    idx_disentangle_start = args.n_iter * 2
    idx_disentangle_end = idx_disentangle_start + args.n_iter
    rsa_disentanglement_minus_baseline_null_dist = \
        null_distribution_test[:,idx_disentangle_start:idx_disentangle_end] - \
        null_distribution_test[:,idx_baseline_start:idx_baseline_end]

    # Compute the within-subject p-values
    rsa_alignment_within_subject_pval = np.zeros((len(args.all_subjects)),
        dtype=np.float32)
    rsa_disentanglement_within_subject_pval = np.zeros((len(args.all_subjects)),
        dtype=np.float32)
    # Compute the p-values
    for s, sub in enumerate(args.all_subjects):
        # Alignment
        idx = sum(rsa_alignment_minus_baseline_null_dist[s] > \
            rsa_alignment_minus_baseline[s])
        rsa_alignment_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
        # Disentanglement
        idx = sum(rsa_disentanglement_minus_baseline_null_dist[s] < \
            rsa_disentanglement_minus_baseline[s])
        rsa_disentanglement_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1)

    # Benjamini/Hochberg correct the within-subject alphas over 2 comparisons
    n_control_conditions = 2
    # Empty significance variables
    rsa_alignment_within_subject_sig = np.zeros((len(args.all_subjects)))
    rsa_disentanglement_within_subject_sig = np.zeros((len(args.all_subjects)))
    # Loop across subjects
    for s in range(len(args.all_subjects)):
        # Append the within-subject p-values across the 2 comparisons
        pvals = np.zeros((n_control_conditions))
        pvals[0] = rsa_alignment_within_subject_pval[s]
        pvals[1] = rsa_disentanglement_within_subject_pval[s]
        # Correct for multiple comparisons
        sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
        # Store the significance scores
        rsa_alignment_within_subject_sig[s] = sig[0]
        rsa_disentanglement_within_subject_sig[s] = sig[1]


# =============================================================================
# Compute the between-subject significance (only for cv==1)
# =============================================================================
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.

if args.cv == 1:

    n = len(args.all_subjects) # Total number of subjects
    p = 0.05 # probability of success in each trial

    # Alignment
    k = sum(rsa_alignment_within_subject_sig) # Number of significant subjects
    # We use "k-1" because otherwise we would get the probability of observing
    # k+1 or more significant results by chance
    rsa_alignment_between_subject_pval = 1 - binom.cdf(k-1, n, p)

    # l1l2
    k = sum(rsa_disentanglement_within_subject_sig)
    rsa_disentanglement_between_subject_pval = 1 - binom.cdf(k-1, n, p)


# =============================================================================
# Save the stats
# =============================================================================
if args.cv == 0:
    results = {
        'time_1': time_1,
        'time_2': time_2,
        'best_generation_scores_train': best_generation_scores_train,
        'best_generation_image_batches': best_generation_image_batches,
        'baseline_images': baseline_images,
        'baseline_images_score': baseline_images_score,
        }

elif args.cv == 1:
    results = {
        'time_1': time_1,
        'time_2': time_2,
        'best_generation_scores_train': best_generation_scores_train,
        'best_generation_scores_test': best_generation_scores_test,
        'best_generation_image_batches': best_generation_image_batches,
        'baseline_images': baseline_images,
        'baseline_images_score_test': baseline_images_score_test,
        'baseline_images_score_train': baseline_images_score_train,
        'ci_align': ci_align,
        'ci_disentangle': ci_disentangle,
        'ci_baseline': ci_baseline,
        'rsa_alignment_within_subject_pval': rsa_alignment_within_subject_pval,
        'rsa_disentanglement_within_subject_pval': rsa_disentanglement_within_subject_pval,
        'rsa_alignment_within_subject_sig': rsa_alignment_within_subject_sig,
        'rsa_disentanglement_within_subject_sig': rsa_disentanglement_within_subject_sig,
        'rsa_alignment_between_subject_pval': rsa_alignment_between_subject_pval,
        'rsa_disentanglement_between_subject_pval': rsa_disentanglement_between_subject_pval,
        }

save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc', 'stats',
    'cv-'+format(args.cv), args.time_pair)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)