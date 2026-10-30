"""Test whether the controlling images found using the t-fMRI responses of the
train subjects generalize to the in t-fMRI responses of the left-out subject.
Stats include confidence intervals and significance.

Parameters
----------
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
roi: str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest.
imageset : str
    The image set to use for the analysis. Possible values are:
    'imagenet_train' (ILSVRC-2012 training split),
    'imagenet_val' (ILSVRC-2012 validation split),
    'imagenet_all' (ILSVRC-2012 training + validation split),
    'coco' (MS COCO 2017 test split).
n_images: int
    Number of retained controlling or baseline images.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
import h5py
from berg import BERG
from scipy.stats import zscore
from scipy.stats import binom

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet_all', type=str)
parser.add_argument('--n_images', default=25, type=int)
parser.add_argument('--n_iter', type=int, default=100000)
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
# Break down the time windows
# =============================================================================
time_window_1_start, time_window_1_end = map(
    float, args.time_window_pair.split('__')[0].split('-'))
time_window_2_start, time_window_2_end = map(
    float, args.time_window_pair.split('__')[1].split('-'))

# Get the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = np.round(metadata_eeg['eeg']['times'], 3)

# Get the time window indices
t_min_1 = np.where(times == time_window_1_start)[0][0]
t_max_1 = np.where(times == time_window_1_end)[0][0]
t_min_2 = np.where(times == time_window_2_start)[0][0]
t_max_2 = np.where(times == time_window_2_end)[0][0]
t_min_all = np.where(times == 0.06)[0][0]
t_max_all = np.where(times == 0.4)[0][0]


# =============================================================================
# Get the total dataset subjects
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# Load the t-fMRI responses, and average them across the time windows
# =============================================================================
# Load the t-fMRI responses of all subjects
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'tfmri_responses')
tfmri = []
if args.imageset in ['imagenet_train', 'imagenet_val', 'coco']:
    for sub in all_subjects:
        file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}_imageset-{args.imageset}.h5'
        tfmri.append(h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'])
elif args.imageset == 'imagenet_all':
    splits = ['imagenet_val', 'imagenet_train']
    for sub in all_subjects:
        tfmri_sub = []
        for split in splits:
            file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}_imageset-{split}.h5'
            tfmri_sub.append(h5py.File(os.path.join(
                data_dir, file_name), 'r')['tfmri'])
        tfmri.append(np.concatenate(tfmri_sub))
        del tfmri_sub
tfmri = np.array(tfmri)
n_images = tfmri.shape[1]

# Average the t-fMRI responses within the time windows of interest
tfmri_1 = np.mean(tfmri[:,:,t_min_1:t_max_1+1], 2)
tfmri_2 = np.mean(tfmri[:,:,t_min_2:t_max_2+1], 2)
tfmri_all = np.mean(tfmri[:,:,t_min_all:t_max_all+1], 2)
del tfmri

# Z-score the t-fMRI responses of each subject
tfmri_1 = zscore(tfmri_1, 1)
tfmri_2 = zscore(tfmri_2, 1)
tfmri_all = zscore(tfmri_all, 1)


# =============================================================================
# Load the univariate RNC controlling images
# =============================================================================
control_types = ['nc_high_1', 'nc_high_2', 'nc_high_all', 'rnc_high_1_low_2',
    'rnc_low_1_high_2', 'rnc_high_1_high_2', 'rnc_low_1_low_2']
controlling_images = {}

data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'quantitative_results', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')

if args.cv == 0:
    file_name = f'image_ranking_roi-{args.roi}.npy'
    data_dict = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    times = data_dict['times']
    for ct in control_types:
        controlling_images[ct] = data_dict['controlling_images'][ct]\
            [:args.n_images]

elif args.cv == 1:
    for ct in control_types:
        controlling_images[ct] = []
    for s in all_subjects:
        file_name = f'image_ranking_cv_subject-{s:02d}_roi-{args.roi}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        times = data_dict['times']
        for ct in control_types:
            controlling_images[ct].append(
                data_dict['controlling_images'][ct][:args.n_images])
    for ct in control_types:
        controlling_images[ct] = np.asarray(controlling_images[ct])


# =============================================================================
# Load the univariate RNC baseline images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'baseline', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')
baseline_images = {}

if args.cv == 0:
    file_name = f'baseline_roi-{args.roi}.npy'
    data_dict = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    baseline_images['time_window_1'] = data_dict['baseline_img']['time_window_1']
    baseline_images['time_window_2'] = data_dict['baseline_img']['time_window_2']
    baseline_images['time_window_all'] = data_dict['baseline_img']['time_window_all']

elif args.cv == 1:
    baseline_images['time_window_1'] = np.zeros((len(all_subjects),
        args.n_images), dtype=np.int32)
    baseline_images['time_window_2'] = np.zeros((len(all_subjects),
        args.n_images), dtype=np.int32)
    baseline_images['time_window_all'] = np.zeros((len(all_subjects),
        args.n_images), dtype=np.int32)
    for s, sub in enumerate(all_subjects):
        file_name = f'baseline_cv_subject-{sub}_roi-{args.roi}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        baseline_images['time_window_1'][s] = \
            data_dict['baseline_img_test']['time_window_1']
        baseline_images['time_window_2'][s] = \
            data_dict['baseline_img_test']['time_window_2']
        baseline_images['time_window_all'][s] = \
            data_dict['baseline_img_test']['time_window_all']


# =============================================================================
# Validate the controlling images across subjects (only for cv==1)
# =============================================================================
# Get the test subject univariate t-fMRI responses for the controlling
# and baseline images.

if args.cv == 1:

    # Univariate t-fMRI response arrays of shape:
    # (8 Subjects × Target images)
    cv_resp_1 = {}
    cv_resp_2 = {}
    cv_resp_all = {}
    for ct in control_types:
        cv_resp_1[ct] = np.zeros((len(all_subjects), args.n_images),
            dtype=np.float32)
        cv_resp_2[ct] = np.zeros((len(all_subjects), args.n_images),
            dtype=np.float32)
        cv_resp_all[ct] = np.zeros((len(all_subjects), args.n_images),
            dtype=np.float32)
    base_resp_1 = np.zeros((len(all_subjects), args.n_images), dtype=np.float32)
    base_resp_2 = np.zeros((len(all_subjects), args.n_images), dtype=np.float32)
    base_resp_all = np.zeros((len(all_subjects), args.n_images), dtype=np.float32)

    # Get the t-fMRI responses for the controlling and baseline images
    for s in range(len(all_subjects)):
        for ct in control_types:
            cv_resp_1[ct][s] = tfmri_1[s,controlling_images[ct][s]]
            cv_resp_2[ct][s] = tfmri_2[s,controlling_images[ct][s]]
            cv_resp_all[ct][s] = tfmri_all[s,controlling_images[ct][s]]
        base_resp_1[s] = tfmri_1[s,baseline_images['time_window_1'][s]]
        base_resp_2[s] = tfmri_2[s,baseline_images['time_window_2'][s]]
        base_resp_all[s] = tfmri_all[s,baseline_images['time_window_all'][s]]


# =============================================================================
# Neural control statistical significance (only for cv==1)
# =============================================================================
# Use permutation tests to compute the within subject significance between the
# univariate t-fMRI responses of driving and baseline images, and then compute
# the population-level significance based on population prevalence.

if args.cv == 1:

    # Define the permutation indices
    idx_ctrl = np.zeros((args.n_iter, args.n_images), dtype=np.int32)
    idx_base = np.zeros((args.n_iter, args.n_images), dtype=np.int32)
    for i in range(args.n_iter):
        idx = rng.choice(n_images, size=args.n_images*2, replace=False)
        idx_ctrl[i] = idx[:args.n_images]
        idx_base[i] = idx[args.n_images:]
        del idx

    # Get the actual difference between the univariate t-fMRI responses of
    # the driving and baseline images
    diff = np.mean(cv_resp_all['nc_high_all'], 1) - np.mean(base_resp_all, 1)

    # Create the difference null distribution using permuted t-fMRI
    # responses
    diff_null_dist = np.zeros((len(all_subjects), args.n_iter),
        dtype=np.float32)
    for i in range(args.n_iter):
        diff_null_dist[:,i] = np.mean(tfmri_all[:,idx_ctrl[i]], 1) - \
            np.mean(tfmri_all[:,idx_base[i]], 1)

    # Compute the within-subject significance
    within_subj_pval = np.zeros((len(all_subjects)))
    for s in range(len(all_subjects)):
        within_subj_pval[s] = \
            (np.sum(diff_null_dist[s] >= diff[s]) + 1) / \
            (args.n_iter + 1)
    within_subj_sig = within_subj_pval < 0.01

    # Compute the population-level significance based on population
    # prevalence: this is the probability of observing k or more
    # significant results by chance, based on the CDF of the binomial
    # distribution of within-subject significances.
    n = len(all_subjects) # Total number of subjects
    p = 0.05 # probability of success in each trial
    k = sum(within_subj_sig) # Number of significant subjects
    # We use "k-1" because otherwise we would get the probability of observing
    # k+1 or more significant results by chance
    nc_between_subj_pval = 1 - binom.cdf(k-1, n, p)


# =============================================================================
# RNC statistical significance (only for cv==1)
# =============================================================================
# For each time window, use permutation tests to compute the within-subject
# significance of the difference between the univariate t-fMRI responses of
# the two controlling conditions (high_1_low_2 vs low_1_high_2), and the
# baseline.

if args.cv == 1:

    # Define the permutation indices
    idx_h1l2 = np.zeros((args.n_iter, args.n_images), dtype=np.int32)
    idx_l1h2 = np.zeros((args.n_iter, args.n_images), dtype=np.int32)
    idx_base = np.zeros((args.n_iter, args.n_images), dtype=np.int32)
    for i in range(args.n_iter):
        idx = rng.choice(n_images, size=args.n_images*2, replace=False)
        idx_h1l2[i] = idx[:args.n_images]
        idx_l1h2[i] = idx[args.n_images:]
        idx_base[i] = rng.choice(n_images, size=args.n_images, replace=False)
        del idx

    # Loop across time windows
    for tw in ['time_window_1', 'time_window_2']:

        # Get the actual difference between the univariate t-fMRI responses of
        # the controlling and baseline images
        if tw == 'time_window_1':
            diff_h1l2 = np.mean(cv_resp_1['rnc_high_1_low_2'], 1) - \
                np.mean(base_resp_1, 1)
            diff_l1h2 = np.mean(base_resp_1, 1) - \
                np.mean(cv_resp_1['rnc_low_1_high_2'], 1)
        elif tw == 'time_window_2':
            diff_h1l2 = np.mean(base_resp_2, 1) - \
                np.mean(cv_resp_2['rnc_high_1_low_2'], 1)
            diff_l1h2 = np.mean(cv_resp_2['rnc_low_1_high_2'], 1) - \
                np.mean(base_resp_2, 1)

        # Create the difference null distribution using permuted t-fMRI
        # responses
        diff_h1l2_null_dist = np.zeros((len(all_subjects), args.n_iter),
            dtype=np.float32)
        diff_l1h2_null_dist = np.zeros((len(all_subjects), args.n_iter),
            dtype=np.float32)
        for i in range(args.n_iter):
            if tw == 'time_window_1':
                diff_h1l2_null_dist[:,i] = \
                    np.mean(tfmri_1[:,idx_h1l2[i]], 1) - \
                    np.mean(tfmri_1[:,idx_base[i]], 1)
                diff_l1h2_null_dist[:,i] =  \
                    np.mean(tfmri_1[:,idx_base[i]], 1) - \
                    np.mean(tfmri_1[:,idx_l1h2[i]], 1)
            elif tw == 'time_window_2':
                diff_h1l2_null_dist[:,i] =  \
                    np.mean(tfmri_2[:,idx_base[i]], 1) - \
                    np.mean(tfmri_2[:,idx_h1l2[i]], 1)
                diff_l1h2_null_dist[:,i] = \
                    np.mean(tfmri_2[:,idx_l1h2[i]], 1) - \
                    np.mean(tfmri_2[:,idx_base[i]], 1)

        # Compute the within-subject significance
        within_subj_pval_h1l2 = np.zeros((len(all_subjects)))
        within_subj_pval_l1h2 = np.zeros((len(all_subjects)))
        for s in range(len(all_subjects)):
            within_subj_pval_h1l2[s] = \
                (np.sum(diff_h1l2_null_dist[s] >= diff_h1l2[s]) + 1) / \
                (args.n_iter + 1)
            within_subj_pval_l1h2[s] = \
                (np.sum(diff_l1h2_null_dist[s] >= diff_l1h2[s]) + 1) / \
                (args.n_iter + 1)
        within_subj_pval_h1l2 = within_subj_pval_h1l2 < 0.01
        within_subj_pval_l1h2 = within_subj_pval_l1h2 < 0.01

        # Compute the population-level significance based on population
        # prevalence: this is the probability of observing k or more
        # significant results by chance, based on the CDF of the binomial
        # distribution of within-subject significances.
        n = len(all_subjects) # Total number of subjects
        p = 0.05 # probability of success in each trial
        k_h1l2 = sum(within_subj_pval_h1l2) # Number of significant subjects
        k_l1h2 = sum(within_subj_pval_l1h2) # Number of significant subjects
        # We use "k-1" because otherwise we would get the probability of observing
        # k+1 or more significant results by chance
        if tw == 'time_window_1':
            rnc_between_subj_pval_h1l2_tw_1 = 1 - binom.cdf(k_h1l2-1, n, p)
            rnc_between_subj_pval_l1h2_tw_1 = 1 - binom.cdf(k_l1h2-1, n, p)
        elif tw == 'time_window_2':
            rnc_between_subj_pval_h1l2_tw_2 = 1 - binom.cdf(k_h1l2-1, n, p)
            rnc_between_subj_pval_l1h2_tw_2 = 1 - binom.cdf(k_l1h2-1, n, p)


# =============================================================================
# Save the results
# =============================================================================
if args.cv == 0:
    stats = {
        'times': times,
        'time_window_1_start': time_window_1_start,
        'time_window_1_end': time_window_1_end,
        'time_window_2_start': time_window_2_start,
        'time_window_2_end': time_window_2_end,
        'tfmri_1': tfmri_1,
        'tfmri_2': tfmri_2,
        'control_types': control_types,
        'controlling_images': controlling_images,
        'baseline_images': baseline_images,
        }

elif args.cv == 1:
    stats = {
        'times': times,
        'time_window_1_start': time_window_1_start,
        'time_window_1_end': time_window_1_end,
        'time_window_2_start': time_window_2_start,
        'time_window_2_end': time_window_2_end,
        'tfmri_1': tfmri_1,
        'tfmri_2': tfmri_2,
        'control_types': control_types,
        'controlling_images': controlling_images,
        'baseline_images': baseline_images,
        'cv_resp_1': cv_resp_1,
        'cv_resp_2': cv_resp_2,
        'cv_resp_all': cv_resp_all,
        'base_resp_1': base_resp_1,
        'base_resp_2': base_resp_2,
        'base_resp_all': base_resp_all,
        'nc_between_subj_pval': nc_between_subj_pval,
        'rnc_between_subj_pval_h1l2_tw_1': rnc_between_subj_pval_h1l2_tw_1,
        'rnc_between_subj_pval_l1h2_tw_1': rnc_between_subj_pval_l1h2_tw_1,
        'rnc_between_subj_pval_h1l2_tw_2': rnc_between_subj_pval_h1l2_tw_2,
        'rnc_between_subj_pval_l1h2_tw_2': rnc_between_subj_pval_l1h2_tw_2
        }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), stats)