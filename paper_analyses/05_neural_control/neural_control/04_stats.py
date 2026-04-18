"""Test whether the controlling images found using the in silico neural
responses of one subject generalize to the in silico neural responses of the
other subject. Stats include confidence intervals and significance.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : int
    Subject identifier for the monkey encoding model. Since the used encoding
    models are trained on the TVSD data, valid subject identifiers are "N" and
    "F".
roi_1: str
    First ROI used. Valid values are "V1", "V4", and "IT".
roi_2: str
    Second ROI used. Valid values are "V1", "V4", and "IT". If None, then only
    one ROI (roi_1) is used for neural control.
control_roi_1: str
    Neural control objective for the first ROI.
    If "early-drive_late-drive", then both the early (25-100ms) and late
    (101-200ms) part of the epoch are driven.
    If "early-suppress_late-suppress", then both the early and late part of the
    epoch are suppressed.
    If "early-drive_late-suppress", then the early part of the epoch is driven
    while the late part is suppressed.
    If "early-suppress_late-drive", then the early part of the epoch is
    suppressed while the late part is driven.
control_roi_2: str
    Neural control objective for the second ROI. The valid values are the same
    as for control_roi_1.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi_1', default='V1', type=str)
parser.add_argument('--roi_2', default=None, type=str)
parser.add_argument('--control_roi_1', default='early-drive_late-drive', type=str)
parser.add_argument('--control_roi_2', default=None, type=str)
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


# =============================================================================
# Load the baseline results
# =============================================================================
# ROI 1
data_dir_roi_1 = os.path.join(args.berg_dir, 'neural_control',
    'neural_control', 'quantitative_results', args.encoding_model,
    f'sub-{args.subject}_roi-{args.roi_1}_baseline.npy')
data = np.load(data_dir_roi_1,allow_pickle=True).item()
base_resp_roi_1 = data['baseline_resp']
img_baseline_roi_1 = data['img_baseline']
ci_low_null_distribution_roi_1 = data['ci_low_null_distribution']
ci_high_null_distribution_roi_1 = data['ci_high_null_distribution']

# ROI 2
if args.roi_2 is not None:
    data_dir_roi_2 = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'quantitative_results', args.encoding_model,
        f'sub-{args.subject}_roi-{args.roi_2}_baseline.npy')
    data = np.load(data_dir_roi_2, allow_pickle=True).item()
    base_resp_roi_2 = data['baseline_resp']
    img_baseline_roi_2 = data['img_baseline']
    ci_low_null_distribution_roi_2 = data['ci_low_null_distribution']
    ci_high_null_distribution_roi_2 = data['ci_high_null_distribution']


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
# ROI 1
data_dir_roi_1 = os.path.join(args.berg_dir, 'neural_control',
    'neural_control', 'insilico_responses', args.encoding_model,
    f'insilico_responses_sub-{args.subject}_roi-{args.roi_1}.npy')
data_roi_1 = np.load(data_dir_roi_1, allow_pickle=True).item()
resp_roi_1 = data_roi_1['responses']
metadata = data_roi_1['metadata']
times = metadata['utah_array']['times']
del data_roi_1

# ROI 2
if args.roi_2 is not None:
    data_dir_roi_2 = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'insilico_responses', args.encoding_model,
        f'insilico_responses_sub-{args.subject}_roi-{args.roi_2}.npy')
    resp_roi_2 = np.load(data_dir_roi_2, allow_pickle=True).item()['responses']


# =============================================================================
# Cross validate the controlling images across subjects
# =============================================================================
# Load the controlling images of the other subject
other_subject = 'F' if args.subject == 'N' else 'N'
data_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'quantitative_results', args.encoding_model)
if args.roi_2 is not None:
    file_name = (f'sub-{other_subject}_roi_1-{args.roi_1}_{args.control_roi_1}'
        f'_roi_2-{args.roi_2}_{args.control_roi_2}.npy')
else:
    file_name = (f'sub-{other_subject}_roi-{args.roi_1}_'
                f'{args.control_roi_1}.npy')
img_control = np.load(os.path.join(data_dir, file_name))

# Cross-validate the controlling images on the subject of interest
control_resp_roi_1 = resp_roi_1[img_control]
if args.roi_2 is not None:
    control_resp_roi_2 = resp_roi_2[img_control]


# =============================================================================
# Compute the confidence intervals
# =============================================================================
# ROI 1
dist_control_roi_1 = np.zeros((args.n_iter, len(times)))
dist_base_roi_1 = np.zeros((args.n_iter, len(times)))
for i in tqdm(range(args.n_iter), leave=False):
    idx = resample(np.arange(len(img_control)))
    dist_control_roi_1[i] = np.mean(control_resp_roi_1[idx], 0)
    dist_base_roi_1[i] = np.mean(base_resp_roi_1[idx], 0)
ci_low_control_resp_roi_1 = np.percentile(dist_control_roi_1, 2.5, 0)
ci_high_control_resp_roi_1 = np.percentile(dist_control_roi_1, 97.5, 0)
ci_low_base_resp_roi_1 = np.percentile(dist_base_roi_1, 2.5, 0)
ci_high_base_resp_roi_1 = np.percentile(dist_base_roi_1, 97.5, 0)

# ROI 2
if args.roi_2 is not None:
    dist_control_roi_2 = np.zeros((args.n_iter, len(times)))
    dist_base_roi_2 = np.zeros((args.n_iter, len(times)))
    for i in tqdm(range(args.n_iter), leave=False):
        idx = resample(np.arange(len(img_control)))
        dist_control_roi_2[i] = np.mean(control_resp_roi_2[idx], 0)
        dist_base_roi_2[i] = np.mean(base_resp_roi_2[idx], 0)
    ci_low_control_resp_roi_2 = np.percentile(dist_control_roi_2, 2.5, 0)
    ci_high_control_resp_roi_2 = np.percentile(dist_control_roi_2, 97.5, 0)
    ci_low_base_resp_roi_2 = np.percentile(dist_base_roi_2, 2.5, 0)
    ci_high_base_resp_roi_2 = np.percentile(dist_base_roi_2, 97.5, 0)


# =============================================================================
# Compute the within-subject significance of the CV neural control scores # !!!
# =============================================================================
# Get indices for early (25-100ms) and late (101-200ms) time points
t_min_early = np.where(times == 25)[0][0]
t_max_early = np.where(times == 100)[0][0]
t_min_late = np.where(times == 101)[0][0]
t_max_late = np.where(times == 199)[0][0]

# # Compute the difference between the mean responses for controlling and
# # baseline images
# control_minus_baseline = np.mean(control_resp, 0) - np.mean(baseline_resp, 0)

# # Create the permutation-based null distribution
# control_minus_baseline_null_dist = np.zeros((args.n_iter, len(times)),
#     dtype=np.float32)
# # Loop across iterations
# for i in tqdm(range(args.n_iter)):
#     # Shuffle the univariate responses across samples
#     idx = np.arange(len(insilico_resp))
#     np.random.shuffle(idx)
#     shuffled_resp = insilico_resp[idx]
#     # Compute the differences between control and baseline images for the
#     # shuffled data
#     control_minus_baseline_null_dist[i] = np.mean(
#         shuffled_resp[img_control], 0) - \
#         np.mean(shuffled_resp[img_baseline], 0)

# # Compute the within-subject p-values
# p_val = np.empty((len(times)), dtype=np.float32)
# p_val[:] = np.nan
# if args.control == 'early-drive_late-drive':
#     idx_early = sum(control_minus_baseline_null_dist > \
#         control_minus_baseline, 0)[t_min_early:t_max_early+1]
#     idx_late = sum(control_minus_baseline_null_dist > \
#         control_minus_baseline, 0)[t_min_late:t_max_late+1]
# elif args.control == 'early-suppress_late-suppress':
#     idx_early = sum(control_minus_baseline_null_dist < \
#         control_minus_baseline, 0)[t_min_early:t_max_early+1]
#     idx_late = sum(control_minus_baseline_null_dist < \
#         control_minus_baseline, 0)[t_min_late:t_max_late+1]
# elif args.control == 'early-drive_late-suppress':
#     idx_early = sum(control_minus_baseline_null_dist > \
#         control_minus_baseline, 0)[t_min_early:t_max_early+1]
#     idx_late = sum(control_minus_baseline_null_dist < \
#         control_minus_baseline, 0)[t_min_late:t_max_late+1]
# elif args.control == 'early-suppress_late-drive':
#     idx_early = sum(control_minus_baseline_null_dist < \
#         control_minus_baseline, 0)[t_min_early:t_max_early+1]
#     idx_late = sum(control_minus_baseline_null_dist > \
#         control_minus_baseline, 0)[t_min_late:t_max_late+1]
# p_val[t_min_early:t_max_early+1] = (idx_early + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
# p_val[t_min_late:t_max_late+1] = (idx_late + 1) / (args.n_iter + 1)

# # Benjamini/Hochberg correct the within-subject p-values for multiple
# # comparisons across time points
# p_val_bh = np.empty((len(times)), dtype=np.float32)
# p_val_bh[:] = np.nan
# p_val_bh[t_min_early:t_max_late+1] = multipletests(
#     p_val[t_min_early:t_max_late+1], 0.05, 'fdr_bh')[1]


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
results = {
    'times': times,
    't_min_early': t_min_early,
    't_max_early': t_max_early,
    't_min_late': t_min_late,
    't_max_late': t_max_late,

    'base_resp_roi_1': base_resp_roi_1,
    'img_baseline_roi_1': img_baseline_roi_1,
    'ci_low_null_distribution_roi_1': ci_low_null_distribution_roi_1,
    'ci_high_null_distribution_roi_1': ci_high_null_distribution_roi_1,

    'img_control': img_control,
    'control_resp_roi_1': control_resp_roi_1,

    'ci_low_control_resp_roi_1': ci_low_control_resp_roi_1,
    'ci_high_control_resp_roi_1': ci_high_control_resp_roi_1,
    'ci_low_base_resp_roi_1': ci_low_base_resp_roi_1,
    'ci_high_base_resp_roi_1': ci_high_base_resp_roi_1,

    # 'p_val': p_val, # !!!
    # 'p_val_bh': p_val_bh # !!!
}

if args.roi_2 is not None:
    results.update({
        'base_resp_roi_2': base_resp_roi_2,
        'img_baseline_roi_2': img_baseline_roi_2,
        'ci_low_null_distribution_roi_2': ci_low_null_distribution_roi_2,
        'ci_high_null_distribution_roi_2': ci_high_null_distribution_roi_2,

        'control_resp_roi_2': control_resp_roi_2,

        'ci_low_control_resp_roi_2': ci_low_control_resp_roi_2,
        'ci_high_control_resp_roi_2': ci_high_control_resp_roi_2,
        'ci_low_base_resp_roi_2': ci_low_base_resp_roi_2,
        'ci_high_base_resp_roi_2': ci_high_base_resp_roi_2,

    # 'p_val': p_val, # !!!
    # 'p_val_bh': p_val_bh # !!!
    })

save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

if args.roi_2 is not None:
    file_name = (f'sub-{args.subject}_roi_1-{args.roi_1}_{args.control_roi_1}'
        f'_roi_2-{args.roi_2}_{args.control_roi_2}.npy')
else:
    file_name = f'sub-{args.subject}_roi-{args.roi_1}_{args.control_roi_1}.npy'

np.save(os.path.join(save_dir, file_name), results)