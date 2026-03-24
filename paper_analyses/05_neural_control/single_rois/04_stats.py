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
roi: str
    ROI used. Valid values are "V1", "V4", and "IT".
control: str
    If "early-drive_late-drive", then both the early (25-100ms) and late
    (101-200ms) part of the epoch are driven.
    If "early-suppress_late-suppress", then both the early and late part of the
    epoch are suppressed.
    If "early-drive_late-suppress", then the early part of the epoch is driven
    while the late part is suppressed.
    If "early-suppress_late-drive", then the early part of the epoch is
    suppressed while the late part is driven.
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
from tqdm import tqdm
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--control', default='early-drive_late-drive', type=str)
parser.add_argument('--n_images', default=50, type=int)
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
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'insilico_responses',
args.encoding_model)
file_name = f'insilico_responses_sub-{args.subject}_roi-{args.roi}.npy'

data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
insilico_resp = data['responses']
metadata = data['metadata']
times = metadata['utah_array']['times']


# =============================================================================
# Cross validate the controlling images across subjects
# =============================================================================
# Load the controlling images of the other subject
other_subject = 'F' if args.subject == 'N' else 'N'
data_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois', 
    'quantitative_results', args.encoding_model,
    f'sub-{other_subject}_roi-{args.roi}_{args.control}.npy')
img_control = np.load(data_dir)

# Select the first N controlling images which are not NaN (i.e., that have in
# silico responses above/below the baseline scores, plus a margin)
idx_nan = np.isnan(img_control)
img_control = img_control[~idx_nan][:args.n_images].astype(np.int32)

# Cross-validate the controlling on the subject of interest
control_resp = insilico_resp[img_control]


# =============================================================================
# Get the baseline results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model,
    f'sub-{args.subject}_roi-{args.roi}_baseline.npy')
baseline_results = np.load(data_dir, allow_pickle=True).item()

baseline_resp = baseline_results['baseline_resp']
ci_low_null_distribution = baseline_results['ci_low_null_distribution']
ci_high_null_distribution = baseline_results['ci_high_null_distribution']


# =============================================================================
# Compute the confidence intervals
# =============================================================================
dist_control = np.zeros((args.n_iter, len(times)))
dist_baseline = np.zeros((args.n_iter, len(times)))

for i in tqdm(range(args.n_iter), leave=False):
    idx = resample(np.arange(args.n_images))
    dist_control[i] = np.mean(control_resp[idx], 0)
    dist_baseline[i] = np.mean(baseline_resp[idx], 0)

ci_low_control_resp = np.percentile(dist_control, 2.5, axis=0)
ci_high_control_resp = np.percentile(dist_control, 97.5, axis=0)
ci_low_baseline_resp = np.percentile(dist_baseline, 2.5, axis=0)
ci_high_baseline_resp = np.percentile(dist_baseline, 97.5, axis=0)
















# =============================================================================
# Compute the within-subject significance of the CV neural control scores # !!!
# =============================================================================
# Empty p-value lists
p_val = []
p_val_bh = []
p_val_bonf = []

# Loop across subjects
for s in range(len(args.subjects)):

    # Compute the within-subject p-values
    s_cv = np.delete((0, 1), s)[0]
    if args.control == 'drive':
        idx = np.sum(
            null_distribution[:,s] > np.mean(cv_control_data[s_cv], 0), 0)
    elif args.control == 'suppress':
        idx = np.sum(
            null_distribution[:,s] < np.mean(cv_control_data[s_cv], 0), 0)
    p_val_sub = (idx + 1) / (args.n_iter + 1) # Add 1 to avoid p-values of 0
    p_val.append(p_val_sub)

    # Correct for multiple comparisons
    p_val_bh.append(multipletests(p_val_sub, 0.05, 'fdr_bh')[1])
    p_val_bonf.append(multipletests(p_val_sub, 0.05, 'bonferroni')[1])

# Format to numpy arrays
p_val = np.array(p_val)
p_val_bh = np.array(p_val_bh)
p_val_bonf = np.array(p_val_bonf)


















# =============================================================================
# Save the quantitative neural control results
# =============================================================================
results = { # !!!
    'times': times,
    'control_resp': control_resp,
    'baseline_resp': baseline_resp,
    'ci_low_null_distribution': ci_low_null_distribution,
    'ci_high_null_distribution': ci_high_null_distribution,
    'ci_low_control_resp': ci_low_control_resp,
    'ci_high_control_resp': ci_high_control_resp,
    'ci_low_baseline_resp': ci_low_baseline_resp,
    'ci_high_baseline_resp': ci_high_baseline_resp,
}


save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'sub-{args.subject}_roi-{args.roi}_{args.control}.npy'

np.save(os.path.join(save_dir, file_name), results)