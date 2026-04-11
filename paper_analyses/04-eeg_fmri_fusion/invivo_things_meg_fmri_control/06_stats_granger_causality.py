"""Aggregate the t-fMRI Granger causality scores for each ROI across fMRI
subjects, and compute the confidence intervals.

Parameters
----------
fmri_subject : list
    Linst of THINGS fMRI1 subject identifiers. Valid subject identifiers are
    integers from 1 to 3.
time_window_s : int
    Time window in seconds for computing Granger Causality.
offset_s : int
    Offset in seconds for the time window.
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
from berg import BERG
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3], type=list)
parser.add_argument('--time_window_s', default=0.1, type=float)
parser.add_argument('--offset_s', default=0.02, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result arrays
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the MEG time points
metadata_meg = berg.get_model_metadata(
    'meg-things_meg_1-vit_b_32',
    subject=1
)
tmax = 0.595
times = metadata_meg['meg']['times']
time_idx = np.zeros(len(times), dtype=int)
time_idx[times <= tmax] = 1
time_idx = np.where(time_idx == 1)[0]
times = times[times <= tmax]

# Get the starting time index for the Granger Causality analysis
t_min = times[0] + args.time_window_s + args.offset_s
idx_t_start = np.where(times == t_min)[0][0]
times = times[idx_t_start:]

# Analysis parameters
n_fsub = len(args.fmri_subjects)
n_time = len(times)

# Empty Granger Causality dictionary
gc_scores = {}


# =============================================================================
# Get the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata_fmri = berg.get_model_metadata(
        'fmri-things_fmri_1-vit_b_32',
        subject=fsub
        )

    # Load the correlation results
    data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'invivo_things_meg_fmri_control', 'granger_causality',
        f'gc_fmri_sub-{fsub:02d}.npy')
    gc_results = np.load(data_dir, allow_pickle=True).item()

    # Loop across ROIs
    for key, val in gc_results.items():

        # Empty Granger causality array of shape:
        # (N fMRI subjects, N MEG time points)
        if fs == 0:
            gc_scores[key] = np.zeros((n_fsub, n_time), dtype=np.float32)

        # Store the Granger Causality scores
        gc_scores[key][fs] = np.mean(val, 0)
        del val
    del gc_results


# =============================================================================
# Bootstrap the confidence intervals
# =============================================================================
ci_gc_scores = {}

for key, val in tqdm(gc_scores.items()):

    ci_gc_scores[key] = np.zeros((2, n_time))
    gc_dist = np.zeros((args.n_iter, n_time))

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.fmri_subjects)))
        gc_dist[i] = np.mean(val[idx], 0)

    ci_gc_scores[key][0] = np.percentile(gc_dist, 2.5, axis=0)
    ci_gc_scores[key][1] = np.percentile(gc_dist, 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'gc_scores': gc_scores,
    'ci_gc_scores': ci_gc_scores
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'stats_granger_causality')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)