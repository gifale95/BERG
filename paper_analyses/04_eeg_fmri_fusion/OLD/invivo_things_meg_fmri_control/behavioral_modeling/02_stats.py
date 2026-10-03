"""Aggregate the behavioral modeling RSA scores across fMRI subjects, and
compute the confidence intervals.

Parameters
----------
fmri_subjects : list
    List of THINGS fMRI1 subject identifiers. Valid subject identifiers are
    integers from 1 to 3.
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
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# MEG metadata and analysis parameters
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the MEG encoding model metadata
metadata_meg = berg.get_model_metadata(
    'meg-things_meg_1-vit_b_32',
    subject=1
)

# Load the MEG time points
tmax = 0.595
times = metadata_meg['meg']['times']
time_idx = np.zeros(len(times), dtype=int)
time_idx[times <= tmax] = 1
time_idx = np.where(time_idx == 1)[0]
times = times[times <= tmax]


# =============================================================================
# Get the MEG time points
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the MEG encoding model metadata
metadata_meg = berg.get_model_metadata(
    'meg-things_meg_1-vit_b_32',
    subject=1
)

# Load the MEG time points
tmax = 0.595
times = metadata_meg['meg']['times']
time_idx = np.zeros(len(times), dtype=int)
time_idx[times <= tmax] = 1
time_idx = np.where(time_idx == 1)[0]
times = times[times <= tmax]


# =============================================================================
# Load the RSA results
# =============================================================================
rsa = {}

# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'invivo_things_meg_fmri_control', 'behavioral_modeling', 'rsa',
        f'rsa_fmri_sub-{fsub:02d}.npy')
    results = np.load(results_dir, allow_pickle=True).item()

    # Loop across ROIs
    for r, roi in enumerate(tqdm(results.keys())):

        # Store the RSA results of all subjects
        if fs == 0:
            rsa[roi] = []
        rsa[roi].append(results[roi])

    del results

# Format the results to numpy arrays
for roi in rsa.keys():
    rsa[roi] = np.array(rsa[roi])


# =============================================================================
# Bootstrap the confidence intervals (ROI-wise in silico fMRI vs t-fMRI
# correlation scores)
# =============================================================================
ci_rsa = {}
ci_rsa_peak_lat = {}

for key, val in tqdm(rsa.items()):

    ci_rsa[key] = np.zeros((2, len(times)))
    ci_rsa_peak_lat[key] = np.zeros((2))
    corr_dist = np.zeros((args.n_iter, len(times)))
    peak_lat_dist = np.zeros((args.n_iter))

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.fmri_subjects)))
        corr_dist[i] = np.mean(val[idx], 0)
        peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

    ci_rsa[key][0] = np.percentile(corr_dist, 2.5, axis=0)
    ci_rsa[key][1] = np.percentile(corr_dist, 97.5, axis=0)
    ci_rsa_peak_lat[key][0] = np.percentile(peak_lat_dist, 2.5)
    ci_rsa_peak_lat[key][1] = np.percentile(peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'rsa': rsa,
    'ci_rsa': ci_rsa,
    'ci_rsa_peak_lat': ci_rsa_peak_lat
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'behavioral_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)