"""Aggregate the t-fMRI encoding accuracies for each ROI across fMRI subjects,
and compute the confidence intervals.

Parameters
----------
fmri_subject : list
    Linst of THINGS fMRI1 subject identifiers. Valid subject identifiers are
    integers from 1 to 3.
noise_ceiling_threshold : float
    The threshold on the noise ceiling for voxel selection.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
from importlib import metadata
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3], type=list)
parser.add_argument('--noise_ceiling_threshold', default=20, type=float)
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

# Analysis parameters
n_fsub = len(args.fmri_subjects)
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'OFA', 'EBA', 'PPA', 'RSC', 'TOS',
    'LOC', 'IT']

# Empty correlation dictionary
corr_tfmri_fmri = {}


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
        'invivo_things_meg_fmri_control', 'encoding_fusion_accuracy',
        f'corr_fmri_sub-{fsub:02d}.npy')
    correlation = np.load(data_dir, allow_pickle=True).item()

    # Loop across ROIs
    for roi in rois:

        # Empty correlation array of shape:
        # (N fMRI subjects, 140 MEG time points)
        if fs == 0:
            corr_tfmri_fmri[roi] = np.zeros((n_fsub, n_time), dtype=np.float32)

        # Noise ceiling voxel selection
        if roi in ['FFA', 'FFA', 'OFA', 'EBA', 'PPA', 'RSC', 'TOS', 'LOC']:
            noise_ceiling_lh = metadata_fmri['encoding_model']\
                ['noise_ceiling_testset'][metadata_fmri['roi'][f'l{roi}']]
            noise_ceiling_rh = metadata_fmri['encoding_model']\
                ['noise_ceiling_testset'][metadata_fmri['roi'][f'r{roi}']]
            idx_nc_lh = noise_ceiling_lh >= args.noise_ceiling_threshold
            idx_nc_rh = noise_ceiling_rh >= args.noise_ceiling_threshold
            corr_roi = np.append(correlation[f'l{roi}'][idx_nc_lh],
                correlation[f'r{roi}'][idx_nc_rh], 0)
        else:
            noise_ceiling = metadata_fmri['encoding_model']\
                ['noise_ceiling_testset'][metadata_fmri['roi'][roi]]
            idx_nc = noise_ceiling >= args.noise_ceiling_threshold
            corr_roi = correlation[roi][idx_nc]

        # Store the correlation scores
        corr_tfmri_fmri[roi][fs] = np.mean(corr_roi, 0)
        del corr_roi
    del correlation


# =============================================================================
# Bootstrap the confidence intervals
# =============================================================================
ci_corr_tfmri_fmri = {}
ci_corr_tfmri_fmri_peak_lat = {}

for key, val in tqdm(corr_tfmri_fmri.items()):

    ci_corr_tfmri_fmri[key] = np.zeros((2, n_time))
    ci_corr_tfmri_fmri_peak_lat[key] = np.zeros((2))
    corr_dist = np.zeros((args.n_iter, n_time))
    peak_lat_dist = np.zeros((args.n_iter))

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.fmri_subjects)))
        corr_dist[i] = np.mean(val[idx], 0)
        peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

    ci_corr_tfmri_fmri[key][0] = np.percentile(corr_dist, 2.5, axis=0)
    ci_corr_tfmri_fmri[key][1] = np.percentile(corr_dist, 97.5, axis=0)
    ci_corr_tfmri_fmri_peak_lat[key][0] = np.percentile(peak_lat_dist, 2.5)
    ci_corr_tfmri_fmri_peak_lat[key][1] = np.percentile(peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'corr_tfmri_fmri': corr_tfmri_fmri,
    'ci_corr_tfmri_fmri': ci_corr_tfmri_fmri,
    'ci_corr_tfmri_fmri_peak_lat': ci_corr_tfmri_fmri_peak_lat
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)