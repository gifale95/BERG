"""Aggregate all RSA results, and compute their ROI-based statistics with
confidence intervals.

Parameters
----------
fmri_subjects : list
    List of subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
hemispheres: list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
tot_time_splits : int
    The total number of splits in which the EEG time points are divided.
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
from tqdm import tqdm
from berg import BERG
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--tot_time_splits', default=10, type=int)
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
# Empty result arrays
# =============================================================================
# Load the time points
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'rsa')
filename = 'rsa_fmri_sub-01_hemisphere-lh_time_split-00.npy'
data = np.load(os.path.join(data_dir, filename), allow_pickle=True).item()
times = data['times']
model_layers = data['model_layers']

# Analysis parameters
metadata = []
n_sub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']

# Empty result arrays of shape:
# (N subjects, 2 hemispheres, 163842 fMRI vertices, N EEG time points)
rsa = {}
for layer in model_layers:
    rsa[layer] = np.zeros((n_sub, n_hemi, n_vertex, n_time), dtype=np.float32)
    rsa[layer][:] = np.nan


# =============================================================================
# Load the RSA results
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Results parent directory
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'rsa')

# Loop across fMRI subjects
for s, sub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=sub
        ))

    # Loop across fMRI hemispheres
    for h, hemi in enumerate(args.hemispheres):

        # Loop across EEG time points
        for t in range(args.tot_time_splits):

            # Get the time indices for the current time split
            times_per_split = int(np.ceil(len(times) / args.tot_time_splits))
            start_idx = t * times_per_split
            end_idx = min((t + 1) * times_per_split, len(times))

            # Load and store the result scores
            file_name = (f'rsa_fmri_sub-{sub:02d}_hemisphere-{hemi}_'
                f'time_split-{t:02d}.npy')
            results = np.load(os.path.join(data_dir, file_name),
                allow_pickle=True).item()
            for layer in model_layers:
                rsa[layer][s,h,:,start_idx:end_idx] = results['rsa'][layer]

        # NCSNR and visual stream vertex selection
        ncsnr = metadata[s]['fmri'][hemi+'_ncsnr']
        idx_nan = ncsnr < args.ncsnr_threshold
        for layer in model_layers:
            rsa[layer][s,h,idx_nan] = np.nan


# =============================================================================
# Get the ROI-wise scores
# =============================================================================
# Empty ROI result array of shape:
# (N fMRI subjects, N EEG time points)
rsa_roi = {}
for roi in rois:
    rsa_roi[roi] = {}
    for layer in model_layers:
        rsa_roi[roi][layer] = np.zeros((n_sub, n_time), dtype=np.float32)

# Loop across ROIs, subjects and hemispheres
for r, roi in enumerate(rois):
    for s, sub in enumerate(args.fmri_subjects):
        score = {}
        for h, hemi in enumerate(args.hemispheres):

            # Get the indices of the ROI vertices
            if roi in ['V1', 'V2', 'V3']:
                idx_roi = np.append(
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
                idx_roi.sort()
            elif roi in ['FFA', 'VWFA', 'FBA']:
                idx_roi = np.append(
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
                idx_roi.sort()
            elif roi in ['intermediate']:
                idx_roi = np.append(
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
                idx_roi = np.append(idx_roi,
                    metadata[s]['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
                idx_roi.sort()
            else:
                idx_roi = metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][roi]

            # Get the correlation scores of the above threshold vertices
            # from the chosen ROI
            for layer in model_layers:
                if h == 0:
                    score[layer] = rsa[layer][s,h,idx_roi]
                else:
                    score[layer] = np.append(score[layer],
                        rsa[layer][s,h,idx_roi], 0)

        # Store the mean correlation across ROI vertices
        for layer in model_layers:
            rsa_roi[roi][layer][s] = np.nanmean(score[layer], 0)
        del score


# =============================================================================
# Bootstrap the confidence intervals of the ROI-wise results
# =============================================================================
ci_rsa_roi = {}
ci_rsa_roi_peak_lat = {}

for roi in rois:
    ci_rsa_roi[roi] = {}
    ci_rsa_roi_peak_lat[roi] = {}
    for layer in model_layers:
        ci_rsa_roi[roi][layer] = np.zeros(((2, n_time)), dtype=np.float32)
        ci_rsa_roi_peak_lat[roi][layer] = np.zeros((2), dtype=np.float32)

for roi in tqdm(rois):
    for key, val in rsa_roi[roi].items():

        score_dist = np.zeros((args.n_iter, n_time))
        peak_lat_dist = np.zeros((args.n_iter))

        for i in range(args.n_iter):
            idx = resample(np.arange(len(args.fmri_subjects)))
            score_dist[i] = np.mean(val[idx], 0)
            peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

        ci_rsa_roi[roi][key][0] = np.percentile(
            score_dist, 2.5, 0)
        ci_rsa_roi[roi][key][1] = np.percentile(
            score_dist, 97.5, 0)
        ci_rsa_roi_peak_lat[roi][key][0] = np.percentile(
            peak_lat_dist, 2.5)
        ci_rsa_roi_peak_lat[roi][key][1] = np.percentile(
            peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'model_layers': model_layers,
    'times': times,
    'rsa': rsa,
    'rsa_roi': rsa_roi,
    'ci_rsa_roi': ci_rsa_roi,
    'ci_rsa_roi_peak_lat': ci_rsa_roi_peak_lat,
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)