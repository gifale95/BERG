"""Aggregate all variance partitioning results, and compute their ROI-based
statistics with confidence intervals.

Parameters
----------
subjects : list
    List of subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
hemispheres: list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
eeg_train_trials : list
    List indicating which training EEG response trials are used. Possible
    values  are: 'even' (even trials), and 'odd' (odd trials).
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
parser.add_argument('--subjects', default=[1, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--eeg_train_trials', default=['even', 'odd'], type=list)
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
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding',
    'variance_partitioning')
filename = (f'variance_partitioning_sub-01_hemisphere-lh_'
    f'eeg_train_trials-even_time_split-00.npy')
times = np.load(os.path.join(data_dir, filename),
    allow_pickle=True).item()['times']

# Analysis parameters
metadata = []
n_sub = len(args.subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']
result_types = ['total_variance', 'total_variance_vision_dnn',
    'total_variance_llm', 'unique_variance_vision_dnn', 'unique_variance_llm',
    'shared_variance']

# Empty result arrays of shape:
# (N subjects, 2 hemispheres, 163842 fMRI vertices, N EEG time points)
variance_partitioning = {}
for rt in result_types:
    variance_partitioning[rt] = np.zeros((n_sub, n_hemi, n_vertex, n_time),
        dtype=np.float32)


# =============================================================================
# Load the variance partitioning results
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Results parent directory
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding',
    'variance_partitioning')

# Loop across fMRI subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=sub
        ))

    # Loop across fMRI hemispheres
    for h, hemi in enumerate(args.hemispheres):

        # Only select vertices falling within the NSD visual streams
        idx_streams = np.zeros(n_vertex, dtype=int)
        streams = ['early', 'midventral', 'midlateral', 'midparietal',
            'ventral', 'lateral', 'parietal']
        for stream in streams:
            idx_streams[metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
        idx_v = np.where(idx_streams == 1)[0]
        idx_v_nan = np.where(idx_streams != 1)[0]

        # Loop across EEG time points
        for t in range(args.tot_time_splits):

            # Get the time indices for the current time split
            times_per_split = int(np.ceil(len(times) / args.tot_time_splits))
            start_idx = t * times_per_split
            end_idx = min((t + 1) * times_per_split, len(times))

            # Loop across cross-validations
            for et in args.eeg_train_trials:

                # Load and store the result scores
                file_name = (f'variance_partitioning_sub-{sub:02d}_'
                    f'hemisphere-{hemi}_eeg_train_trials-{et}_'
                    f'time_split-{t:02d}.npy')
                results = np.load(os.path.join(data_dir, file_name),
                    allow_pickle=True).item()
                for rt in result_types:
                    variance_partitioning[rt][s,h,idx_v,start_idx:end_idx] += \
                        results['variance_partitioning'][rt]

        # Divide the results by the number of CV splits
        for rt in result_types:
            variance_partitioning[rt][s,h] /= len(args.eeg_train_trials)

        # NCSNR and visual stream vertex selection
        ncsnr = metadata[s]['fmri'][hemi+'_ncsnr']
        idx_nan = ncsnr < args.ncsnr_threshold
        for rt in result_types:
            variance_partitioning[rt][s,h,idx_nan] = np.nan
            variance_partitioning[rt][s,h,idx_v_nan] = np.nan


# =============================================================================
# Get the ROI-wise scores
# =============================================================================
# Empty ROI result array of shape:
# (N fMRI subjects, N EEG time points)
variance_partitioning_roi = {}
for rt in result_types:
    variance_partitioning_roi[rt] = {}
    for roi in rois:
        variance_partitioning_roi[rt][roi] = np.zeros((n_sub, n_time),
            dtype=np.float32)

# Loop across ROIs, subjects and hemispheres
for r, roi in enumerate(rois):
    for s, sub in enumerate(args.subjects):
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
            for rt in result_types:
                if h == 0:
                    score[rt] = variance_partitioning[rt][s,h,idx_roi]
                else:
                    score[rt] = np.append(score[rt],
                        variance_partitioning[rt][s,h,idx_roi], 0)

        # Store the mean correlation across ROI vertices
        for rt in result_types:
            variance_partitioning_roi[rt][roi][s] = np.nanmean(score[rt], 0)
        del score


# =============================================================================
# Bootstrap the confidence intervals of the ROI-wise results
# =============================================================================
ci_variance_partitioning_roi = {}
ci_variance_partitioning_roi_peak_lat = {}
for rt in result_types:
    ci_variance_partitioning_roi[rt] = {}
    ci_variance_partitioning_roi_peak_lat[rt] = {}
    for roi in rois:
        ci_variance_partitioning_roi[rt][roi] = np.zeros(((2, n_time)),
            dtype=np.float32)
        ci_variance_partitioning_roi_peak_lat[rt][roi] = np.zeros((2),
            dtype=np.float32)

for rt in tqdm(result_types):
    for key, val in variance_partitioning_roi[rt].items():

        score_dist = np.zeros((args.n_iter, n_time))
        peak_lat_dist = np.zeros((args.n_iter))

        for i in range(args.n_iter):
            idx = resample(np.arange(len(args.subjects)))
            score_dist[i] = np.mean(val[idx], 0)
            peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

        ci_variance_partitioning_roi[rt][key][0] = np.percentile(
            score_dist, 2.5, 0)
        ci_variance_partitioning_roi[rt][key][1] = np.percentile(
            score_dist, 97.5, 0)
        ci_variance_partitioning_roi_peak_lat[rt][key][0] = np.percentile(
            peak_lat_dist, 2.5)
        ci_variance_partitioning_roi_peak_lat[rt][key][1] = np.percentile(
            peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'times': times,
    'variance_partitioning': variance_partitioning,
    'variance_partitioning_roi': variance_partitioning_roi,
    'ci_variance_partitioning_roi': ci_variance_partitioning_roi,
    'ci_variance_partitioning_roi_peak_lat': ci_variance_partitioning_roi_peak_lat,
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)