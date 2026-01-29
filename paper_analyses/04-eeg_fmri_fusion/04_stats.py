"""Aggregate the t-fMRI encoding accuracies across fMRI subjects and
hemispheres, and compute the ROI-wise correlations between the in silico fMRI
responses and the t-fMRI responses.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
source_dataset : str
    If 'things_eeg_2', the source dataset is THINGS EEG2. If 'things_meg_1',
    the source dataset  is THINGS MEG1. (The source dataset is the dataset that
    is mapped onto fMRI responses.)
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
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
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--source_dataset', default='things_eeg_2', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
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

# Load the M/EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']

# Analysis parameters
metadata = []
n_fsub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']

# Empty correlation array of shape:
# (N fMRI subjects, 2 hemispheres, 163842 fMRI vertices, 140 EEG time points)
corr_tfmri_fmri = np.zeros((n_fsub, n_hemi, n_vertex, n_time),
    dtype=np.float32)
corr_tfmri_fmri[:] = np.nan


# =============================================================================
# Load the correlation results
# =============================================================================
# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=fsub
        ))

    # Loop across fMRI hemispheres
    for h, hemi in enumerate(args.hemispheres):

        # Only select vertices falling within the NSD visual streams
        idx_v = np.zeros(n_vertex, dtype=int)
        streams = ['early', 'midventral', 'midlateral', 'midparietal',
            'ventral', 'lateral', 'parietal']
        for stream in streams:
            idx_v[metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
        idx_v = np.where(idx_v == 1)[0]

        # Load and store the correlation scores
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'encoding_fusion_accuracy', f'source_dataset-{args.source_dataset}')
        file_name = f'corr_fmri_sub-{fsub:02d}_hemi-{hemi}.npy'
        corr_tfmri_fmri[fs,h,idx_v] = np.load(os.path.join(data_dir, file_name))

        # NCSNR and encoding accuracy vertex selection
        ncsnr = metadata[fs]['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = metadata[fs]['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
        corr_tfmri_fmri[fs,h,idx_nan] = np.nan


# =============================================================================
# Get the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# Empty result dictionary
corr_tfmri_fmri_roi = {}

# Loop across ROIs
for r, roi in enumerate(rois):

    # Empty ROI correlation array of shape:
    # (N fMRI subjects, 140 EEG time points)
    corr_tfmri_fmri_roi[roi] = np.zeros((len(args.fmri_subjects), n_time),
        dtype=np.float32)

    # Loop across subjects and hemispheres
    for fs, fsub in enumerate(args.fmri_subjects):
        for h, hemi in enumerate(args.hemispheres):

            # Get the indices of the ROI vertices
            if roi in ['V1', 'V2', 'V3']:
                idx_roi = np.append(
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
                idx_roi.sort()
            elif roi in ['FFA', 'VWFA', 'FBA']:
                idx_roi = np.append(
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
                idx_roi.sort()
            elif roi in ['intermediate']:
                idx_roi = np.append(
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
                idx_roi = np.append(idx_roi,
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
                idx_roi.sort()
            else:
                idx_roi = metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][roi]

            # Get the correlation scores of the above threshold vertices
            # from the chosen ROI
            if h == 0:
                corr = corr_tfmri_fmri[fs,h,idx_roi]
            else:
                corr = np.append(corr, corr_tfmri_fmri[fs,h,idx_roi], 0)
        
        # Store the mean correlation across ROI vertices
        corr_tfmri_fmri_roi[roi][fs] = np.nanmean(corr, 0)
        del corr


# =============================================================================
# Bootstrap the confidence intervals (ROI-wise in silico fMRI vs t-fMRI
# correlation scores)
# =============================================================================
ci_corr_tfmri_fmri_roi = {}
ci_corr_tfmri_fmri_roi_peak_lat = {}

for key, val in tqdm(corr_tfmri_fmri_roi.items()):

    ci_corr_tfmri_fmri_roi[key] = np.zeros((2, n_time))
    ci_corr_tfmri_fmri_roi_peak_lat[key] = np.zeros((2))
    corr_dist = np.zeros((args.n_iter, n_time))
    peak_lat_dist = np.zeros((args.n_iter))

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.fmri_subjects)))
        corr_dist[i] = np.mean(val[idx], 0)
        peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

    ci_corr_tfmri_fmri_roi[key][0] = np.percentile(corr_dist, 2.5, axis=0)
    ci_corr_tfmri_fmri_roi[key][1] = np.percentile(corr_dist, 97.5, axis=0)
    ci_corr_tfmri_fmri_roi_peak_lat[key][0] = np.percentile(peak_lat_dist, 2.5)
    ci_corr_tfmri_fmri_roi_peak_lat[key][1] = np.percentile(peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'times': times,
    'corr_tfmri_fmri': corr_tfmri_fmri,
    'corr_tfmri_fmri_roi': corr_tfmri_fmri_roi,
    'ci_corr_tfmri_fmri_roi': ci_corr_tfmri_fmri_roi,
    'ci_corr_tfmri_fmri_roi_peak_lat': ci_corr_tfmri_fmri_roi_peak_lat
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'stats',
    f'source_dataset-{args.source_dataset}')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)