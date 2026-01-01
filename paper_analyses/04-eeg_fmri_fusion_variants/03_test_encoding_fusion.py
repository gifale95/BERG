"""Test the EEG-fMRI encoding fusion models by correlating the t-fMRI responses
for the 200 THINGS EEG2 test images with the corresponding in silico fMRI
responses (independently for each vertex and time point).

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemisphere : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects.
    Valid subject identifiers are integers from 1 10.
eeg_reps : str
    If 'average' average the EEG responses across repeats. If 'single', use the
    single-trial EEG responses.
regression : str
    If 'linear', apply PCA to the EEG responses. If 'ridge', do not apply PCA.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
import h5py
from tqdm import tqdm
from berg import BERG
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--eeg_subject', default=[1, 2], type=list)
parser.add_argument('--eeg_reps', default='single', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result arrays
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the EEG time points
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
n_rep = 4
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']
n_roi = len(rois)

# Empty correlation array of shape:
# (8 fMRI subjects, 2 hemispheres, 163842 fMRI vertices, 140 EEG time points)
corr_tfmri_fmri = np.zeros((n_fsub, n_hemi, n_vertex, n_time), dtype=np.float32)

# Only select vertices falling within the NSD visual streams
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=1
    )
idx_v = np.zeros(n_vertex, dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]


# =============================================================================
# Load the in silico fMRI and t-fMRI test responses
# =============================================================================
# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=fsub
        ))

    # Loop across EEG hemispheres
    for h, hemi in enumerate(args.hemispheres):

        # Load the in silico fMRI test responses
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'insilico_fmri_responses')
        file_name = f'things_eeg_2_test_sub-{fsub:02d}_{hemi}.h5'
        fmri_test = h5py.File(os.path.join(data_dir, file_name),
            'r')['fmri'][:,idx_v]

        # Loop across EEG subjects
        for e, esub in enumerate(args.eeg_subject):

            # Load the t-fMRI test responses
            data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
                'tfmri_responses', 'things_eeg_2_test_images',
                f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
            file_name = (f'tfmri_sub-{fsub:02d}_hemi-{hemi}'
                f'_eeg_sub-{esub:02d}.h5')
            tfmri_test = h5py.File(os.path.join(data_dir, file_name),
                'r')['tfmri'][:]


# =============================================================================
# Correlate the t-fMRI with the in silico fMRI responses
# =============================================================================
            # Loop across fMRI vertices
            for v in range(tfmri_test.shape[1]):

                # Center the data
                fmri = fmri_test[:,v] - fmri_test[:,v].mean(axis=0)
                tfmri = tfmri_test[:,v] - tfmri_test[:,v].mean(axis=0)

                # Normalize the data
                fmri /= np.linalg.norm(fmri, axis=0)
                tfmri /= np.linalg.norm(tfmri, axis=0)

                # Compute the correlations
                if args.eeg_reps == 'average':
                    corr_tfmri_fmri[fs,h,v] += fmri @ tfmri
                elif args.eeg_reps == 'single':
                    for r in range(n_rep):
                        corr_tfmri_fmri[fs,h,v] += fmri @ tfmri[:,r]
                del fmri, tfmri
            del tfmri_test

        # Average the correlation scores across EEG subjects and repeats
        corr_tfmri_fmri[fs,h] /= (len(args.eeg_subject) * (1 if args.eeg_reps == 'average' else n_rep))
        del fmri_test


# =============================================================================
# Get the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# # Empty result dictionary
# corr_tfmri_fmri_roi = {}

# # Loop across ROIs
# for r, roi in enumerate(rois):

#     # Empty ROI correlation array of shape:
#     # (8 subjects, 140 EEG time points)
#     corr_tfmri_fmri_roi[roi] = np.zeros((n_sub, n_time), dtype=np.float32)

#     # Loop across subjects and hemispheres
#     for s, sub in enumerate(tqdm(args.fmri_subjects)):
#         for h, hemi in enumerate(args.hemispheres):

#             # Get the indices of the ROI vertices
#             if roi in ['V1', 'V2', 'V3']:
#                 idx_roi = np.append(
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
#                 idx_roi.sort()
#             elif roi in ['FFA', 'VWFA', 'FBA']:
#                 idx_roi = np.append(
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
#                 idx_roi.sort()
#             elif roi in ['intermediate']:
#                 idx_roi = np.append(
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
#                 idx_roi = np.append(idx_roi,
#                     metadata[s]['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
#                 idx_roi.sort()
#             else:
#                 idx_roi = metadata[s]['fmri'][f'{hemi}_fsaverage_rois'][roi]

#             # NCNSR vertex selection
#             ncsnr = metadata[s]['fmri'][f'{hemi}_ncsnr'][idx_roi]
#             idx_ncsnr = np.where(ncsnr >= args.ncsnr_threshold)[0]
#             idx_roi = idx_roi[idx_ncsnr]

#             # Get the correlation scores of the above-NCSNR-threshold vertices
#             # from the chosen ROI
#             if h == 0:
#                 corr = corr_tfmri_fmri[s,h,idx_roi]
#             else:
#                 corr = np.append(corr, corr_tfmri_fmri[s,h,idx_roi], 0)
#                 corr_tfmri_fmri_roi[roi][s] = np.mean(corr, 0)


# =============================================================================
# Compute the significance (in silico fMRI vs t-fMRI correlation scores)
# =============================================================================
# # Calculate the p-values with t-tests
# pval = ttest_1samp(corr_tfmri_fmri, 0, axis=0, alternative='greater')[1]

# # Correct for multiple comparisons
# shape = pval.shape
# sig_corr_tfmri_fmri = multipletests(pval.flatten(), 0.05, 'fdr_bh')[0]
# sig_corr_tfmri_fmri = np.reshape(sig_corr_tfmri_fmri, (shape))


# =============================================================================
# Compute the significance (ROI-wise in silico fMRI vs t-fMRI correlation
# scores)
# =============================================================================
# sig_corr_tfmri_fmri_roi = {}

# for key, val in corr_tfmri_fmri_roi.items():

#     # Calculate the p-values with t-tests
#     pval = ttest_1samp(val, 0, axis=0, alternative='greater')[1]

#     # Correct for multiple comparisons
#     sig_corr_tfmri_fmri_roi[key] = multipletests(pval, 0.05, 'fdr_bh')[0]


# =============================================================================
# Bootstrap the confidence intervals (ROI-wise in silico fMRI vs t-fMRI
# correlation scores)
# =============================================================================
# ci_corr_tfmri_fmri_roi = {}
# ci_corr_tfmri_fmri_roi_peak_lat = {}

# for key, val in tqdm(corr_tfmri_fmri_roi.items()):

#     ci_corr_tfmri_fmri_roi[key] = np.zeros((2, n_time))
#     ci_corr_tfmri_fmri_roi_peak_lat[key] = np.zeros((2))
#     corr_dist = np.zeros((args.n_iter, n_time))
#     peak_lat_dist = np.zeros((args.n_iter))

#     for i in range(args.n_iter):
#         idx = resample(np.arange(len(args.fmri_subjects)))
#         corr_dist[i] = np.mean(val[idx], 0)
#         peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

#     ci_corr_tfmri_fmri_roi[key][0] = np.percentile(corr_dist, 2.5, axis=0)
#     ci_corr_tfmri_fmri_roi[key][1] = np.percentile(corr_dist, 97.5, axis=0)
#     ci_corr_tfmri_fmri_roi_peak_lat[key][0] = np.percentile(peak_lat_dist, 2.5)
#     ci_corr_tfmri_fmri_roi_peak_lat[key][1] = np.percentile(peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
# results = {
#     'metadata': metadata,
#     'times': times,
#     'corr_tfmri_fmri': corr_tfmri_fmri,
#     'corr_tfmri_fmri_roi': corr_tfmri_fmri_roi,
#     'sig_corr_tfmri_fmri': sig_corr_tfmri_fmri,
#     'sig_corr_tfmri_fmri_roi': sig_corr_tfmri_fmri_roi,
#     'ci_corr_tfmri_fmri_roi': ci_corr_tfmri_fmri_roi,
#     'ci_corr_tfmri_fmri_roi_peak_lat': ci_corr_tfmri_fmri_roi_peak_lat
# }

results = {
    'metadata': metadata,
    'times': times,
    'corr_tfmri_fmri': corr_tfmri_fmri
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
    'encoding_fusion_accuracy',
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
os.makedirs(save_dir, exist_ok=True)

file_name = 'encoding_fusion_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)