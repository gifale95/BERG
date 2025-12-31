"""Test the categorical selectivity of high-level visual cortex ROIs on in
t-fMRI responses.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemisphere : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
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
from berg import BERG
from tqdm import tqdm
import random
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> HVC selectivity - Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the t-fMRI responses
# =============================================================================
# Loop across subjects
tfmri = {}
metadata = []
for s, sub in enumerate(args.fmri_subjects):

    # Get the subject's metadata
    berg = BERG(berg_dir=args.berg_dir)
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=sub
    ))

    # Loop across hemipsheres
    for h, hem in enumerate(args.hemispheres):

        # Load the t-fMRI responses
        file_name = f'tfmri_sub-{sub:02d}_hemi-{hem}.npy'
        tfmri_path = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'hvc_selectivity_single_eeg_trials', 'tfmri_responses', file_name)
        tfmri_sub = np.load(tfmri_path, allow_pickle=True).item()

        # Empty t-fMRI response arrays
        if s == 0:
            for key in tfmri_sub.keys():
                if h == 0:
                    tfmri[key] = {}
                # Empty t-fMRI response array of shape:
                # (8 Subjects, 163,842 Vertices, 140 Time points)
                tfmri[key][hem] = np.zeros(
                    (len(args.fmri_subjects), tfmri_sub[key].shape[0],
                    tfmri_sub[key].shape[1]), dtype=np.float32)

        # Store the t-fMRI responses
        for key in tfmri.keys():
            tfmri[key][hem][s,:,:] = tfmri_sub[key]
        del tfmri_sub

# Load the EEG times
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# NCSNR and prediction accuracy thresholding
# =============================================================================
# Only retain vertices that have above threshold (i) NCSNR AND (ii) encoding
# prediction accuracy.

# Loop across subjects and hemispheres
for s in range(len(metadata)):
    for hem in args.hemispheres:

        ncsnr = metadata[s]['fmri'][hem+'_ncsnr']
        idx_ncsnr = ncsnr > args.ncsnr_threshold
        encoding = metadata[s]['encoding_models'][hem+'_explained_variance_nsdcore']
        idx_encoding = encoding > args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

        for key in tfmri.keys():
            tfmri[key][hem][s,idx_nan] = np.nan


# =============================================================================
# Get the mean ROI response across vertices for each category
# =============================================================================
# Empty results dictionary
tfmri_roi_avg = {}

# Loop across categories and ROIs
categories = ['Bodies', 'Faces', 'Objects', 'Scenes']
rois = ['EBA', 'FBA', 'FFA', 'OFA', 'PPA', 'OPA', 'RSC']
for c, cat in enumerate(categories):
    for roi in rois:

        # Empty arrays of shape:
        # (8 Subjects, 163,842 Vertices, 140 Time points)
        tfmri_roi_avg[roi+'_'+cat] = np.zeros((
            len(args.fmri_subjects), len(times)), dtype=np.float32)

        # Loop across subjects
        for s in range(len(args.fmri_subjects)):

            # Empty subject response list
            vertex_mean_resp_sub = []

            # Loop across hemispheres
            for hem in args.hemispheres:

                # Get the vertex indices for the ROI
                if roi == 'FFA' or roi == 'FBA':
                    # Get the vertex indices for both parts of the ROI
                    idx = np.append(
                        metadata[s]['fmri'][hem+'_fsaverage_rois'][f'{roi}-1'],
                        metadata[s]['fmri'][hem+'_fsaverage_rois'][f'{roi}-2'])
                    idx.sort()
                else:
                    # Get the vertex indices for the ROI
                    idx = metadata[s]['fmri'][hem+'_fsaverage_rois'][roi]

                # Store the responses across selected vertices
                vertex_mean_resp_sub.append(tfmri[cat][hem][s,idx])

            # Compute the mean ROI response across vertices
            tfmri_roi_avg[roi+'_'+cat][s] = np.nanmean(np.concatenate(
                vertex_mean_resp_sub), 0)


# =============================================================================
# Compute significant difference between responses for different categories
# =============================================================================
# Empty result dictionary
sig_cat_diff = {}

# Loop across target categories and ROIs
rois = [['EBA', 'FBA'], ['FFA', 'OFA'], [], ['PPA', 'OPA', 'RSC']]
for c, cat in enumerate(categories):
    for roi in rois[c]:

        # Loop across non-target categories
        other_cat = [item for item in categories if item != cat]
        for oc, ocat in enumerate(other_cat):

            # Compute the significance
            pval_cat_diff = ttest_rel(tfmri_roi_avg[roi+'_'+cat],
                tfmri_roi_avg[roi+'_'+ocat], alternative='greater')[1]
            
            # Correct for multiple comparisons
            sig_cat_diff[roi+'_'+cat+'_>'+ocat] = multipletests(pval_cat_diff,
                0.05, 'fdr_bh')[0]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Empty result variables
ci_tfmri_roi_avg = {}
dist = {}
for key, val in tfmri_roi_avg.items():
    ci_tfmri_roi_avg[key] = np.zeros((2, len(times)))
    dist[key] = np.zeros((args.n_iter, len(times)))

# Create the bootstrap distribution
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(metadata)))
    for key, val in tfmri_roi_avg.items():
        dist[key][i] = np.mean(val[idx], 0)

# Compute the CIs from the bootstrap distribution
for key in tfmri_roi_avg.keys():
    ci_tfmri_roi_avg[key][0] = np.percentile(dist[key], 2.5, axis=0)
    ci_tfmri_roi_avg[key][1] = np.percentile(dist[key], 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'tfmri_roi_avg': tfmri_roi_avg,
    'sig_cat_diff': sig_cat_diff,
    'ci_tfmri_roi_avg': ci_tfmri_roi_avg,
    'times': times
    }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'hvc_selectivity_single_eeg_trials',
    'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)