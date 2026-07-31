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
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr
from berg import BERG
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


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

# Empty metadata list
metadata = []

n_sub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']
n_roi = len(rois)

# Empty correlation array of shape:
# (8 subjects, 2 hemispheres, 163842 fMRI vertices, 140 EEG time points)
corr_tfmri_fmri = np.zeros((n_sub, n_hemi, n_vertex, n_time), dtype=np.float32)

# Empty correlation dictionaries
corr_fmri_ncsnr = {}
corr_insilico_fmri_encoding_acc = {}


# =============================================================================
# Loop across subjects and hemispheres
# =============================================================================
for s, sub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=sub
        ))

    for h, hemi in enumerate(args.hemispheres):


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'insilico_fmri_responses')
        file_name = f'things_eeg_2_test_sub-{sub:02d}_{hemi}.h5'

        fmri_test = h5py.File(os.path.join(data_dir, file_name),
            'r')['fmri'][:]


# =============================================================================
# Load the in t-fMRI test responses
# =============================================================================
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'tfmri_responses', 'things_eeg_2_test_images')
        file_name = f'tfmri_sub-{sub:02d}_hemi-{hemi}.h5'

        tfmri_test = h5py.File(os.path.join(data_dir, file_name),
            'r')['tfmri'][:]


# =============================================================================
# Correlate the t-fMRI with the in silico fMRI responses
# =============================================================================
        # Loop across fMRI vertices
        for v in range(tfmri_test.shape[1]):

            # Center the data
            fmri = fmri_test[:,v] - fmri_test[:,v].mean()
            tfmri = tfmri_test[:,v] - tfmri_test[:,v].mean(axis=0)

            # Normalize the data
            fmri /= np.linalg.norm(fmri)
            tfmri /= np.linalg.norm(tfmri, axis=0)

            # Compute the correlations
            corr_tfmri_fmri[s,h,v] = fmri @ tfmri
            del fmri, tfmri
        del fmri_test, tfmri_test


# =============================================================================
# Correlate the t-fMRI encoding accuracies with the NSD NCSNR and the in silico
# fMRI encoding accuracy
# =============================================================================
        # Three types of correlations are performed:
        # (1) Using all vertices.
        # (2) Using vertices with NCSNR above threshold.
        # (3) Using vertices with NCSNR below threshold.

        # Get the vertex indices for the three correlation types
        if h == 0:
            ncsnr = metadata[s]['fmri'][f'{hemi}_ncsnr']
            enc_acc = metadata[s]['encoding_models']\
                [f'{hemi}_correlation_nsdcore']
        else:
            ncsnr = np.append(ncsnr, metadata[s]['fmri'][f'{hemi}_ncsnr'])
            enc_acc = np.append(enc_acc, metadata[s]['encoding_models']\
                [f'{hemi}_correlation_nsdcore'])
            correlation = np.append(corr_tfmri_fmri[s,0], corr_tfmri_fmri[s,1],
                0)
            vertex_idx = {}
            vertex_idx['all'] = np.arange(n_vertex*2)
            vertex_idx['below_threshold'] = np.where(
                ncsnr < args.ncsnr_threshold)[0]
            vertex_idx['above_threshold'] = np.where(
                ncsnr >= args.ncsnr_threshold)[0]

            # Loop across correlation types
            for key, val in vertex_idx.items():

                # Empty correlation arrays of shape:
                # (8 subjects, 140 EEG time points)
                if s == 0:
                    corr_fmri_ncsnr[key] = np.zeros((n_sub, n_time),
                        dtype=np.float32)
                    corr_insilico_fmri_encoding_acc[key] = np.zeros((
                        n_sub, n_time), dtype=np.float32)

                # Center the data
                nc = ncsnr[val] - ncsnr[val].mean()
                acc = enc_acc[val] - enc_acc[val].mean()
                corr = correlation[val] - correlation[val].mean(axis=0)

                # Normalize the data
                nc /= np.linalg.norm(nc)
                acc /= np.linalg.norm(acc)
                corr /= np.linalg.norm(corr, axis=0)

                # Compute the correlations
                corr_fmri_ncsnr[key][s] = nc @ corr
                corr_insilico_fmri_encoding_acc[key][s] = acc @ corr
                del nc, acc, corr
            del ncsnr, enc_acc, correlation


# =============================================================================
# Get the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# Empty result dictionary
corr_tfmri_fmri_roi = {}

# Loop across ROIs
for r, roi in enumerate(rois):

    # Empty ROI correlation array of shape:
    # (8 subjects, 140 EEG time points)
    corr_tfmri_fmri_roi[roi] = np.zeros((n_sub, n_time), dtype=np.float32)

    # Loop across subjects and hemispheres
    for s, sub in enumerate(tqdm(args.fmri_subjects)):
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

            # NCNSR vertex selection
            ncsnr = metadata[s]['fmri'][f'{hemi}_ncsnr'][idx_roi]
            idx_ncsnr = np.where(ncsnr >= args.ncsnr_threshold)[0]
            idx_roi = idx_roi[idx_ncsnr]

            # Get the correlation scores of the above-NCSNR-threshold vertices
            # from the chosen ROI
            if h == 0:
                corr = corr_tfmri_fmri[s,h,idx_roi]
            else:
                corr = np.append(corr, corr_tfmri_fmri[s,h,idx_roi], 0)
                corr_tfmri_fmri_roi[roi][s] = np.mean(corr, 0)


# =============================================================================
# Compute the significance (in silico fMRI vs t-fMRI correlation scores)
# =============================================================================
# Calculate the p-values with t-tests
pval = ttest_1samp(corr_tfmri_fmri, 0, axis=0, alternative='greater')[1]

# Correct for multiple comparisons
shape = pval.shape
sig_corr_tfmri_fmri = multipletests(pval.flatten(), 0.05, 'fdr_bh')[0]
sig_corr_tfmri_fmri = np.reshape(sig_corr_tfmri_fmri, (shape))


# =============================================================================
# Compute the significance (ROI-wise in silico fMRI vs t-fMRI correlation
# scores)
# =============================================================================
sig_corr_tfmri_fmri_roi = {}

for key, val in corr_tfmri_fmri_roi.items():

    # Calculate the p-values with t-tests
    pval = ttest_1samp(val, 0, axis=0, alternative='greater')[1]

    # Correct for multiple comparisons
    sig_corr_tfmri_fmri_roi[key] = multipletests(pval, 0.05, 'fdr_bh')[0]


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
# Compute the significance (t-fMRI encoding accuracies vs. NSD NCSNR and in
# silico fMRI encoding accuracies)
# =============================================================================
# Empty result dictionaries
sig_corr_fmri_ncsnr = {}
sig_corr_insilico_fmri_encoding_acc = {}

# Calculate the p-values with t-tests
for key in corr_fmri_ncsnr.keys():
    pval_corr_fmri_ncsnr = ttest_1samp(corr_fmri_ncsnr[key], 0, axis=0,
        alternative='greater')[1]
    pval_corr_insilico_fmri_encoding_acc = ttest_1samp(corr_fmri_ncsnr[key], 0,
        axis=0, alternative='greater')[1]

    # Correct for multiple comparisons
    sig_corr_fmri_ncsnr[key] = multipletests(pval_corr_fmri_ncsnr,
        0.05, 'fdr_bh')[0]
    sig_corr_insilico_fmri_encoding_acc[key] = multipletests(
        pval_corr_insilico_fmri_encoding_acc, 0.05, 'fdr_bh')[0]


# =============================================================================
# Bootstrap the confidence intervals (t-fMRI encoding accuracies vs. NSD NCSNR
# and in silico fMRI encoding accuracies)
# =============================================================================
ci_corr_fmri_ncsnr = {}
ci_corr_insilico_fmri_encoding_acc = {}

for key in corr_fmri_ncsnr.keys():

    ci_corr_fmri_ncsnr[key] = np.zeros((2, n_time))
    ci_corr_insilico_fmri_encoding_acc[key] = np.zeros((2, n_time))

    ncsnr_dist = np.zeros((args.n_iter, n_time))
    encoding_acc_dist = np.zeros((args.n_iter, n_time))

    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.fmri_subjects)))
        ncsnr_dist[i] = np.mean(corr_fmri_ncsnr[key][idx], 0)
        encoding_acc_dist[i] = np.mean(
            corr_insilico_fmri_encoding_acc[key][idx], 0)

    ci_corr_fmri_ncsnr[key][0] = np.percentile(ncsnr_dist, 2.5, axis=0)
    ci_corr_fmri_ncsnr[key][1] = np.percentile(ncsnr_dist, 97.5, axis=0)
    ci_corr_insilico_fmri_encoding_acc[key][0] = np.percentile(
        encoding_acc_dist, 2.5, axis=0)
    ci_corr_insilico_fmri_encoding_acc[key][1] = np.percentile(
        encoding_acc_dist, 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'times': times,
    'corr_tfmri_fmri': corr_tfmri_fmri,
    'corr_tfmri_fmri_roi': corr_tfmri_fmri_roi,
    'corr_fmri_ncsnr': corr_fmri_ncsnr,
    'corr_insilico_fmri_encoding_acc': corr_insilico_fmri_encoding_acc,
    'sig_corr_tfmri_fmri': sig_corr_tfmri_fmri,
    'sig_corr_tfmri_fmri_roi': sig_corr_tfmri_fmri_roi,
    'sig_corr_fmri_ncsnr': sig_corr_fmri_ncsnr,
    'sig_corr_insilico_fmri_encoding_acc': sig_corr_insilico_fmri_encoding_acc,
    'ci_corr_tfmri_fmri_roi': ci_corr_tfmri_fmri_roi,
    'ci_corr_tfmri_fmri_roi_peak_lat': ci_corr_tfmri_fmri_roi_peak_lat,
    'ci_corr_fmri_ncsnr': ci_corr_fmri_ncsnr,
    'ci_corr_insilico_fmri_encoding_acc': ci_corr_insilico_fmri_encoding_acc
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_accuracy')
os.makedirs(save_dir, exist_ok=True)

file_name = 'encoding_fusion_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)