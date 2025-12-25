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
    siubject=1
)
times = metadata_eeg['eeg']['times']

# Empty metadata list
metadata = []

n_sub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times)

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

        fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri'][:]


# =============================================================================
# Load the in t-fMRI test responses
# =============================================================================
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'tfmri_responses', 'things_eeg_2_test_images')
        file_name = f'tfmri_sub-{sub:02d}_hemi-{hemi}.h5'

        tfmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'][:]


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
        threshold = 0.2
        if h == 0:
            ncsnr = metadata[s]['fmri'][f'{hemi}_ncsnr']
            enc_acc = metadata[s]['encoding_models']\
                [f'{hemi}_correlation_nsdcore']
        else:
            ncsnr = np.append(ncsnr, metadata[s]['fmri'][f'{hemi}_ncsnr'])
            enc_acc = np.append(enc_acc, metadata[s]['encoding_models']\
                [f'{hemi}_correlation_nsdcore'])
            correlation = np.append(corr_tfmri_fmri[s,0], corr_tfmri_fmri[s,1],
                1)
            vertex_idx = {}
            vertex_idx['all'] = np.arange(n_vertex*2)
            vertex_idx['below_threshold'] = np.where(ncsnr < 0.2)[0]
            vertex_idx['above_threshold'] = np.where(ncsnr >= 0.2)[0]

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
                corr = correlation[s,val] - correlation[s,val].mean(axis=0)

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
# Compute the significance (in silico fMRI vs t-fMRI correlation scores)
# =============================================================================
# Calculate the p-values with t-tests
pval = ttest_1samp(corr_tfmri_fmri, 0, axis=0, alternative='two-sided')[1]

# Correct for multiple comparisons
shape = pval.shape
sig_corr_tfmri_fmri = multipletests(pval.flatten(), 0.05, 'fdr_bh')[0]
sig_corr_tfmri_fmri = np.reshape(sig_corr_tfmri_fmri, (shape))


# =============================================================================
# Compute the significance (t-fMRI encoding accuracies vs. NSD NCSNR and in
# silico fMRI encodimg accuracies)
# =============================================================================
# Empty result dictionaries
sig_corr_fmri_ncsnr = {}
sig_corr_insilico_fmri_encoding_acc = {}

# Calculate the p-values with t-tests
for key in corr_fmri_ncsnr.keys():
    pval_corr_fmri_ncsnr = ttest_1samp(corr_fmri_ncsnr[key], 0, axis=0,
        alternative='two-sided')[1]
    pval_corr_insilico_fmri_encoding_acc = ttest_1samp(corr_fmri_ncsnr[key], 0,
        axis=0, alternative='two-sided')[1]

    # Correct for multiple comparisons
    sig_corr_fmri_ncsnr[key] = multipletests(pval_corr_fmri_ncsnr,
        0.05, 'fdr_bh')[0]
    sig_corr_insilico_fmri_encoding_acc[key] = multipletests(
        pval_corr_insilico_fmri_encoding_acc, 0.05, 'fdr_bh')[0]


# =============================================================================
# Bootstrap the confidence intervals (t-fMRI encoding accuracies vs. NSD NCSNR
# and in silico fMRI encodimg accuracies)
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
        encoding_acc_dist[i] = np.mean(corr_insilico_fmri_encoding_acc[idx], 0)

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
    'corr_fmri_ncsnr': corr_fmri_ncsnr,
    'corr_insilico_fmri_encoding_acc': corr_insilico_fmri_encoding_acc,
    'sig_corr_tfmri_fmri': sig_corr_tfmri_fmri,
    'sig_corr_fmri_ncsnr': sig_corr_fmri_ncsnr,
    'sig_corr_insilico_fmri_encoding_acc': sig_corr_insilico_fmri_encoding_acc,
    'ci_corr_fmri_ncsnr': ci_corr_fmri_ncsnr,
    'ci_corr_insilico_fmri_encoding_acc': ci_corr_insilico_fmri_encoding_acc
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_accuracy')
os.makedirs(save_dir, exist_ok=True)

file_name = 'encoding_fusion_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)