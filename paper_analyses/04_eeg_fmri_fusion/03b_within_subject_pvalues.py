"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 200 THINGS EEG2 test images, and use these responses
to create the prediction accuracy null distribution of each ROI through
permutation tests. These null distributions are then used to compute the
within-subject p-values.

To reduce computational load, the EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

The in vivo THINGS EEG2 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_eeg_2

Parameters
----------
fmri_subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values are:
    'lh' (left hemisphere) and 'rh' (right hemisphere).
roi : str
    Name of used ROI.
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'all' (all trials), 'even' (even trials), and 'odd' (odd
    trials).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
n_iter : int
    Amount of iterations for creating the prediction accuracy null distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import gc
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_train_trials', default='all', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Encoding fusion null distribution <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Get the fMRI ROI indices
# =============================================================================
# Load the fMRI metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Loop across hemispheres
idx_streams = {}
idx_roi_ncsnr = {}
for hemi in args.hemispheres:

    # Only select vertices falling within the NSD visual streams
    n_vertices = 163842
    idx_streams_hemi = np.zeros(n_vertices, dtype=int)
    streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
        'lateral', 'parietal']
    for stream in streams:
        idx_streams_hemi[metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
    idx_streams_hemi = np.where(idx_streams_hemi == 1)[0]
    idx_streams[hemi] = idx_streams_hemi
    del idx_streams_hemi

    # Get the indices of ROI vertices within the NSD visual streams
    if args.roi in ['V1', 'V2', 'V3']:
        idx_r = np.append(
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}v'],
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}d'])
        idx_r.sort()
    elif args.roi in ['FFA', 'VWFA', 'FBA']:
        idx_r = np.append(
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}-1'],
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}-2'])
        idx_r.sort()
    else:
        idx_r = metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}']
        idx_r.sort()
    idx_roi = np.zeros(n_vertices, dtype=bool)
    idx_roi[idx_r] = 1
    idx_roi = idx_roi[idx_streams[hemi]]
    idx_roi = np.where(idx_roi)[0]

    # NCSNR vertex selection
    ncsnr = metadata_fmri['fmri'][hemi+'_ncsnr'][idx_streams[hemi]]
    idx_ncsnr = ncsnr >= args.ncsnr_threshold
    idx_ncsnr = np.where(idx_ncsnr)[0]
    idx_roi_ncsnr[hemi] = np.intersect1d(idx_roi, idx_ncsnr)


# =============================================================================
# Load the correlation results
# =============================================================================
# Loop across fMRI hemispheres
for h, hemi in enumerate(args.hemispheres):

    # Load and store the correlation scores
    data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_accuracy')
    file_name = (f'corr_fmri_sub-{args.fmri_subject:02d}_hemi-{hemi}_'
        f'eeg_train_trials-{args.eeg_train_trials}.npy')
    corr_hemi = np.load(os.path.join(data_dir, file_name))[idx_roi_ncsnr[hemi]]

    # Append the correlation scores across hemispheres, and average them across
    # ROI vertices
    if h == 0:
        corr = corr_hemi
    else:
        corr = np.mean(np.append(corr, corr_hemi, 0), 0)


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
# Loop across hemispheres
fmri_test_z = {}
for hemi in args.hemispheres:

    # Load the fMRI responses
    data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'insilico_fmri_responses', 'imageset-things_eeg_2')
    file_name = f'fmri_sub-{args.fmri_subject:02d}_{hemi}_split-test.h5'
    fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

    # Only select the fMRI vertices falling within the NSD visual streams and
    # from the selected ROI
    fmri_test = fmri_test[:,idx_streams[hemi]]
    fmri_test = fmri_test[:,idx_roi_ncsnr[hemi]]
    # Center and normalize the test fMRI responses (for later correlation)
    eps = 1e-8
    fmri_test_z[hemi] = (fmri_test - fmri_test.mean(0)) / \
        (fmri_test.std(0) + eps)
    del fmri_test


# =============================================================================
# Load and append the in vivo EEG test responses across subjects
# =============================================================================
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
    eeg_test_sub = np.mean(h5py.File(eeg_dir_test, 'r')['eeg'][:],
        1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_test = eeg_test_sub
    else:
        eeg_test = np.append(eeg_test, eeg_test_sub, 1)
    del eeg_test_sub


# =============================================================================
# EEG time point selection
# =============================================================================
# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = np.round(metadata_eeg['eeg']['times'], 3)

# Select EEG time points between 0-400 ms
tmin = np.where(times >= 0)[0][0]
tmax = np.where(times <= 0.4)[0][-1]


# =============================================================================
# Create the permutation indices for the prediction accuracy null distribution
# =============================================================================
n_img = fmri_test_z[hemi].shape[0]

idx_perm = np.zeros((n_img, args.n_iter), dtype=np.int16)
idx_perm_pred = np.zeros((n_img, args.n_iter), dtype=np.int16)

for i in range(args.n_iter):

    idx_perm[:,i] = resample(np.arange(n_img), replace=False)
    idx_perm_pred[:,i] = resample(np.arange(n_img), replace=False)


# =============================================================================
# Loop across EEG time points and hemispheres
# =============================================================================
# Within-subject p-value array of shape (N EEG time points)
within_subj_pval = np.zeros(len(times), dtype=np.float32)
within_subj_pval[:] = np.nan

# Loop across EEG time points and hemispheres
for t in tqdm(range(tmin, tmax+1)):
    for h, hemi in enumerate(args.hemispheres):


# =============================================================================
# Create the prediction accuracy null distribution
# =============================================================================
        # Empty correlation null distribution array for each of shape:
        # (N ROI/hemi vertices, N iterations)
        corr_null_dist_hemi = np.zeros((len(idx_roi_ncsnr[hemi]), args.n_iter),
            dtype=np.float32)

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_hemi-'
            f'{hemi}_eeg_train_trials-{args.eeg_train_trials}_'
            f'eeg_time-{t:03d}.npy')
        reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'encoding_fusion_weights', file_name), allow_pickle=True).item()

        # Instantiate the fusion regression model
        reg = LinearRegression()
        reg.coef_ = reg_param['coef_'][idx_roi_ncsnr[hemi]]
        reg.intercept_ = reg_param['intercept_'][idx_roi_ncsnr[hemi]]
        reg.n_features_in_ = reg_param['n_features_in_']
        del reg_param

        # Generate the t-fMRI responses for the test images with in vivo EEG
        tfmri = reg.predict(eeg_test[:,:,t])
        del reg

        # Center and normalize the t-fMRI responses
        tfmri_z = (tfmri - tfmri.mean(0)) /  (tfmri.std(0) + eps)

        # Loop across iterations
        for i in range(args.n_iter):

            # Permute the t-fMRI and fMRI test responses
            tfmri_perm = tfmri_z[idx_perm_pred[:,i]]
            fmri_perm = fmri_test_z[hemi][idx_perm[:,i]]

            # Correlate the t-fMRI test responses with the fMRI test responses
            corr_null_dist_hemi[:,i] = \
                np.diag(tfmri_perm.T @ fmri_perm) / len(tfmri_perm)
        del tfmri_perm, fmri_perm

        # Delete unused variables
        del tfmri, tfmri_z
        gc.collect()

        # Append the null distribution across hemispheres, and average it across
        # ROI vertices
        if h == 0:
            corr_null_dist = corr_null_dist_hemi
        else:
            corr_null_dist = np.mean(np.append(
                corr_null_dist, corr_null_dist_hemi, 0), 0)
        del corr_null_dist_hemi


# =============================================================================
# Compute the within-subject p-value
# =============================================================================
    within_subj_pval[t] = (np.sum(corr_null_dist >= corr[t]) + 1) / \
        (args.n_iter + 1)


# =============================================================================
# Save the results
# =============================================================================
# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_accuracy_within_subject_pvalues')
os.makedirs(save_dir, exist_ok=True)

# Save the correlation scores
file_name = (f'corr_pval_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'eeg_train_trials-{args.eeg_train_trials}.npy')
np.save(os.path.join(save_dir, file_name), within_subj_pval)