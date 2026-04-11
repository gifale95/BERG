"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 100 THINGS MEG1/fMRI1 test images. These t-fMRI
responses are then used to compute granger causality scores between ROIs.

To reduce computational load, the MEG-fMRI fusion encoding models are only
trained, tested, and used for voxels falling within the THINGS fMRI1 visual
ROIs.

The in vivo THINGS MEG1 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_meg_1

The in vivo THINGS fMRI1 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_fmri_1

Parameters
----------
fmri_subject : int
    THINGS fMRI1 subject identifiers. Valid subject identifiers are integers
    from 1 to 3.
rois: list
    List of ROIs between which to compute Granger Causality.
nc_threshold : float
    The threshold on the noise ceiling for fMRI voxel selection.
meg_subjects : list
    List containing the subject identifiers for the THINGS MEG1 subjects. Valid
    subject identifiers are integers from 1 to 4.
time_window_s : int
    Time window in seconds for computing Granger Causality.
offset_s : int
    Offset in seconds for the time window.
regression : str
    The type of regression to use for computing Granger Causality. If 'linear',
    use linear regression. If 'ridge', use RidgeCV regression.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
import random
from berg import BERG
from tqdm import tqdm
from copy import copy
import gc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--rois', default=['V1', 'V2', 'V3', 'hV4', 'IT'], type=list)
parser.add_argument('--nc_threshold', default=20, type=float)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--time_window_s', default=0.1, type=float)
parser.add_argument('--offset_s', default=0.02, type=float)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Granger Causality <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Get the THINGS fMRI1 metadata and test image file names
# =============================================================================
# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

# Get the image files names
unique_test_stimuli = np.unique(
    metadata_fmri['encoding_model']['test_stimuli'])

# Get the noise ceiling
noise_ceiling = metadata_fmri['encoding_model']['noise_ceiling_testset']


# =============================================================================
# Load and append the in vivo THINGS MEG1 test responses across subjects
# =============================================================================
# Loop across MEG subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Load the MEG responses
    meg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_all = h5py.File(meg_dir, 'r')['neural_data']

    # Time point selection
    tmax = 0.595
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]
    meg_all = meg_all[:,:,time_idx].astype(np.float32)

    # Create 4 pseudo-trial repeats using the MEG responses for the images
    # shared with the fMRI
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_sub = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        np.random.shuffle(idx)
        resp_img = []
        pseudotrials = 4
        n_reps = len(idx) // pseudotrials
        for p in range(pseudotrials):
            idx_start = p * n_reps
            idx_end = (p + 1) * n_reps
            idx_pseudo = idx[idx_start:idx_end]
            idx_pseudo.sort()
            resp_img.append(np.mean(meg_all[idx_pseudo], 0))
        meg_sub.append(np.array(resp_img))
        del resp_img

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg = np.array(meg_sub)
    else:
        meg = np.append(meg, np.array(meg_sub), 2)
    del meg_all, meg_sub


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Load the encoding fusion model regression weights
weight_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'encoding_fusion_weights',
    f'weights_fmri_sub-{args.fmri_subject:02d}.npy')
reg_param = np.load(weight_dir, allow_pickle=True).item()

# Loop across ROIs
tfmri = {}
for r, roi in enumerate(tqdm(args.rois)):

    # Only generate responses for voxels with noise ceiling above the threshold
    nc_roi = noise_ceiling[metadata_fmri['roi'][roi]]
    voxels_keep = np.where(nc_roi >= args.nc_threshold)[0]

    # Empty response array of shape:
    # (100 Image conditions, N repeats, N ROI voxels, 140 MEG time points)
    tfmri[roi] = np.zeros((len(unique_test_stimuli), pseudotrials,
        len(voxels_keep), len(times)), dtype=np.float32)

    # Loop across MEG time points
    for t in range(len(times)):

        # Instantiate the fusion regression model
        reg = LinearRegression()
        reg.coef_ = reg_param[roi]['coef_'][t][voxels_keep]
        reg.intercept_ = reg_param[roi]['intercept_'][t][voxels_keep]
        reg.n_features_in_ = reg_param[roi]['n_features_in_'][t]

        # Generate the t-fMRI responses for the test images, independently for
        # each MEG repeat
        for r in range(pseudotrials):
            tfmri[roi][:,r,:,t] = reg.predict(meg[:,r,:,t])

        # Delete unused variables
        del reg
        gc.collect()
del reg_param, metadata_meg, meg


# =============================================================================
# Compute the RSMs of each ROI at each time point
# =============================================================================
# There are multiple RSMs for each ROI and time point, computed by correlating
# the responses between different repeats: these different RSMs are later used
# to cross-validate the regressions used to compute the GC scores

# 6 RSMs in total, divided into 3 splits of 2 RSMs each (one used for training,
# and the other for testing), where each RMS is computed by correlating the
# responses of 2 repeats (the numbers below indicate the repeats used to
# compute each RSM)
rep_splits = [
    [[0, 1], [2, 3]], # Split 1: RSM 1 computed from repeats 0 and 1, RSM 2 computed from repeats 2 and 3
    [[0, 2], [1, 3]], # Split 2: RSM 1 computed from repeats 0 and 2, RSM 2 computed from repeats 1 and 3
    [[0, 3], [1, 2]] # Split 3: RSM 1 computed from repeats 0 and 3, RSM 2 computed from repeats 1 and 2
    ]

roi_rsms = {}
idx_triu = np.triu_indices(len(unique_test_stimuli), k=1)
for roi in tqdm(args.rois):

    roi_rsms[roi] = []

    for s, split in enumerate(rep_splits):

        rsms_split = []

        for r, rep in enumerate(split):

            # Get the responses for the two repetition splits
            X = copy(tfmri[roi][:,rep[0]])
            Y = copy(tfmri[roi][:,rep[1]])

            # Z-score across channels
            X_mean = X.mean(axis=1, keepdims=True)
            Y_mean = Y.mean(axis=1, keepdims=True)
            X_std = X.std(axis=1, keepdims=True)
            Y_std = Y.std(axis=1, keepdims=True)
            X_z = (X - X_mean) / (X_std + 1e-8)
            Y_z = (Y - Y_mean) / (Y_std + 1e-8)

            # Reshape to (time, images, channels)
            X_t = np.transpose(X_z, (2, 0, 1))  # (time, images_X, channels)
            Y_t = np.transpose(Y_z, (2, 0, 1))  # (time, images_Y, channels)

            # Cross-correlation via batch matmul
            rsm = np.matmul(X_t, Y_t.transpose(0, 2, 1)) / (X.shape[1])

            # Back to (images_X, images_Y, time)
            rsm = np.transpose(rsm, (1, 2, 0))

            # Store the upper triangle of the RSMs without the main diagonal
            rsms_split.append(rsm[idx_triu])
            del X, Y, X_z, Y_z, X_t, Y_t, rsm

        # Store the RSMs
        roi_rsms[roi].append(rsms_split)
        del rsms_split


# =============================================================================
# Compute the Granger Causality (RSM averaged over past times)
# =============================================================================
# Get the time indices
t_min = times[0] + args.time_window_s + args.offset_s
idx_t_start = np.where(times == t_min)[0][0]

# Loop across ROIs
granger_c = {}
for roi_target in args.rois:
    for roi_source in args.rois:
        if roi_target != roi_source:

            # Empty result list
            gc_roi = []

            # Loop across time points
            for t in tqdm(range(idx_t_start, len(times))):

                gc_roi_t = []

                # Get the onset and offset time points for the time window
                time_onset = np.round(
                    (times[t] - args.time_window_s - args.offset_s), 3)
                idx_onset = np.where(times == time_onset)[0][0]
                time_offset = np.round((times[t] - args.offset_s), 3)
                idx_offset = np.where(times == time_offset)[0][0]

                # Loop across splits for cross-validation
                for s in range(len(rep_splits)):
                    for r in range(len(rep_splits[s])):

                        # Get the train and test RSMs of the target and source
                        # ROIs, and average them across time points
                        # Train
                        rsm_roi_target_train = np.reshape(
                            roi_rsms[roi_target][s][r][:,t], (-1, 1))
                        rsm_roi_target_past_train = np.mean(
                            roi_rsms[roi_target][s][r][:,idx_onset:idx_offset],
                            1, keepdims=True)
                        rsm_roi_source_past_train = np.mean(
                            roi_rsms[roi_source][s][r][:,idx_onset:idx_offset],
                            1, keepdims=True)
                        # Test (use a different repeat for the test target
                        # than for the test predictors, to reduce the effect of
                        # noise correlations)
                        rsm_roi_target_test = np.reshape(
                            roi_rsms[roi_target][s][abs(r-1)][:,t], (-1, 1))
                        rsm_roi_target_past_test = np.mean(
                            roi_rsms[roi_target][s][r][:,idx_onset:idx_offset],
                            1, keepdims=True)
                        rsm_roi_source_past_test = np.mean(
                            roi_rsms[roi_source][s][r][:,idx_onset:idx_offset],
                            1, keepdims=True)

                        # Fit the linear regressions for the full and reduced
                        # models
                        if args.regression == 'linear':
                            reg_reduced = LinearRegression()
                            reg_full = LinearRegression()
                        elif args.regression == 'ridge':
                            alphas = np.logspace(-6, 10, 17)
                            reg_reduced = RidgeCV(alphas=alphas, cv=None,
                                alpha_per_target=True)
                            reg_full = RidgeCV(alphas=alphas, cv=None,
                                alpha_per_target=True)
                        reg_reduced.fit(rsm_roi_target_past_train,
                            rsm_roi_target_train)
                        reg_full.fit(np.append(rsm_roi_target_past_train,
                            rsm_roi_source_past_train, 1),
                            rsm_roi_target_train)

                        # Compute the unexplained variance for the full and
                        # reduced models (MSE)
                        u_reduced = np.mean((
                            reg_reduced.predict(rsm_roi_target_past_test) -
                            rsm_roi_target_test) ** 2)
                        u_full = np.mean((reg_full.predict(np.append(
                            rsm_roi_target_past_test, rsm_roi_source_past_test,
                            1)) - rsm_roi_target_test) ** 2)

                        # Adjust the MSE scores for the number of predictors in
                        # the models
                        n = len(rsm_roi_target_test)
                        p_reduced = rsm_roi_target_past_train.shape[1]
                        p_full = p_reduced + \
                            rsm_roi_source_past_train.shape[1]
                        u_reduced = u_reduced * (n - 1) / (n - p_reduced - 1)
                        u_full = u_full * (n - 1) / (n - p_full - 1)

                        # Compute the GC score as the log ratio of the
                        # unexplained variance of the reduced and full models
                        gc_roi_t.append(np.log(u_reduced / u_full))

                        # Remove unused variables
                        del rsm_roi_target_train, rsm_roi_target_past_train, \
                            rsm_roi_source_past_train, rsm_roi_target_test, \
                            rsm_roi_target_past_test, \
                            rsm_roi_source_past_test, reg_reduced, reg_full, \
                            u_reduced, u_full

                # Store the GC scores of the multiple RSM splits for the
                # current time point
                gc_roi.append(np.array(gc_roi_t))
                del gc_roi_t

            # Store the GC results in a dictionary
            granger_c[f'{roi_source}_to_{roi_target}'] = np.transpose(np.array(
                gc_roi))
            del gc_roi


# =============================================================================
# Save the results
# =============================================================================
# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'granger_causality')
os.makedirs(save_dir, exist_ok=True)

# Save the Granger causality scores
file_name = f'gc_fmri_sub-{args.fmri_subject:02d}.npy'
np.save(os.path.join(save_dir, file_name), granger_c)