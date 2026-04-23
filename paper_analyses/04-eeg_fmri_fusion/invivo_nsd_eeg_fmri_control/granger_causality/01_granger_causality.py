"""Apply Granger Causality (GC) between the t-fMRI responses of two ROIs.

To reduce computational load, the EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1 to 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
rois : list
    List containing the ROIs used for the Granger Causality analysis. All ROIs
    are tested in a pairwise fashion.
regression : str
    The type of regression to use for computing Granger Causality. If 'linear',
    use linear regression. If 'ridge', use RidgeCV regression.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
from berg import BERG
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--rois', default=['V1', 'V2', 'V3', 'hV4', 'ventral', 'FFA', 'EBA', 'PPA'], type=list)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Granger Causality <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the EEG test responses
# =============================================================================
# Load the EEG responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_test_data')
file_name = f'eeg_test_sub-{args.subject:02d}_.npy'
eeg_test_dict = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()
eeg = eeg_test_dict['eeg_test']
del eeg_test_dict

# Average the EEG responses into 4 pseudo-trials of 7 repeats each (this is to
# create multiple t-fMRI data instances to cross-validate the Granger Causality
# regression models)
n_rep = eeg.shape[1]
n_pseudo = 4
n_rep_per_pseudo = n_rep // n_pseudo
eeg_test = np.zeros((eeg.shape[0], n_pseudo, eeg.shape[2], eeg.shape[3]),
    dtype=np.float32)
shuffle_idx = resample(np.arange(eeg.shape[1]))
eeg = eeg[:,shuffle_idx]
for p in range(n_pseudo):
    start = p * n_rep_per_pseudo
    end = (p + 1) * n_rep_per_pseudo
    eeg_test[:,p] = np.mean(eeg[:,start:end], 1)
del eeg

# Get the time points # !!! Use official time points
n_times = 615
times = np.round(np.linspace(-200, 1000, n_times)).astype(int)
# Account for the 50ms shift in the EEG responses # !!!
shift = -50
times = times + shift
# Only select time points between -100ms and 600ms
t_start = np.where(times == -100)[0][0]
t_end = np.where(times == 600)[0][0]
times = times[t_start:t_end+1]


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the fMRI metadata
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Empty t-fMRI data dictionary
tfmri = {}

# Loop across ROIs
for roi in tqdm(args.rois):

    # Loop across hemisphers
    for h, hemi in enumerate(args.hemispheres):

        # Only select vertices falling within the NSD visual streams
        n_vertices = 163842
        idx_streams = np.zeros(n_vertices, dtype=bool)
        streams = ['early', 'midventral', 'midlateral', 'midparietal',
            'ventral', 'lateral', 'parietal']
        for stream in streams:
            idx_streams[metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
        idx_streams = np.where(idx_streams)[0]

        # Only select stream vertices with NCSNR above threshold
        ncsnr = metadata_fmri['fmri'][f'{hemi}_ncsnr']
        idx_ncsnr = np.where(ncsnr[idx_streams] >= args.ncsnr_threshold)[0]

        # Only select stream vertices of the chosen ROI
        if roi in ['V1', 'V2', 'V3']:
            idx_r = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
            idx_r.sort()
        elif roi in ['FFA', 'VWFA', 'FBA']:
            idx_r = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
            idx_r.sort()
        else:
            idx_r = metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][roi]
            idx_r.sort()
        idx_roi = np.zeros(n_vertices, dtype=bool)
        idx_roi[idx_r] = 1
        idx_roi = idx_roi[idx_streams]
        idx_roi = np.where(idx_roi)[0]

        # Get the indices of ROI vertices with NCSNR above threshold
        idx_v = np.intersect1d(idx_roi, idx_ncsnr)

        # Empty t-fMRI response array of shape:
        # (N Images, N Pseudo Trials, N Vertices, N times)
        tfmri_hemi = np.zeros((len(eeg_test), n_pseudo, len(idx_v),
            len(times)), dtype=np.float32)

        # Loop across EEG time points
        for t in range(len(times)):

            # Load the EEG-fMRI encoding fusion models weights
            file_name = (f'weights_sub-{args.subject:02d}_'
                f'hemi-{hemi}_eeg_time-{t:03d}.npy')
            reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
                'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights',
                file_name), allow_pickle=True).item()

            # Instantiate the fusion regression model
            reg = LinearRegression()
            reg.coef_ = reg_param['coef_'][idx_v]
            reg.intercept_ = reg_param['intercept_'][idx_v]
            reg.n_features_in_ = reg_param['n_features_in_']

            # Generate the t-fMRI responses for the test images
            for pt in range(n_pseudo):
                tfmri_hemi[:,pt,:,t] = reg.predict(eeg_test[:,pt,:,t])
            del reg_param, reg

        # Store the t-fMRI responses
        if h == 0:
            tfmri[roi] = tfmri_hemi
        else:
            tfmri[roi] = np.append(tfmri[roi], tfmri_hemi, 2)
        del tfmri_hemi


# =============================================================================
# Compute the RSMs of each ROI at each time point
# =============================================================================
# There are multiple RSMs for each ROI and time point, computed by correlating
# the responses between different repeats: these different RSMs are later used
# to cross-validate the regressions used to compute the GC scores

# 6 RSMs in total, divided into 3 splits of 2 RSMs each, where each RMS is
# computed by correlating the responses of 2 repeats (the numbers below
# indicate the repeats used to compute each RSM)
rep_splits = [
    [[0, 1], [2, 3]], # Split 1: RSM 1 computed from repeats 0 and 1, RSM 2 computed from repeats 2 and 3
    [[0, 2], [1, 3]], # Split 2: RSM 1 computed from repeats 0 and 2, RSM 2 computed from repeats 1 and 3
    [[0, 3], [1, 2]] # Split 3: RSM 1 computed from repeats 0 and 3, RSM 2 computed from repeats 1 and 2
    ]

roi_rsms = {}
idx_tril = np.tril_indices(len(tfmri[args.rois[0]]), k=-1)
idx_triu = np.triu_indices(len(tfmri[args.rois[0]]), k=1)
for roi in tqdm(args.rois):

    roi_rsms[roi] = []

    for s, split in enumerate(rep_splits):

        rsms_split = []

        for r, rep in enumerate(split):

            # Get the responses for the two repetition splits
            X = copy(tfmri[roi][:,rep[0]])
            Y = copy(tfmri[roi][:,rep[1]])

            # Z-score across vertices
            X_mean = X.mean(axis=1, keepdims=True)
            Y_mean = Y.mean(axis=1, keepdims=True)
            X_std = X.std(axis=1, keepdims=True)
            Y_std = Y.std(axis=1, keepdims=True)
            X_z = (X - X_mean) / (X_std + 1e-8)
            Y_z = (Y - Y_mean) / (Y_std + 1e-8)

            # Reshape to (time, images, vertices)
            X_t = np.transpose(X_z, (2, 0, 1))  # (time, images_X, vertices)
            Y_t = np.transpose(Y_z, (2, 0, 1))  # (time, images_Y, vertices)

            # Cross-correlation via batch matmul
            rsm = np.matmul(X_t, Y_t.transpose(0, 2, 1)) / (X.shape[1])

            # Back to (images_X, images_Y, time)
            rsm = np.transpose(rsm, (1, 2, 0))

            # Store the upper triangle of the RSMs without the main diagonal # !!! Use both upper and lower triangle
            rsms_split.append(rsm[idx_triu])
            del X, Y, X_z, Y_z, X_t, Y_t, rsm

        # Store the RSMs
        roi_rsms[roi].append(rsms_split)
        del rsms_split


# =============================================================================
# Compute the Granger Causality
# =============================================================================
# Define the target times (starting from time 0)
idx_t_start_target = np.where(times == 0)[0][0]
times_target = times[idx_t_start_target:]

# Define the test times (always up to 100 ms prior to the target time)
offset = np.where(times == 0)[0][0] - np.where(times == -100)[0][0]

# Loop across ROIs
gc = {}
for roi_target in tqdm(args.rois):
    for roi_source in args.rois:
        if roi_target != roi_source:

            # Empty result array
            tot_splits = len(rep_splits) * len(rep_splits[0])
            gc_roi = np.zeros((tot_splits, offset, len(times_target)),
                dtype=np.float32)

            # Loop across time points of the target's present time point to be
            # predicted
            for tt_idx, tt in enumerate(range(idx_t_start_target, len(times))): # time target

                # Loop across time points of the target and source past time
                # points used for the prediction
                for ts_idx, ts in enumerate(range(tt-offset, tt)): # time source

                    # Loop across splits for cross-validation
                    idx_split = 0
                    for s in range(len(rep_splits)):
                        for r in range(len(rep_splits[s])):

                            # Get the train and test RSMs of the target and
                            # source ROIs
                            # Train
                            rsm_roi_target_train = np.reshape(
                                roi_rsms[roi_target][s][r][:,tt], (-1, 1))
                            rsm_roi_target_past_train = np.reshape(
                                roi_rsms[roi_target][s][r][:,ts], (-1, 1))
                            rsm_roi_source_past_train = np.reshape(
                                roi_rsms[roi_source][s][r][:,ts], (-1, 1))
                            # Test (use a different repeat for the test target
                            # than for the test predictors, to reduce the
                            # effect of noise correlations)
                            rsm_roi_target_test = np.reshape(
                                roi_rsms[roi_target][s][abs(r-1)][:,tt],
                                (-1, 1))
                            rsm_roi_target_past_test = np.reshape(
                                roi_rsms[roi_target][s][r][:,ts],
                                (-1, 1))
                            rsm_roi_source_past_test = np.reshape(
                                roi_rsms[roi_source][s][r][:,ts],
                                (-1, 1))

                            # Fit the linear regressions for the full and
                            # reduced models
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
                                rsm_roi_target_past_test,
                                rsm_roi_source_past_test, 1)) -
                                rsm_roi_target_test) ** 2)

                            # Adjust the MSE scores for the number of
                            # predictors in the models
                            n = len(rsm_roi_target_test)
                            p_reduced = rsm_roi_target_past_train.shape[1]
                            p_full = p_reduced + \
                                rsm_roi_source_past_train.shape[1]
                            u_reduced = u_reduced * (n - 1) / \
                                (n - p_reduced - 1)
                            u_full = u_full * (n - 1) / (n - p_full - 1)

                            # Compute the GC score as the log ratio of the
                            # unexplained variance of the reduced and full
                            # models
                            gc_roi[idx_split,ts_idx,tt_idx] = \
                                np.log(u_reduced / u_full)
                            idx_split += 1

                            # Remove unused variables
                            del rsm_roi_target_train, \
                                rsm_roi_target_past_train, \
                                rsm_roi_source_past_train, \
                                rsm_roi_target_test, \
                                rsm_roi_target_past_test, \
                                rsm_roi_source_past_test, reg_reduced, \
                                reg_full, u_reduced, u_full

            # Store the GC results in a dictionary
            gc[f'{roi_source}_to_{roi_target}'] = np.mean(gc_roi, 0)
            del gc_roi


# =============================================================================
# Save the results
# =============================================================================
results = {
    'gc': gc,
    'times': times,
    'times_target': times_target
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'granger_causality',
    'gc_scores')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'gc_sub-{args.fmri_subject:02d}_regression-{args.regression}.npy')

np.save(os.path.join(save_dir, file_name), results)