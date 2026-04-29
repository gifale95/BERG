"""Apply Granger Causality (GC) between the t-fMRI responses of two ROIs, using
linear encoding.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1 to 8.
rois : list
    List containing the ROIs used for the Granger Causality analysis. All ROIs
    are tested in a pairwise fashion.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'even' (even trials), and 'odd' (odd trials).
regression : str
    The type of regression to use for computing Granger Causality. If 'linear',
    use linear regression. If 'ridge', use RidgeCV regression.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--rois', default=['V1', 'hV4', 'ventral'], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--eeg_train_trials', default='even', type=str)
parser.add_argument('--regression', default='ridge', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Granger Causality <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Get the fMRI ROI indices
# =============================================================================
# Load the fMRI metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.subject
    )

idx_v = {}

# Loop across ROIs
for roi in args.rois:

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
        idx_v[(roi, hemi)] = np.intersect1d(idx_roi, idx_ncsnr)


# =============================================================================
# EEG time point selection
# =============================================================================
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

# Limit the analysis to the first 400 milliseconds to reduce compute time
t_end = np.where(times == 399)[0][0]
times = times[:t_end+1]


# =============================================================================
# Load the train and test EEG responses
# =============================================================================
# Load the EEG train responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_data')
file_name_train = (f'eeg_train_sub-{args.subject:02d}_'
    f'trial_avg-{args.eeg_train_trials}.npy')
eeg_train = np.load(os.path.join(data_dir, file_name_train),
    allow_pickle=True).item()['eeg_train'][:,:,:t_end+1].astype(np.float32)

# Load the EEG test responses, and average them across repeats
file_name_test = f'eeg_test_sub-{args.subject:02d}.npy'
eeg_test = np.mean(np.load(os.path.join(data_dir, file_name_test),
    allow_pickle=True).item()['eeg_test'][:,:,:,:t_end+1], 1).astype(np.float32)


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Empty t-fMRI data dictionary
tfmri_train = {}
tfmri_test = {}

# Loop across EEG time points
for t in tqdm(range(eeg_train.shape[2])):

    # Loop across ROIs
    for roi in args.rois:

        # Loop across hemisphers
        for h, hemi in enumerate(args.hemispheres):

            # Load the EEG-fMRI encoding fusion models weights (if using the
            # even EEG training trials, then load the models trained on the odd
            # EEG training trials, and vice versa, to account for the fusion
            # models using the noise in the EEG responses to predict fMRI)
            if args.eeg_train_trials == 'even':
                weights_eeg_train_trials = 'odd'
            elif args.eeg_train_trials == 'odd':
                weights_eeg_train_trials = 'even'
            file_name = (f'weights_sub-{args.subject:02d}_hemi-{hemi}_'
                f'eeg_train_trials-{weights_eeg_train_trials}_eeg_time-{t:03d}.npy')
            reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
                'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights',
                file_name), allow_pickle=True).item()

            # Instantiate the fusion regression model
            reg = LinearRegression()
            reg.coef_ = reg_param['coef_'][idx_v[(roi, hemi)]]
            reg.intercept_ = reg_param['intercept_'][idx_v[(roi, hemi)]]
            reg.n_features_in_ = reg_param['n_features_in_']

            # Generate the t-fMRI responses
            tfmri_hemi_train = np.expand_dims(reg.predict(eeg_train[:,:,t]),
                2).astype(dtype=np.float32)
            tfmri_hemi_test = np.expand_dims(reg.predict(eeg_test[:,:,t]),
                2).astype(dtype=np.float32)
            del reg_param, reg

            # Append the t-fMRI responses across hemispheres
            if h == 0:
                tfmri_time_train = tfmri_hemi_train
                tfmri_time_test = tfmri_hemi_test
            else:
                tfmri_time_train = np.append(tfmri_time_train,
                    tfmri_hemi_train, 1)
                tfmri_time_test = np.append(tfmri_time_test, tfmri_hemi_test, 1)
            del tfmri_hemi_train, tfmri_hemi_test

        # Store the t-fMRI responses
        if t == 0:
            tfmri_train[roi] = tfmri_time_train
            tfmri_test[roi] = tfmri_time_test
        else:
            tfmri_train[roi] = np.append(tfmri_train[roi], tfmri_time_train, 2)
            tfmri_test[roi] = np.append(tfmri_test[roi], tfmri_time_test, 2)
        del tfmri_time_train, tfmri_time_test

# Delete the EEG responses
del eeg_train, eeg_test


# =============================================================================
# Compute the Granger Causality
# =============================================================================
# Define the target times (starting from time 0)
idx_t_start_target = np.where(times == 0)[0][0]
times_target = times[idx_t_start_target:]

# Define the test times (always up to 75 ms prior to the target time)
offset = np.where(times == 0)[0][0] - np.where(times == -75)[0][0]

# Loop across ROIs
gc = {}
for roi_target in tqdm(args.rois):
    for roi_source in args.rois:
        if roi_target != roi_source:

            # Empty result array
            gc_roi = np.zeros((offset, len(times_target)), dtype=np.float32)

            # Loop across time points of the target's present time point to be
            # predicted
            for tt_idx, tt in enumerate(range(idx_t_start_target, len(times))): # time target

                # Loop across time points of the target and source past time
                # points used for the prediction
                for ts_idx, ts in enumerate(range(tt-offset, tt)): # time source

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
                    reg_reduced.fit(tfmri_train[roi_target][:,:,ts],
                        tfmri_train[roi_target][:,:,tt])
                    reg_full.fit(np.append(tfmri_train[roi_target][:,:,ts],
                        tfmri_train[roi_source][:,:,ts], 1),
                        tfmri_train[roi_target][:,:,tt])

                    # Compute the unexplained variance for the full and
                    # reduced models (MSE)
                    u_reduced = np.mean((
                        reg_reduced.predict(tfmri_test[roi_target][:,:,ts]) -
                        tfmri_test[roi_target][:,:,tt]) ** 2)
                    u_full = np.mean((reg_full.predict(np.append(
                        tfmri_test[roi_target][:,:,ts],
                        tfmri_test[roi_source][:,:,ts], 1)) -
                        tfmri_test[roi_target][:,:,tt]) ** 2)

                    # Adjust the MSE scores for the number of
                    # predictors in the models
                    n = len(tfmri_test[roi_target])
                    p_reduced = tfmri_test[roi_target].shape[1]
                    p_full = p_reduced + tfmri_test[roi_source].shape[1]
                    u_reduced = u_reduced * (n - 1) / \
                        (n - p_reduced - 1)
                    u_full = u_full * (n - 1) / (n - p_full - 1)

                    # Compute the GC score as the log ratio of the
                    # unexplained variance of the reduced and full
                    # models
                    gc_roi[ts_idx,tt_idx] = np.log(u_reduced / u_full)

                    # Remove unused variables
                    del reg_reduced, reg_full, u_reduced, u_full

            # Store the GC results in a dictionary
            gc[f'{roi_source}_to_{roi_target}'] = gc_roi
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
    'invivo_nsd_eeg_fmri_control', 'granger_causality', 'encoding',
    'gc_scores')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'gc_sub-{args.subject:02d}_eeg_train_trials-'
    f'{args._eeg_train_trials}_regression-{args.regression}.npy')

np.save(os.path.join(save_dir, file_name), results)