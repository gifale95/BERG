"""Apply Granger Causality (GC) between the t-fMRI responses of two ROIs.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1 to 8.
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
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--rois', default=['V1', 'hV4', 'ventral'], type=list)
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
# Load the RSMs for each ROI
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

data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'granger_causality', 'roi_rsms')

roi_rsms = {}

for roi in args.rois:

    file_name = (f'rsms_sub-{args.subject:02d}_roi-{roi}.npy')
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()

    roi_rsms[roi] = data['roi_rsms']
    times = data['times']
    del data


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
# Compute RSA between neural time points
# =============================================================================
rsa_times = {}

for roi in tqdm(args.rois):

    # Empty result array
    rsa_roi = np.zeros((len(rep_splits), len(times), len(times)),
        dtype=np.float32)

    # Loop across time points
    for t1 in range(len(times)):
        for t2 in range(len(times)):

            # Loop across splits for cross-validation
            idx_split = 0
            for s in range(len(rep_splits)):

                # Correlate the RSMs of two time points
                rsa_roi[idx_split,t1,t2] = pearsonr(roi_rsms[roi][s][0][:,t1],
                    roi_rsms[roi][s][1][:,t2])[0]

    # Store the results
    rsa_times[roi] = np.mean(rsa_roi, 0)
    del rsa_roi


# =============================================================================
# RSA between neural responses and layerwise AlexNet activations
# =============================================================================
idx_triu = np.triu_indices(len(roi_rsms[args.rois[0]][0][0]), k=1)

rsa_alexnet = {}

for roi in tqdm(args.rois):

    # Average the neural RSM across splits, and convert them to RDMs with
    # "1 - RSM" (to match the AlexNet representations in RDM format)
    rsm = []
    for s in range(len(rep_splits)):
        for r in range(len(rep_splits[s])):
            rsm.append(roi_rsms[roi][s][r])
    rsm = np.mean(rsm, 0)
    rdm = 1 - rsm

    # Load the DNN layerwise RDMs
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'dnn_layerwise_modeling', 'dnn_rdms', 'dnn_rdms_alexnet.npy')
    dnn_rdms = np.load(data_dir, allow_pickle=True).item()

    # Loop across DNN layers
    for key, val in dnn_rdms.items():

        val = val[idx_triu]

        rsa_alexnet[(roi, key)] = np.zeros((len(times)), dtype=np.float32)

        # Loop across neural time points
        for t in range(len(times)):

            # Correlate neural and AlexNet RDMs
            rsa_alexnet[(roi, key)][t] = pearsonr(val, rdm[:,t])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'gc': gc,
    'times': times,
    'times_target': times_target,
    'rsa_times': rsa_times,
    'rsa_alexnet': rsa_alexnet
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'granger_causality',
    'gc_scores')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'gc_sub-{args.subject:02d}_regression-{args.regression}.npy')

np.save(os.path.join(save_dir, file_name), results)