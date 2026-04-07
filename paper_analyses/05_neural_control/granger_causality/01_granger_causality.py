"""Compute Granger Causality, using RSA, on the TVSD in vivo or in silico
responses for the test images.

Parameters
----------
data_type : str
    If 'invivo', compute Granger Causality for in vivo responses.
    If 'insilico', compute it for in silico responses.
encoding_model : str
    The name of BERG's encoding model used for generating the in silico
    responses.
subject : str
    The subject identifier for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
rois: list
    List of ROIs between which to compute Granger Causality.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    channel selection.
time_window_ms : int
    Time window in milliseconds for computing Granger Causality.
offset_ms : int
    Offset in milliseconds for the time window.
regression : str
    The type of regression to use for computing Granger Causality. If 'linear',
    use linear regression. If 'ridge', use RidgeCV regression.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from berg import BERG
from copy import copy
from PIL import Image
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='insilico')
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='F', type=str)
parser.add_argument('--rois', default=['V1', 'V4'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--time_window_ms', default=100, type=int)
parser.add_argument('--offset_ms', default=20, type=int)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
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
# Load the metadata
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject)


# =============================================================================
# Load the in vivo responses for the test images
# =============================================================================
# The in vivo data has been prepared using this code:
# https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-tvsd

if args.data_type == 'invivo':

    data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-tvsd', f'tvsd_monkey{args.subject}_split-test.h5')
    data = h5py.File(data_dir, 'r')['neural_data']

    # Create 4 pseudo-trials by averaging the responses across repeats
    test_img_ids = metadata['encoding_model']['test_img_ids']
    unique_img_ids = np.unique(test_img_ids)
    resp = []
    for img_id in tqdm(unique_img_ids):
        idx = np.where(test_img_ids == img_id)[0]
        np.random.shuffle(idx)
        resp_img = []
        pseudotrials = 4
        n_reps = len(idx) // pseudotrials
        for p in range(pseudotrials):
            idx_start = p * n_reps
            idx_end = (p + 1) * n_reps
            idx_pseudo = idx[idx_start:idx_end]
            idx_pseudo.sort()
            resp_img.append(np.mean(data[idx_pseudo], 0))
        resp.append(np.array(resp_img))
        del resp_img
    # Shape: (Images, Repeats, Channels, Times)
    resp = np.array(resp)


# =============================================================================
# Generate the in silico responses for the test images
# =============================================================================
elif args.data_type == 'insilico':

    # Get the test image file names
    test_img_ids = metadata['encoding_model']['test_img_ids']
    unique_img_ids = np.unique(test_img_ids)
    test_stimuli = metadata['encoding_model']['test_stimuli']

    # Loop across test image files
    images = []
    for img_id in tqdm(unique_img_ids):

        # Find the corresponding image file name
        idx = np.where(test_img_ids == img_id)[0][0]
        file = test_stimuli[idx]

        # Find correct subfolder
        img_path = None
        for root, _, files in os.walk(os.path.join(args.things_dir)):
            if file in files:
                img_path = os.path.join(root, file)
                break
    
        # Load and transform the image
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
        img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
        images.append(img)
    images = np.array(images)

    # Load the encoding model
    model = berg.get_encoding_model(
        args.encoding_model,
        subject=args.subject,
        train_splits='single'
        )

    # Generate the in silico neural responses
    # Shape: (Images, Repeats, Channels, Times)
    resp = berg.encode(model, images)


# =============================================================================
# Divide the neural responses based on ROIs
# =============================================================================
# Retain channels based on their NCSNR score averaged across the time window
# around peak activity (use the same time window as in the TVSD paper)
ncsnr = metadata['encoding_model']['ncsnr']
times = metadata['utah_array']['times']
roi_assignments = metadata['roi']['roi_assignments']
peaks = {
    'V1': (25, 125),
    'V4': (50, 150),
    'IT': (75, 175)
}

# Loop across ROIs
roi_resp = {}
for r, roi in enumerate(args.rois):

    # Get the channels assigned to the ROI
    if roi == 'V1':
        roi_num = 0
    elif roi == 'V4':
        roi_num = 1
    elif roi == 'IT':
        roi_num = 2
    idx_roi = np.where(roi_assignments == roi_num)[0]

    # Get the NCSNR scores for those channels, averaged across the time window
    # around peak activity
    t_min = np.where(times == peaks[roi][0])[0][0]
    t_max = np.where(times == peaks[roi][1])[0][0]
    ncsnr_roi = np.mean(ncsnr[idx_roi][:,t_min:t_max+1], 1)

    # Retain channels with NCSNR above the threshold
    idx_ncsnr = ncsnr_roi >= args.ncsnr_threshold
    roi_resp[roi] = resp[:,:,idx_roi[idx_ncsnr]]


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
idx_triu = np.triu_indices(len(roi_resp[args.rois[0]]), k=1)
for roi in tqdm(args.rois):

    roi_rsms[roi] = []

    for s, split in enumerate(rep_splits):

        rsms_split = []

        for r, rep in enumerate(split):

            # Get the responses for the two repetition splits
            X = copy(roi_resp[roi][:,rep[0]])
            Y = copy(roi_resp[roi][:,rep[1]])

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

            # Store the upper and lower triangle of the RSMs without the main
            # diagonal
            rsms_split.append(rsm[idx_triu])
            del X, Y, X_z, Y_z, X_t, Y_t, rsm

        # Store the RSMs
        roi_rsms[roi].append(rsms_split)
        del rsms_split


# =============================================================================
# Compute the Granger Causality (RSM averaged over past times)
# =============================================================================
# Get the time indices
t_min = times[0] + args.time_window_ms + args.offset_ms
idx_t_start = np.where(times == t_min)[0][0]

# Loop across ROIs
gc = {}
for roi_target in args.rois:
    for roi_source in args.rois:
        if roi_target != roi_source:

            # Empty result list
            gc_roi = []

            # Loop across time points
            for t in tqdm(range(idx_t_start, len(times))):

                gc_roi_t = []

                # Loop across splits for cross-validation
                for s in range(len(rep_splits)):
                    for r in range(len(rep_splits[s])):

                        # Get the train and test RSMs of the target and source
                        # ROIs
                        # Train
                        rsm_roi_target_train = np.reshape(
                            roi_rsms[roi_target][s][r][:,t], (-1, 1))
                        rsm_roi_target_past_train = np.mean(
                            roi_rsms[roi_target][s][r]\
                            [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
                            1, keepdims=True)
                        rsm_roi_source_past_train = np.mean(
                            roi_rsms[roi_source][s][r]\
                            [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
                            1, keepdims=True)
                        # Test (use a different repeat for the test target
                        # than for the test predictors, to reduce the effect of
                        # noise correlations)
                        rsm_roi_target_test = np.reshape(
                            roi_rsms[roi_target][s][abs(r-1)][:,t], (-1, 1))
                        rsm_roi_target_past_test = np.mean(
                            roi_rsms[roi_target][s][r]\
                            [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
                            1, keepdims=True)
                        rsm_roi_source_past_test = np.mean(
                            roi_rsms[roi_source][s][r]\
                            [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
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
            gc[f'{roi_source}_to_{roi_target}'] = np.transpose(np.array(
                gc_roi))
            del gc_roi


# =============================================================================
# Save the results
# =============================================================================
data = {
    'gc': gc,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_control', 'granger_causality',
    'granger_causality', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = (f'gc_data_type-{args.data_type}_sub-{args.subject}_'
            f'time_window_ms-{args.time_window_ms:03d}_'
            f'offset_ms-{args.offset_ms:03d}_regression-{args.regression}.npy')

np.save(os.path.join(save_dir, file_name), data)


# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(times[idx_t_start:], np.mean(gc['V1_to_V4'], 0), label='V1 to V4')
# plt.plot(times[idx_t_start:], np.mean(gc['V4_to_V1'], 0), label='V4 to V1')
# plt.xlabel('Time (ms)')
# plt.ylabel('Granger Causality')
# plt.legend()
# plt.show()