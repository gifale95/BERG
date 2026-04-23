"""Compute Granger Causality, using RSA, on in silico neural responses found
through neural control.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico
    responses.
subject : str
    The subject identifier for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
roi_target: str
    The target ROI for computing Granger Causality. Valid values are "V1",
    "V4", and "IT".
roi_source: str
    The source ROI for computing Granger Causality. Valid values are "V1",
    "V4", and "IT".
rois_neural_control: str
    If 'single', use images from neural control applied to only the source ROI.
    If 'both', use images from neural control applied to both the source and
    target ROIs.
objective: str
    If 'max', use images that will maximize the Granger Causality score.
    If 'min', use images that will minimize the Granger Causality score.
    If 'baseline', use image that will keep the Granger Causality score at
    baseline level.
cv: int
    If 1, cross-validate the controlling images across the two monkyes.
    If 0, do not cross-validate.
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
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from berg import BERG
from copy import copy
import torchvision
from torchvision import transforms as trn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='F', type=str) # 'N' 'F'
parser.add_argument('--roi_target', default='V1', type=str)
parser.add_argument('--roi_source', default='V4', type=str)
parser.add_argument('--rois_neural_control', default='single', type=str) # 'single' 'both'
parser.add_argument('--objective', default='max', type=str) # 'max' 'min' 'baseline'
parser.add_argument('--cv', default=1, type=int) # '0' '1'
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--time_window_ms', default=100, type=int)
parser.add_argument('--offset_ms', default=20, type=int)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
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
# Get the image number from the neural control analysis
# =============================================================================
# Define the subject used to find the images
if args.cv == 0:
    sub_img = args.subject
else:
    sub_img = 'F' if args.subject == 'N' else 'N'

# Load the image numbers
if args.objective == 'baseline':

    data_dir = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'quantitative_results', args.encoding_model,
        f'sub-{sub_img}_roi-{args.roi_source}_baseline.npy')
    data = np.load(data_dir, allow_pickle=True).item()
    img_num = data['img_baseline']

else:

    data_dir = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'stats', args.encoding_model)

    if args.rois_neural_control == 'single':
        if args.objective == 'max':
            file_name = f'sub-{sub_img}_roi-{args.roi_source}_early-drive_late-drive.npy'
        elif args.objective == 'min':
            file_name = f'sub-{sub_img}_roi-{args.roi_source}_early-suppress_late-suppress.npy'

    if args.rois_neural_control == 'both':
        if args.objective == 'max':
            file_name = (f'sub-{sub_img}_roi_1-{args.roi_target}_early-drive_late-drive'
                f'_roi_2-{args.roi_source}_early-drive_late-drive.npy')
        elif args.objective == 'min':
            file_name = (f'sub-{sub_img}_roi_1-{args.roi_target}_early-drive_late-drive'
                f'_roi_2-{args.roi_source}_early-suppress_late-suppress.npy')

    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    img_num = data['img_control']


# =============================================================================
# Load the images
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224)),
    trn.Lambda(lambda img: np.transpose(img, (2, 0, 1))) # HWC to CHW
])

# Access the ILSVRC-2012 train split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train',
    transform=transform)

# Load the images
images = []
for img in tqdm(img_num):
    images.append(imageset.__getitem__(img)[0])
images = np.array(images)


# =============================================================================
# Load BERG's metadata
# =============================================================================
# Load BERG's metadata
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
)


# =============================================================================
# Generate the in silico responses
# =============================================================================
# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    train_splits='single'
    )

# Generate the in silico neural responses
# Shape: (Images, Repeats, Channels, Times)
resp = berg.encode(model, images)
del model, images


# =============================================================================
# Divide the neural responses based on ROIs
# =============================================================================
# Retain channels based on their NCSNR score averaged across the time window
# around peak activity of the chosen ROI (use the same time window as in the
# TVSD paper)
ncsnr = metadata['encoding_model']['ncsnr']
times = metadata['utah_array']['times']
roi_assignments = metadata['roi']['roi_assignments']
peaks = {
    'V1': (25, 125),
    'V4': (50, 150),
    'IT': (75, 175)
}
roi_num = {
    'V1': 0,
    'V4': 1,
    'IT': 2
}

# Loop across ROIs
roi_resp = {}
for r, roi in enumerate([args.roi_source, args.roi_target]):

    # Get the channels assigned to the ROI
    idx_roi = np.where(roi_assignments == roi_num[roi])[0]

    # Get the NCSNR scores for those channels, averaged across the time window
    # around peak activity of the chosen
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
idx_triu = np.triu_indices(len(roi_resp[args.roi_source]), k=1)
for roi in tqdm(roi_resp.keys()):

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

            # Store the upper triangle of the RSMs without the main diagonal
            rsms_split.append(rsm[idx_triu])
            del X, Y, X_z, Y_z, X_t, Y_t, rsm

        # Store the RSMs
        roi_rsms[roi].append(rsms_split)
        del rsms_split


# =============================================================================
# Compute the Granger Causality (RSMs averaged over past times)
# =============================================================================
# Get the time indices
t_min = times[0] + args.time_window_ms + args.offset_ms
idx_t_start = np.where(times == t_min)[0][0]

# Empty result list
gc = []

# Loop across time points
for t in tqdm(range(idx_t_start, len(times))):

    gc_roi_t = []

    # Loop across splits for cross-validation
    for s in range(len(rep_splits)):
        for r in range(len(rep_splits[s])):

            # Get the train and test RSMs of the target and source
            # ROIs, and average them across time points
            # Train
            rsm_roi_target_train = np.reshape(
                roi_rsms[args.roi_target][s][r][:,t], (-1, 1))
            rsm_roi_target_past_train = np.mean(
                roi_rsms[args.roi_target][s][r]\
                [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
                1, keepdims=True)
            rsm_roi_source_past_train = np.mean(
                roi_rsms[args.roi_source][s][r]\
                [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
                1, keepdims=True)
            # Test (use a different repeat for the test target
            # than for the test predictors, to reduce the effect of
            # noise correlations)
            rsm_roi_target_test = np.reshape(
                roi_rsms[args.roi_target][s][abs(r-1)][:,t], (-1, 1))
            rsm_roi_target_past_test = np.mean(
                roi_rsms[args.roi_target][s][r]\
                [:,t-args.time_window_ms-args.offset_ms:t-args.offset_ms],
                1, keepdims=True)
            rsm_roi_source_past_test = np.mean(
                roi_rsms[args.roi_source][s][r]\
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
    gc.append(np.mean(gc_roi_t))
    del gc_roi_t

# Format the GC results
gc = np.array(gc)


# =============================================================================
# Save the results
# =============================================================================
data = {
    'gc': gc,
    'times': times,
    'idx_t_start': idx_t_start
}

save_dir = os.path.join(args.berg_dir, 'neural_control',
    'granger_causality_neural_control', 'granger_causality',
    args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = (f'gc_sub-{args.subject}_roi_target-{args.roi_target}_'
    f'roi_source-{args.roi_source}_rois_neural_control-'
    f'{args.rois_neural_control}_objective-{args.objective}_cv-{args.cv}_'
    f'time_window_ms-{args.time_window_ms:03d}_offset_ms-'
    f'{args.offset_ms:03d}_regression-{args.regression}.npy')

np.save(os.path.join(save_dir, file_name), data)

# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(times[idx_t_start:], gc)
# plt.xlabel('Time (ms)')
# plt.ylabel('Granger Causality')
# plt.legend()
# plt.show()