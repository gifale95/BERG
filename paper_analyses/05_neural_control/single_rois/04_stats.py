"""Apply neural control to find images that drive or suppress the in silico
monkey electrophysiology responses. The controlling images are then
cross-validated across subjects.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subjects : list
    List of subject identifiers for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
roi: str
    ROI used. Valid values are "V1", "V4", and "IT".
control: str
    If "early-drive_late-drive", then both the early (25-100ms) and late
    (101-200ms) part of the epoch are driven.
    If "early-suppress_late-suppress", then both the early and late part of the
    epoch are suppressed.
    If "early-drive_late-suppress", then the early part of the epoch is driven
    while the late part is suppressed.
    If "early-suppress_late-drive", then the early part of the epoch is
    suppressed while the late part is driven.
n_images: int
    Number of retained controlling or baseline images.
n_iter : int
    Amount of iterations to generate the bootstrap distributions.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import h5py
import numpy as np
import random
import torchvision
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy
from torchvision import transforms as trn
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--control', default='early-drive_late-drive', type=str)
parser.add_argument('--n_images', default=50, type=int)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Neural control <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)

- Add condition where early part of the epoch is suppressed and the late part
    is driven, and vice versa. => early/late time points (1, 100) (101, 200)


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
# Load the in silico responses
data_dir = os.path.join(args.berg_dir, 'neural_control', 'insilico_responses',
    args.encoding_model)
insilico_data = []
metadata = []
for sub in args.subjects:
    file_name = f'insilico_responses_sub-{sub}_roi-{args.roi}.npy'
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    insilico_data.append(data['responses'])
    metadata.append(data['metadata'])
insilico_data = np.array(insilico_data)

# Average the in silico neural responses across the time window around peak
# activity (as in the TVSD paper)
# times = metadata[0]['utah_array']['times']
# peaks = {
#     'V1': (25, 125),
#     'V4': (50, 150),
#     'IT': (75, 175)
# }
# t_min = np.where(times == peaks[args.roi][0])[0][0]
# t_max = np.where(times == peaks[args.roi][1])[0][0]

# Average the in silico neural responses across early parts of the epoch
# (25-100ms), late parts of the epoch (101-200ms), or the entire epoch
# (25-200ms)
times = metadata[0]['utah_array']['times']
t_min_early = np.where(times == 25)[0][0]
t_max_early = np.where(times == 100)[0][0]
t_min_late = np.where(times == 101)[0][0]
t_max_late = np.where(times == 199)[0][0]
insilico_data_early = np.mean(insilico_data[:,:,t_min_early:t_max_early], 2)
insilico_data_late = np.mean(insilico_data[:,:,t_min_late:t_max_late], 2)
insilico_data_full = np.mean(insilico_data[:,:,t_min_early:t_max_late], 2)


# =============================================================================
# Load the baseline results
# =============================================================================
# Load the baseline results
data_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model,
    f'roi-{args.roi}_baseline.npy')
baseline_results = np.load(data_dir, allow_pickle=True).item()

# Average the baseline responses across early parts of the epoch
# (25-100ms), late parts of the epoch (101-200ms), or the entire epoch
# (25-200ms)
baseline_data = baseline_results['baseline_data']
baseline_data_early = np.mean(baseline_data[:,:,t_min_early:t_max_early], 2)
baseline_data_late = np.mean(baseline_data[:,:,t_min_late:t_max_late], 2)
baseline_data_full = np.mean(baseline_data[:,:,t_min_early:t_max_late], 2)


# =============================================================================
# Neural control # !!!
# =============================================================================
# Response score margin used to constrain the selection of the controlling
# images
margin = 0.04

# Select the top N images that drive or suppress both early and late part of
# the epoch
if args.control in ['early-drive_late-drive', 'early-suppress_late-suppress']:

    response_sum = insilico_data_early + insilico_data_late

    # Select the top N images that drive both early and late part of the epoch
    if args.control == 'early-drive_late-drive':
        img_control = np.argsort(response_sum, 1)[::-1].astype(np.float32)
        # Ignore images conditions with univariate responses below the baseline
        # scores (plus a margin)
        idx_bad_early = np.where(
            insilico_data_early[:,img_control.astype(np.int32)] < \
            baseline_data_early[0]+margin)[0]
        idx_bad_late = np.where(resp_roi_2[high_1_high_2.astype(np.int32)] < \
            baseline_scores[1]+margin)[0]
        img_control[idx_bad_early] = np.nan
        img_control[idx_bad_late] = np.nan

# 2nd ranking: images with low univariate responses for both ROIs
low_1_low_2 = np.argsort(roi_sum).astype(np.float32)
# Ignore images conditions with univariate responses above the baseline
# scores (plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[low_1_low_2.astype(np.int32)] > \
	baseline_scores[0]-margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[low_1_low_2.astype(np.int32)] > \
	baseline_scores[1]-margin)[0]
low_1_low_2[idx_bad_roi_1] = np.nan
low_1_low_2[idx_bad_roi_2] = np.nan

# Select the top N images that disentangle the in silico univariate fMRI
# responses of the two ROIs (i.e., that lead one ROI having high
# responses and the other ROI low responses, or vice versa).
# 3rd ranking: images with high univariate responses for ROI 1 and low
# univariate responses for ROI 2
roi_diff = resp_roi_1 - resp_roi_2
high_1_low_2 = np.argsort(roi_diff)[::-1].astype(np.float32)
# Ignore images conditions with univariate responses below (ROI 1) or above
# (ROI 2) the baseline scores (plus/minus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[high_1_low_2.astype(np.int32)] < \
	baseline_scores[0]+margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[high_1_low_2.astype(np.int32)] > \
	baseline_scores[1]-margin)[0]
high_1_low_2[idx_bad_roi_1] = np.nan
high_1_low_2[idx_bad_roi_2] = np.nan
# 4th ranking: images with low univariate responses for ROI 1 and high
# univariate responses for ROI 2
low_1_high_2 = np.argsort(roi_diff).astype(np.float32)
# Ignore images conditions with univariate responses above (ROI 1) or below
# (ROI 2) the baseline scores (minus/plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[low_1_high_2.astype(np.int32)] > \
	baseline_scores[0]-margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[low_1_high_2.astype(np.int32)] < \
	baseline_scores[1]+margin)[0]
low_1_high_2[idx_bad_roi_1] = np.nan
low_1_high_2[idx_bad_roi_2] = np.nan


































# Find the images controlling the neural responses
if args.control == 'drive':
    img_control = np.argsort(insilico_data, axis=1)[:,::-1]
elif args.control == 'suppress':
    img_control = np.argsort(insilico_data, axis=1)
img_control = img_control[:,:args.n_images]

# Cross-validate the controlling images across subjects
control_data = []
cv_control_data = []
for s in range(len(args.subjects)):
    s_cv = np.delete((0, 1), s)[0]
    control_data.append(insilico_data[s,img_control[s]])
    cv_control_data.append(insilico_data[s_cv,img_control[s]])
control_data = np.array(control_data)
cv_control_data = np.array(cv_control_data)


# =============================================================================
# Compute the confidence intervals
# =============================================================================
dist = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_cv = np.zeros((args.n_iter, len(args.subjects), len(times)))

for i in tqdm(range(args.n_iter), leave=False):
    idx = resample(np.arange(args.n_images))
    dist[i] = np.mean(control_data[:,idx], axis=1)
    dist_cv[i] = np.mean(cv_control_data[:,idx], axis=1)

ci_low_control_data = np.percentile(dist, 2.5, axis=0)
ci_high_control_data = np.percentile(dist, 97.5, axis=0)
ci_low_cv_control_data = np.percentile(dist_cv, 2.5, axis=0)
ci_high_cv_control_data = np.percentile(dist_cv, 97.5, axis=0)


# =============================================================================
# Compute the significance of the CV neural control scores # !!!
# =============================================================================
# Empty p-value lists
p_val = []
p_val_bh = []
p_val_bonf = []

# Loop across subjects
for s in range(len(args.subjects)):

    # Compute the within-subject p-values
    s_cv = np.delete((0, 1), s)[0]
    if args.control == 'drive':
        idx = np.sum(
            null_distribution[:,s] > np.mean(cv_control_data[s_cv], 0), 0)
    elif args.control == 'suppress':
        idx = np.sum(
            null_distribution[:,s] < np.mean(cv_control_data[s_cv], 0), 0)
    p_val_sub = (idx + 1) / (args.n_iter + 1) # Add 1 to avoid p-values of 0
    p_val.append(p_val_sub)

    # Correct for multiple comparisons
    p_val_bh.append(multipletests(p_val_sub, 0.05, 'fdr_bh')[1])
    p_val_bonf.append(multipletests(p_val_sub, 0.05, 'bonferroni')[1])

# Format to numpy arrays
p_val = np.array(p_val)
p_val_bh = np.array(p_val_bh)
p_val_bonf = np.array(p_val_bonf)


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
results = {
    'img_control': img_control,
    'control_data': control_data,
    'ci_low_control_data': ci_low_control_data,
    'ci_high_control_data': ci_high_control_data,
    'cv_control_data': cv_control_data,
    'ci_low_cv_control_data': ci_low_cv_control_data,
    'ci_high_cv_control_data': ci_high_cv_control_data,
    'p_val': p_val,
    'p_val_bh': p_val_bh,
    'p_val_bonf': p_val_bonf
}

save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'roi-{args.roi}_{args.control}.npy'

np.save(os.path.join(save_dir, file_name), results)


# =============================================================================
# Save the controlling images
# =============================================================================
# Access the ILSVRC-2012 train split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train')

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Save directory
    save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
        'controlling_images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'{args.control}')
    os.makedirs(save_dir, exist_ok=True)

    # Loop across images
    images = []
    for i in range(args.n_images):

        # Get and preprocess the controlling images
        img, _ = imageset.__getitem__(img_control[s,i])
        min_size = min(img.size)
        transform = trn.Compose([
            trn.CenterCrop(min_size),
            trn.Resize((425,425))
            ])
        img = transform(img)
        images.append(np.array(img))

        # Save the controlling and baseline images as .png files
        file_name = f'{args.control}_img-{i:03}.png'
        img.save(os.path.join(save_dir, file_name))

    # Save the controlling and baseline images as h5py files
    with h5py.File(os.path.join(save_dir, 'controlling_images.h5'), 'w') as f:
        f.create_dataset('images', data=np.array(images))