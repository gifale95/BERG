"""Apply neural control to find images that drive or suppress the in silico
monkey electrophysiology responses.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : int
    Subject identifier for the monkey encoding model. Since the used encoding
    models are trained on the TVSD data, valid subject identifiers are "N" and
    "F".
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
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
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


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'insilico_responses',
args.encoding_model)
file_name = f'insilico_responses_sub-{args.subject}_roi-{args.roi}.npy'

data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
insilico_resp = data['responses']
metadata = data['metadata']

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
times = metadata['utah_array']['times']
t_min_early = np.where(times == 25)[0][0]
t_max_early = np.where(times == 100)[0][0]
t_min_late = np.where(times == 101)[0][0]
t_max_late = np.where(times == 199)[0][0]
insilico_resp_early = np.mean(insilico_resp[:,t_min_early:t_max_early], 1)
insilico_resp_late = np.mean(insilico_resp[:,t_min_late:t_max_late], 1)
insilico_resp_full = np.mean(insilico_resp[:,t_min_early:t_max_late], 1)


# =============================================================================
# Load the baseline results
# =============================================================================
# Load the baseline results
data_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model,
    f'sub-{args.subject}_roi-{args.roi}_baseline.npy')
baseline_results = np.load(data_dir, allow_pickle=True).item()

# Average the baseline responses across early parts of the epoch
# (25-100ms), late parts of the epoch (101-200ms), or the entire epoch
# (25-200ms)
baseline_resp = baseline_results['baseline_resp']
baseline_score_early = np.mean(baseline_resp[:,t_min_early:t_max_early], 1)
baseline_score_late = np.mean(baseline_resp[:,t_min_late:t_max_late], 1)
baseline_score_full = np.mean(baseline_resp[:,t_min_early:t_max_late], 1)


# =============================================================================
# Neural control # !!!
# =============================================================================
# Response score margin used to constrain the selection of the controlling
# images
margin = 0.04

# Select the top N images that drive or suppress both early and late part of
# the epoch
if args.control in ['early-drive_late-drive', 'early-suppress_late-suppress']:

    response_sum = insilico_resp_early + insilico_resp_late

    # Select the top N images that drive both early and late part of the epoch
    if args.control == 'early-drive_late-drive':
        img_control = np.argsort(response_sum, 1)[::-1].astype(np.float32)
        # Ignore images conditions with univariate responses below the baseline
        # scores (plus a margin)
        idx_bad_early = np.where(
            insilico_resp_early[:,img_control.astype(np.int32)] < \
            baseline_resp_early[0]+margin)[0]
        idx_bad_late = np.where(resp_roi_2[high_1_high_2.astype(np.int32)] < \
            baseline_resp[1]+margin)[0]
        img_control[idx_bad_early] = np.nan
        img_control[idx_bad_late] = np.nan

# 2nd ranking: images with low univariate responses for both ROIs
low_1_low_2 = np.argsort(roi_sum).astype(np.float32)
# Ignore images conditions with univariate responses above the baseline
# scores (plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[low_1_low_2.astype(np.int32)] > \
	baseline_resp[0]-margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[low_1_low_2.astype(np.int32)] > \
	baseline_resp[1]-margin)[0]
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
	baseline_resp[0]+margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[high_1_low_2.astype(np.int32)] > \
	baseline_resp[1]-margin)[0]
high_1_low_2[idx_bad_roi_1] = np.nan
high_1_low_2[idx_bad_roi_2] = np.nan
# 4th ranking: images with low univariate responses for ROI 1 and high
# univariate responses for ROI 2
low_1_high_2 = np.argsort(roi_diff).astype(np.float32)
# Ignore images conditions with univariate responses above (ROI 1) or below
# (ROI 2) the baseline scores (minus/plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[low_1_high_2.astype(np.int32)] > \
	baseline_resp[0]-margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[low_1_high_2.astype(np.int32)] < \
	baseline_resp[1]+margin)[0]
low_1_high_2[idx_bad_roi_1] = np.nan
low_1_high_2[idx_bad_roi_2] = np.nan


































# Find the images controlling the neural responses
if args.control == 'drive':
    img_control = np.argsort(insilico_resp, axis=1)[:,::-1]
elif args.control == 'suppress':
    img_control = np.argsort(insilico_resp, axis=1)
img_control = img_control[:,:args.n_images]

# Cross-validate the controlling images across subjects
control_data = []
cv_control_data = []
for s in range(len(args.subjects)):
    s_cv = np.delete((0, 1), s)[0]
    control_data.append(insilico_resp[s,img_control[s]])
    cv_control_data.append(insilico_resp[s_cv,img_control[s]])
control_data = np.array(control_data)
cv_control_data = np.array(cv_control_data)


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
results = {
    'img_control': img_control
}

save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'sub-{args.subject}_roi-{args.roi}_{args.control}.npy'

np.save(os.path.join(save_dir, file_name), results)


# =============================================================================
# Save the controlling images
# =============================================================================
# Access the ILSVRC-2012 train split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train')

# Save directory
save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'controlling_images', args.encoding_model, f'subject-{args.subject}',
    f'roi-{args.roi}')
os.makedirs(save_dir, exist_ok=True)

# Loop across images
images = []
for i in range(args.n_images):

    # Get and preprocess the controlling images
    img, _ = imageset.__getitem__(img_control[i])
    min_size = min(img.size)
    transform = trn.Compose([
        trn.CenterCrop(min_size),
        trn.Resize((425,425))
        ])
    img = transform(img)
    images.append(np.array(img))

    # Save the controlling and baseline images as .png files
    file_name = f'{args.control}_img-{i:03}.png'
    # img.save(os.path.join(save_dir, file_name))

# Save the controlling and baseline images as h5py files
file_name = f'{args.control}_images.h5'
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('images', data=np.array(images))