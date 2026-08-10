"""Predict t-fMRI responses using in silico EEG responses for the ILSVRC-2012
validation images. The t-fMRI responses are then converted to univariate
responses by averaging across vertices within each ROI, for later use in the
univariate RNC algorithm.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
roi : str
    Used ROI.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
tot_img_batches : int
    The total number of batches in which the images are divided.
current_batch : int
    The image batch number used, out of the total image batches.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
import torchvision
from torchvision import transforms as trn
import gc
import torch
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--tot_img_batches', default=10, type=int)
parser.add_argument('--current_batch', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Predict t-fMRI responses <<<')
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
    subject=args.fmri_subject
    )

idx_v = {}

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
    idx_roi = idx_roi[idx_streams]
    idx_roi = np.where(idx_roi)[0]

    # Get the indices of ROI vertices with NCSNR above threshold
    idx_v[hemi] = np.intersect1d(idx_roi, idx_ncsnr)


# =============================================================================
# Access the ILSVRC-2012 val split, and load the images for the current batch
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224)),
    trn.Lambda(lambda img: np.transpose(img, (2, 0, 1))) # HWC to CHW
])

# Access the ILSVRC-2012 validation split
images = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val',
    transform=transform)

# Select the images from the current batch
imgs_per_batch = int(np.ceil(len(images) / args.tot_img_batches))
start_idx = args.current_batch * imgs_per_batch
end_idx = min((args.current_batch + 1) * imgs_per_batch, len(images))

# Load the images from the current batch
for i in tqdm(np.arange(start_idx, end_idx)):
    img, _ = images.__getitem__(i)
    if i == start_idx:
        images_batch = np.expand_dims(img, 0)
    else:
        images_batch = np.append(images_batch, np.expand_dims(img, 0), 0)


# =============================================================================
# Predict the in silico EEG responses
# =============================================================================
for es, esub in enumerate(args.eeg_subjects):

    # Load the EEG encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=esub
        )

    # Predict the in silico EEG responses, and append them across subjects
    # across the channels dimension
    if es == 0:
        eeg = berg.encode(model, images_batch)
    else:
        eeg = np.append(eeg, berg.encode(model, images_batch), 2)

    # Remove the model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Average the EEG responses across repeats
eeg = np.mean(eeg, 1)


# =============================================================================
# Loop across EEG time points
# =============================================================================
# Get the EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=args.eeg_subjects[0]
)
times = metadata_eeg['eeg']['times']

# Loop across EEG time points
for t in tqdm(range(len(times))):


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
    # Loop across hemisphers
    for h, hemi in enumerate(args.hemispheres):

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
            f'hemi-{hemi}_eeg_train_trials-all_eeg_time-{t:03d}.npy')
        reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'encoding_fusion_weights', file_name), allow_pickle=True).item()

        # Instantiate the fusion regression model
        reg = LinearRegression()
        reg.coef_ = reg_param['coef_'][idx_v[hemi]]
        reg.intercept_ = reg_param['intercept_'][idx_v[hemi]]
        reg.n_features_in_ = reg_param['n_features_in_']

        # Generate the t-fMRI responses
        tfmri_hemi = np.expand_dims(reg.predict(eeg[:,:,t]), 2)
        del reg_param, reg

        # Append the t-fMRI responses across hemispheres
        if h == 0:
            tfmri_time = tfmri_hemi
        else:
            tfmri_time = np.append(tfmri_time, tfmri_hemi, 1)
        del tfmri_hemi

    # Average the t-fMRI responses across vertices to create the ROI
    # univariate responses, and append them across time points
    if t == 0:
        tfmri = np.mean(tfmri_time, 1)
    else:
        tfmri = np.append(tfmri, np.mean(tfmri_time, 1), 2)
    del tfmri_time

# Delete the EEG reponses
del eeg


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'tfmri_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'batch-{args.current_batch:02d}.npy')

np.save(os.path.join(save_dir, file_name), tfmri)