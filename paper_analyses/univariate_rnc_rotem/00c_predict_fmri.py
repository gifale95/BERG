"""Predict fMRI responses for the ILSVRC-2012 images.

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
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
tot_img_batches : int
    The total number of batches in which the images are divided.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
import h5py
import torch

print(torch.cuda.get_device_name(0))
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_capability(0))

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--tot_img_batches', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Predict fMRI responses <<<')
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

vertices = {}

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
    idx = np.intersect1d(idx_roi, idx_ncsnr)

    # Store the vertex indices
    vertices[hemi] = np.zeros(len(ncsnr), dtype=int)
    vertices[hemi][idx] = 1


# =============================================================================
# Load the fMRI encoding model
# =============================================================================
model = berg.get_encoding_model(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject,
    selection={
        'lh_vertices': vertices['lh'],
        'rh_vertices': vertices['rh']
        }
    )


# =============================================================================
# Access the ILSVRC-2012 images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'images',
    f'imagenet_split-{args.imagenet_split}.h5')
images = h5py.File(data_dir, 'r')['images']


# =============================================================================
# Predict the fMRI responses to images, in batches
# =============================================================================
# Empty fMRI univariate response array
fmri_uni = np.zeros((len(images)), dtype=np.float32)

# Loop across image batches
tot_img_batches = args.tot_img_batches
imgs_per_batch = int(np.ceil(len(images) / tot_img_batches))
for i in tqdm(np.arange(tot_img_batches)):

    # Define the images from the current batch
    start_idx = i * imgs_per_batch
    end_idx = min((i + 1) * imgs_per_batch, len(images))
    images_batch = images[start_idx:end_idx]

    # Predict the fMRI responses
    fmri = berg.encode(model, images_batch)

    # Average the predicted fMRI responses across the vertices from the same
    # ROI, to get that ROIs univariate response
    fmri_uni[start_idx:end_idx] = np.mean(np.append(fmri[0], fmri[1], 1), 1)
    del images_batch, fmri


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'fmri_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'fmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'imagenet_split-{args.imagenet_split}.npy')

np.save(os.path.join(save_dir, file_name), fmri_uni)