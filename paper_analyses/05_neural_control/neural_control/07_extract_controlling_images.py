"""Extract and save the controlling images from the h5py files.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : int
    Subject identifier for the monkey encoding model. Since the used encoding
    models are trained on the TVSD data, valid subject identifiers are "N" and
    "F".
controls: list
    List of neural control types used. Valid values are "baseline",
    "early-drive_late-drive", "early-suppress_late-suppress",
    "early-drive_late-suppress", and "early-suppress_late-drive".
berg_dir : str
    Directory of the BERG.
"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--controls', default=['early-drive_late-drive', 'early-suppress_late-suppress', 'early-drive_late-suppress', 'early-suppress_late-drive'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Extract and save the beaseline images
# =============================================================================
rois = ['V1', 'V4', 'IT']

for sub in tqdm(args.subjects):
    for roi in rois:

        # Data directory
        data_dir = os.path.join(args.berg_dir, 'neural_control',
            'neural_control', 'baseline_images', args.encoding_model,
            f'subject-{sub}', f'{roi}')

        # Load the baseline image h5py file
        h5_dir = os.path.join(data_dir, f'{roi}_baseline_images.h5')
        images = h5py.File(h5_dir, 'r')['images'][:]

        # Save the controlling images as .png files
        for i in range(len(images)):
            img = Image.fromarray(images[i])
            file_name = f'{roi}_baseline_img-{i+1:03}.png'
            img.save(os.path.join(data_dir, file_name))


# =============================================================================
# Extract and save the controlling images for the single-ROI neural control
# =============================================================================
for sub in tqdm(args.subjects):
    for roi in rois:

        # Data directory
        data_dir = os.path.join(args.berg_dir, 'neural_control',
            'neural_control', 'controlling_images', args.encoding_model,
            f'subject-{sub}', f'{roi}')

        for control in args.controls:

            # Load the controlling image h5py file
            h5_dir = os.path.join(data_dir, f'{roi}_{control}_images.h5')
            images = h5py.File(h5_dir, 'r')['images'][:]

            # Save the controlling images as .png files
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                file_name = f'{roi}_{control}_img-{i+1:03}.png'
                img.save(os.path.join(data_dir, file_name))


# =============================================================================
# Extract and save the controlling images for RNC
# =============================================================================
rois = ['V1', 'V4']

for sub in tqdm(args.subjects):

    # Data directory
    data_dir = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'controlling_images', args.encoding_model,
        f'subject-{sub}', f'{rois[0]}-{rois[1]}')

    for control_roi_1 in args.controls:
        for control_roi_2 in args.controls:

            # Load the controlling image h5py file
            file_name = (f'{rois[0]}_{control_roi_1}_'
                f'{rois[1]}_{control_roi_2}_images.h5')
            h5_dir = os.path.join(data_dir, file_name)
            images = h5py.File(h5_dir, 'r')['images'][:]

            # Save the controlling images as .png files
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                file_name = (f'{rois[0]}_{control_roi_1}_{rois[1]}_'
                    f'{control_roi_2}_img-{i+1:03}.png')
                img.save(os.path.join(data_dir, file_name))