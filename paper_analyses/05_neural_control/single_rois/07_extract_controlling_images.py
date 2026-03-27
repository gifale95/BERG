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
roi: str
    ROI used. Valid values are "V1", "V4", and "IT".
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

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--rois', default=['V1', 'V4', 'IT'], type=list)
parser.add_argument('--controls', default=['baseline', 'early-drive_late-drive', 'early-suppress_late-suppress', 'early-drive_late-suppress', 'early-suppress_late-drive'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Extract and save the controlling image h5py files
# =============================================================================
for sub in tqdm(args.subjects):
    for roi in args.rois:

        # Save directory
        save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
            'controlling_images', args.encoding_model,
            f'subject-{args.subject}', f'roi-{args.roi}')
        os.makedirs(save_dir, exist_ok=True)

        for control in args.controls:

            # Load the controlling image h5py file
            data_dir = os.path.join(args.berg_dir, 'neural_control',
                'single_rois', 'controlling_images', args.encoding_model,
                f'subject-{sub}', f'roi-{roi}', f'{control}_images.h5')
            images = h5py.File(data_dir, 'r')['images'][:]

            # Save the controlling images as .png files
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                file_name = f'{control}_img-{i+1:03}'
                img.save(os.path.join(save_dir, file_name))