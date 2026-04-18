"""Save the controlling images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : int
    Subject identifier for the monkey encoding model. Since the used encoding
    models are trained on the TVSD data, valid subject identifiers are "N" and
    "F".
roi_1: str
    First ROI used. Valid values are "V1", "V4", and "IT".
roi_2: str
    Second ROI used. Valid values are "V1", "V4", and "IT". If None, then only
    one ROI (roi_1) is used for neural control.
control_roi_1: str
    Neural control objective for the first ROI.
    If "early-drive_late-drive", then both the early (25-100ms) and late
    (101-200ms) part of the epoch are driven.
    If "early-suppress_late-suppress", then both the early and late part of the
    epoch are suppressed.
    If "early-drive_late-suppress", then the early part of the epoch is driven
    while the late part is suppressed.
    If "early-suppress_late-drive", then the early part of the epoch is
    suppressed while the late part is driven.
control_roi_2: str
    Neural control objective for the second ROI. The valid values are the same
    as for control_roi_1.
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
import torchvision
from tqdm import tqdm
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi_1', default='V1', type=str)
parser.add_argument('--roi_2', default=None, type=str)
parser.add_argument('--control_roi_1', default='early-drive_late-drive', type=str)
parser.add_argument('--control_roi_2', default=None, type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Save controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the controlling image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'stats', args.encoding_model)
if args.roi_2 is not None:
    file_name = (f'sub-{args.subject}_roi_1-{args.roi_1}_{args.control_roi_1}'
        f'_roi_2-{args.roi_2}_{args.control_roi_2}.npy')
else:
    file_name = f'sub-{args.subject}_roi-{args.roi_1}_{args.control_roi_1}.npy'

data = np.load(data_dir, allow_pickle=True).item()

img_control = data['img_control']


# =============================================================================
# Save the controlling images
# =============================================================================
# Access the ILSVRC-2012 train split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train')

# Save directory
if args.roi_2 is not None:
    save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
        'controlling_images', args.encoding_model, f'subject-{args.subject}',
        f'{args.roi_1}-{args.roi_2}')
else:
    save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
        'controlling_images', args.encoding_model, f'subject-{args.subject}',
        f'{args.roi_1}')
os.makedirs(save_dir, exist_ok=True)

# Loop across images
images = []
for i in tqdm(range(len(img_control))):

    # Get and preprocess the controlling images
    img, _ = imageset.__getitem__(img_control[i])
    min_size = min(img.size)
    transform = trn.Compose([
        trn.CenterCrop(min_size),
        trn.Resize((425,425))
        ])
    img = transform(img)
    images.append(np.array(img))

# Save the controlling and baseline images as h5py files
if args.roi_2 is not None:
    file_name = (f'{args.roi_1}_{args.control_roi_1}_'
        f'{args.roi_2}_{args.control_roi_2}_images.h5')
else:
    file_name = f'{args.roi_1}_{args.control_roi_1}_images.h5'
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('images', data=np.array(images))