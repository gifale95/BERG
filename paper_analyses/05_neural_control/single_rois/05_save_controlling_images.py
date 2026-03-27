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
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--control', default='early-drive_late-drive', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Save controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the controlling image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'stats', args.encoding_model,
    f'sub-{args.subject}_roi-{args.roi}_{args.control}.npy')

data = np.load(data_dir, allow_pickle=True).item()

img_control = data['img_control']


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

    # Save the controlling and baseline images as .png files
    file_name = (f'{args.control}_img-{i+1:03}'
        f'_imagenet_train-{img_control[i]:06}.png')
    # img.save(os.path.join(save_dir, file_name))

# Save the controlling and baseline images as h5py files
file_name = f'{args.control}_images.h5'
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('images', data=np.array(images))