"""Save the univariate RNC baseline and controlling images.

Parameters
----------
roi: str
    Used ROI.
time_window_1_start: float
    The starting point, in seconds, of first time window of interest.
time_window_1_end: float
    The ending point, in seconds, of first time window of interest.
time_window_2_start: float
    The starting point, in seconds, of second time window of interest.
time_window_2_end: float
    The ending point, in seconds, of second time window of interest.
n_images: int
    Number of retained controlling or baseline images.
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
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_1_start', default=0.06, type=float)
parser.add_argument('--time_window_1_end', default=0.1, type=float)
parser.add_argument('--time_window_2_start', default=0.1, type=float)
parser.add_argument('--time_window_2_end', default=0.2, type=float)
parser.add_argument('--n_images', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Save controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the baseline and control image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'stats', 'cv-0', (f'time_window_1-{args.time_window_1_start}_'
    f'{args.time_window_1_end}__time_window_2-{args.time_window_2_start}_'
    f'{args.time_window_2_end}'), f'stats_roi-{args.roi}.npy')
data = np.load(data_dir, allow_pickle=True).item()

baseline_images = data['baseline_images']
controlling_images = data['controlling_images']


# =============================================================================
# Save the baseline images
# =============================================================================
# Access the ILSVRC-2012 validation split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')

# Save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'images', (f'time_window_1-{args.time_window_1_start}_'
    f'{args.time_window_1_end}__time_window_2-{args.time_window_2_start}_'
    f'{args.time_window_2_end}'), f'roi-{args.roi}')
os.makedirs(save_dir, exist_ok=True)

# Loop across time windows
for key, val in baseline_images.items():

    # Loop across images
    images = []
    for i in tqdm(range(args.n_images)):

        # Get and preprocess the baseline images
        img, _ = imageset.__getitem__(val[i])
        min_size = min(img.size)
        transform = trn.Compose([
            trn.CenterCrop(min_size),
            trn.Resize((425,425))
            ])
        img = transform(img)
        images.append(np.array(img))

    # Save the baseline images as h5py files
    h5_dir = os.path.join(save_dir, f'baseline_images_{key}.h5')
    with h5py.File(h5_dir, 'w') as f:
        f.create_dataset('images', data=np.array(images))


# =============================================================================
# Save the controlling images
# =============================================================================
# Loop across control types
for key, val in controlling_images.items():

    # Loop across images
    images = []
    for i in tqdm(range(args.n_images)):

        # Get and preprocess the controlling images
        img, _ = imageset.__getitem__(val[i])
        min_size = min(img.size)
        transform = trn.Compose([
            trn.CenterCrop(min_size),
            trn.Resize((425,425))
            ])
        img = transform(img)
        images.append(np.array(img))

    # Save the controlling images as h5py files
    h5_dir = os.path.join(save_dir, f'controlling_images_{key}.h5')
    with h5py.File(h5_dir, 'w') as f:
        f.create_dataset('images', data=np.array(images))