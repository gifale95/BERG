"""Save the univariate RNC baseline and controlling images.

Parameters
----------
roi: str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest.
imageset : str
    The image set to use for the analysis. Possible values are:
    'imagenet_train' (ILSVRC-2012 training split),
    'imagenet_val' (ILSVRC-2012 validation split),
    'imagenet_all' (ILSVRC-2012 training + validation split),
    'coco' (MS COCO 2017 test split).
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
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet_all', type=str)
parser.add_argument('--n_images', default=25, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Save controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Break down the time windows
# =============================================================================
time_window_1_start, time_window_1_end = map(
    float, args.time_window_pair.split('__')[0].split('-'))
time_window_2_start, time_window_2_end = map(
    float, args.time_window_pair.split('__')[1].split('-'))


# =============================================================================
# Load the baseline and controlling image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', 'cv-0',
    args.time_window_pair, f'imageset-{args.imageset}',
    f'stats_roi-{args.roi}.npy')
data = np.load(data_dir, allow_pickle=True).item()

baseline_images = data['baseline_images']
controlling_images = data['controlling_images']


# =============================================================================
# Save the baseline images
# =============================================================================
# Access the ILSVRC-2012 validation split
if args.imageset == 'imagenet_val':
    imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir,
        split='val')
elif args.imageset == 'imagenet_train':
    imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir,
        split='train')
elif args.imageset == 'imagenet_all':
    imageset_val = torchvision.datasets.ImageNet(root=args.imagenet_dir,
        split='val')
    imageset_train = torchvision.datasets.ImageNet(root=args.imagenet_dir,
        split='train')

# Save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'images', f'roi-{args.roi}',
    args.time_window_pair, f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)

# Loop across time windows
for key, val in baseline_images.items():

    # Loop across images
    images = []
    for i in tqdm(range(args.n_images)):

        # Get and preprocess the baseline images
        if args.imageset in ['imagenet_val', 'imagenet_train']:
            img, _ = imageset.__getitem__(val[i])
        elif args.imageset == 'imagenet_all':
            if val[i] < 50000:
                img, _ = imageset_val.__getitem__(val[i])
            else:
                img, _ = imageset_train.__getitem__(val[i]-50000)
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
for key, val in controlling_images.items():

    # Loop across images
    images = []
    for i in tqdm(range(args.n_images)):

        # Get and preprocess the controlling images
        if args.imageset in ['imagenet_val', 'imagenet_train']:
            img, _ = imageset.__getitem__(controlling_images[key][i])
        elif args.imageset == 'imagenet_all':
            if controlling_images[key][i] < 50000:
                img, _ = imageset_val.__getitem__(controlling_images[key][i])
            else:
                img, _ = imageset_train.__getitem__(
                    controlling_images[key][i]-50000)
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