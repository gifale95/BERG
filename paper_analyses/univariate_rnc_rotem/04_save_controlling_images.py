"""Save the univariate RNC controlling images.

Parameters
----------
roi_pair : str
    Used pairwise ROI combination.
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
n_categories: int
    Number of retained image categories.
n_exemplars: int
    Number of retained image exemplars for each category and neural control
    condition.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
import torchvision
from torchvision import transforms as trn
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--roi_pair', default='V1-ventral', type=str)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--n_categories', default=50, type=int)
parser.add_argument('--n_exemplars', default=4, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Save controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Get the total dataset subjects
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]
rois = [roi_1, roi_2]


# =============================================================================
# Load the controlling image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'stats', 'cv-0',
    f'stats_{args.roi_pair}.npy')

controlling_images = np.load(data_dir,
    allow_pickle=True).item()['controlling_images']


# =============================================================================
# Save the controlling images
# =============================================================================
# Create the plot save directory
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'controlling_images')
os.makedirs(save_dir, exist_ok=True)

# Access the ILSVRC-2012 image set
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir,
    split=args.imagenet_split)

# Loop across image categories
for c, (cat, val) in enumerate(tqdm(controlling_images.items())):

    # Loop across neural control types
    control_types = ['high_1_high_2', 'low_1_low_2', 'high_1_low_2',
        'low_1_high_2']
    for ct in control_types:
        if ct == 'high_1_high_2':
            ct_roi = f'high_{roi_1}_high_{roi_2}'
        elif ct == 'low_1_low_2':
            ct_roi = f'low_{roi_1}_low_{roi_2}'
        elif ct == 'high_1_low_2':
            ct_roi = f'high_{roi_1}_low_{roi_2}'
        elif ct == 'low_1_high_2':
            ct_roi = f'low_{roi_1}_high_{roi_2}'

        # Loop across images
        for i in range(args.n_exemplars):

            # Get and preprocess the controlling images
            img, _ = imageset.__getitem__(val[ct][i])
            min_size = min(img.size)
            transform = trn.Compose([
                trn.CenterCrop(min_size),
                trn.Resize((425,425))
                ])
            img = transform(img)
            img_name = f'{cat}__{ct_roi}__img-{i+1:02d}.png'
            img.save(os.path.join(save_dir, img_name))