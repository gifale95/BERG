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
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
from torchvision import transforms as trn
from PIL import Image
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--roi_pair', default='V1-ventral', type=str)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--n_categories', default=10, type=int)
parser.add_argument('--n_exemplars', default=4, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/ccn_datasets/things_database', type=str)
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
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'things',
    'stats', 'cv-0', f'stats_{args.roi_pair}.npy')

controlling_images = np.load(data_dir,
    allow_pickle=True).item()['controlling_images']


# =============================================================================
# Save the controlling images
# =============================================================================
# Create the plot save directory
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'things',
    'controlling_images')
os.makedirs(save_dir, exist_ok=True)

# Get the image file paths
data_dir = os.path.join(args.things_dir, '01_image-level', 'image-paths.csv')
image_paths = pd.read_csv(data_dir, header=None).values.tolist()

# Imnage transform
transform = trn.Compose([
    trn.Resize((425,425))
    ])

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
            img_path = image_paths[val[ct][i]][0]
            img = Image.open(os.path.join(args.things_dir, img_path)).convert('RGB')
            img = transform(img)
            img_name = f'{cat}__{ct_roi}__img-{i+1:02d}.png'
            img.save(os.path.join(save_dir, img_name))