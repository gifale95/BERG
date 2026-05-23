"""Aggregate the preprocessed ILSVRC-2012 images into an h5py file.

Parameters
----------
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
import h5py
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--tot_img_batches', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Aggregate ILSVRC-2012 images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and aggregate the preprocessed ILSVRC-2012 images across image batches
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'images')

for b in tqdm(range(args.tot_img_batches)):

    file_name = f'imagenet_split-{args.imagenet_split}_batch-{b:02d}.npy'

    image_batch = np.load(os.path.join(data_dir, file_name))

    if b == 0:
        images = image_batch
    else:
        images = np.append(images, image_batch, 0)
    del image_batch


# =============================================================================
# Save the preprocessed ILSVRC-2012 images
# =============================================================================
file_name = f'imagenet_split-{args.imagenet_split}.h5'

with h5py.File(os.path.join(data_dir, file_name), 'w') as f:
    f.create_dataset('images', data=images, dtype=np.uint8)