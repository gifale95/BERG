"""Load, preprocess, and save the ILSVRC-2012 images, in batches.

Parameters
----------
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
tot_img_batches : int
    The total number of batches in which the images are divided.
current_batch : int
    The image batch number used, out of the total image batches.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--tot_img_batches', default=100, type=int)
parser.add_argument('--current_batch', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Load and preprocess ILSVRC-2012 images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and preprocess the ILSVRC-2012 images
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224)),
    trn.Lambda(lambda img: np.transpose(img, (2, 0, 1))) # HWC to CHW
])

# Access the ILSVRC-2012 imageset
images = torchvision.datasets.ImageNet(root=args.imagenet_dir,
    split=args.imagenet_split, transform=transform)

# Select the images from the current batch
imgs_per_batch = int(np.ceil(len(images) / args.tot_img_batches))
start_idx = args.current_batch * imgs_per_batch
end_idx = min((args.current_batch + 1) * imgs_per_batch, len(images))

# Load the images from the current batch
for i in tqdm(np.arange(start_idx, end_idx)):
    img, _ = images.__getitem__(i)
    if i == start_idx:
        images_batch = np.expand_dims(img, 0)
    else:
        images_batch = np.append(images_batch, np.expand_dims(img, 0), 0)


# =============================================================================
# Save the images
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'images')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'imagenet_split-{args.imagenet_split}_batch-'
    f'{args.current_batch:02d}.npy')

np.save(os.path.join(save_dir, file_name), images_batch)