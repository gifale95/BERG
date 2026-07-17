"""Load, preprocess, and save the 50,000 images from the ILSVRC-2012 validation
split.

Parameters
----------
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php
project_dir : str
    Directory of the ImageNet image set.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import h5py
import torchvision
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_split', default='val', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/ccn_2026_tutorial', type=str)
args, unknown = parser.parse_known_args()

print('>>> Load and preprocess ILSVRC-2012 validation images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and preprocess the ILSVRC-2012 images
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224))
])

# Access the ILSVRC-2012 imageset
images_all = torchvision.datasets.ImageNet(root=args.imagenet_dir,
    split=args.imagenet_split, transform=transform)

# Load the images
images = []
for i in tqdm(range(len(images_all))):
    images.append(np.asarray(images_all.__getitem__(i)[0]))
images = np.array(images, dtype=np.uint8)


# =============================================================================
# Save the images
# =============================================================================
save_dir = os.path.join(args.project_dir, 'images')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'images_imagenet.h5')

with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('images', data=images, dtype=np.uint8)