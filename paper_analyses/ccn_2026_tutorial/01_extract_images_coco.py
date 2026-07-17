"""Load, preprocess, and save the 40,670 images from the MS COCO 2017 test
split.

Parameters
----------
coco_split : str
    Whether to use the 'train', 'val', or 'test split of MS COCO 2017.
coco_dir : str
    Directory of the MS COCO image set.
project_dir : str
    Directory of the ImageNet image set.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import h5py
from torchvision import transforms as trn
import random
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--coco_split', default='val', type=str)
parser.add_argument('--coco_dir', default='/scratch/giffordale95/datasets/image_sets/coco', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/ccn_2026_tutorial', type=str)
args, unknown = parser.parse_known_args()

print('>>> Load and preprocess MS COCO 2017 test images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load and preprocess the MS COCO images
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224))
])

# Get the list of image files from the MS COCO 2017 test split
coco_test2017_dir = os.path.join(args.coco_dir, 'images', 'test2017')
img_files = os.listdir(coco_test2017_dir)
img_files.sort()

# Load the images
images = []
for i in tqdm(range(len(img_files))):
    img_path = os.path.join(coco_test2017_dir, img_files[i])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    images.append(np.asarray(img))
images = np.array(images, dtype=np.uint8)


# =============================================================================
# Save the images
# =============================================================================
save_dir = os.path.join(args.project_dir, 'images')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'images_coco.h5')

with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('images', data=images, dtype=np.uint8)