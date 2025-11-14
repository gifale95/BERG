"""Get the 200 images used for the object exemplar and animacy categorization.
All images come from ImageNet: 100 images consist of animate objects, and 100
images consist of inanimate objects.

Parameters
----------
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import torchvision
from torchvision import transforms as trn
from sklearn.utils import resample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012/', type=str)
args, unknown = parser.parse_known_args()

print('>>> Get stimulus images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Define the 200 ImageNet images used
# =============================================================================
# List the imagenet synsets
synsets = os.listdir(os.path.join(args.imagenet_dir, 'val'))
synsets.sort()

# Get the indices of animate and inanimate categories
animate_categories = np.zeros(len(synsets), dtype=np.int16)
inanimate_categories = np.zeros(len(synsets), dtype=np.int16)
animate_cat = wn.synset('animal.n.01')
for s, synset in enumerate(synsets):
    synset_name = wn.synset_from_pos_and_offset('n', int(synset[1:]))
    synset_cat = synset_name.hypernym_paths()
    if any(animate_cat in cat for cat in synset_cat):
        animate_categories[s] = 1
    else:
        inanimate_categories[s] = 1

# Get the indices of 100 randomly selected animate categories ('animal.n.01')
idx_animate = np.where(animate_categories == 1)[0]
idx_animate = resample(idx_animate, replace=False, n_samples=100,
    random_state=seed)

# Get the indices of the inanimate categories (all which are not an animal).
# Since some of these images might still contain humans/animals, we save all
# possible inanimate categories and then manually retain the first 100 images
# that do not contain any humans/animals.
idx_inanimate = np.where(inanimate_categories == 1)[0]
idx_inanimate = resample(idx_inanimate, replace=False, random_state=seed)

# Multiply the indices by 50, since the ILSVRC-2012 validation split has 50
# images per category (and we will only use one image per each of the 200
# cateogries)
idx_animate *= 50
idx_inanimate *= 50
idx_all = np.append(idx_animate, idx_inanimate)

# Access the ILSVRC-2012 validation split
dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')


# =============================================================================
# Load the 200 images, resize to 224x224 (center crop), and save as PNG files
# =============================================================================
# Create the output directories
image_path = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'object_exemplar_animacy_categorization', 'stimuli')
os.makedirs(os.path.join(image_path, 'animate'), exist_ok=True)
os.makedirs(os.path.join(image_path, 'inanimate'), exist_ok=True)

# Define the image transform
image_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224)
    ])

# Save the images
for i in tqdm(range(len(idx_all))):
    img_idx = idx_all[i]
    img, label = dataset[img_idx]
    img = image_transform(img)
    if i < 100:
        img.save(os.path.join(image_path, 'animate',
            'img-{:04d}_imagenet_label-{:04d}.png'.format(i+1, label)))
    else:
        img.save(os.path.join(image_path, 'inanimate',
            'img-{:04d}_imagenet_label-{:04d}.png'.format(i-99, label)))