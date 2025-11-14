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

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012/', type=str)
args, unknown = parser.parse_known_args()

print('>>> Get stimulus images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define the 200 ImageNet images used
# =============================================================================
# List the imagenet synsets
synsets = os.listdir(os.path.join(args.imagenet_dir, 'val'))
synsets.sort()

# List the categories
categories = [
    'animal.n.01', # 398 (* 50 = 19,900)
    'food.n.01', # 26 (* 50 = 1,300)
    'device.n.01', # 130 (* 50 = 6,500)
    'geological_formation.n.01' # 10 (* 50 = 500)
    ]

# Categorize each synset
synset_categories = np.zeros((len(synsets), len(categories)), dtype=np.int16)
for s, synset in enumerate(tqdm(synsets)):
    synset_name = wn.synset_from_pos_and_offset('n', int(synset[1:]))
    synset_name = synset_name.name()
    synset_name = wn.synset(synset_name)
    synset_cat = synset_name.hypernym_paths()
    for c, category in enumerate(categories):
        category_name = wn.synset(category)
        for sc in synset_cat:
            if np.isin(category_name, sc):
                synset_categories[s,c] = 1

# Get the indices of 100 animate categories ('animal.n.01')
idx_animate = np.where(synset_categories[:,0])[0][:100]

# Get the indices of 100 inanimate categories ('device.n.01')
idx_inanimate = np.where(synset_categories[:,2])[0][:100]

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
            'img-{:04d}_imagenet_label-{}.png'.format(i+1, label)))
    else:
        img.save(os.path.join(image_path, 'inanimate',
            'img-{:04d}_imagenet_label-{}.png'.format(i-99, label)))