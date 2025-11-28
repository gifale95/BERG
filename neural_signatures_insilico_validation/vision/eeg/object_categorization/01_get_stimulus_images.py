"""Get the ImageNet images used for the object categorization analysis.

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
# Define the used ImageNet images
# =============================================================================
# Animate tmage category labels
animate_obj = {
    1: 'goldfish',
    2: 'shark',
    8: 'hen',
    22: 'eagle',
    24: 'owl',
    145: 'penguin',
    147: 'whale',
    235: 'dog',
    270: 'wolf',
    292: 'tiger',
    294: 'bear',
    315: 'mantis',
    335: 'squirrel',
    340: 'zebra',
    347: 'bison',
    348: 'ram',
    366: 'gorilla',
    373: 'macaque',
    386: 'elephant',
    388: 'panda',
}

# Inanimate tmage category labels
inanimate_obj = {
    404: 'airliner',
    407: 'ambulance',
    425: 'barn',
    466: 'train',
    494: 'bell',
    497: 'church',
    550: 'espresso maker',
    604: 'hourglass',
    637: 'mailbox',
    681: 'notebook',
    687: 'organ',
    754: 'radio',
    783: 'screw',
    820: 'steam locomotive',
    859: 'toaster',
    866: 'tractor',
    900: 'water tower',
    938: 'cauliflower',
    949: 'strawberry',
    954: 'banana'
}

# Multiply the indices by 50, since the ILSVRC-2012 validation split has 50
# images per category (and we will only use one image per each of the 200
# cateogries)
idx = {}
idx['animate'] = np.array(list(animate_obj.keys())) * 50
idx['inanimate'] = np.array(list(inanimate_obj.keys())) * 50

# Access the ILSVRC-2012 validation split
dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')


# =============================================================================
# Load the images, resize to 224x224 (center crop), and save as PNG files
# =============================================================================
# Create the output directories
image_path = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'object_categorization', 'stimuli')
os.makedirs(os.path.join(image_path, 'animate'), exist_ok=True)
os.makedirs(os.path.join(image_path, 'inanimate'), exist_ok=True)

# Define the image transform
image_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224)
    ])

# Save the first 10 exemplars of each image category
exemplars = 10
for animacy in ['animate', 'inanimate']:
    count = 1
    for i in tqdm(range(len(idx[animacy]))):
        for e in range(10):
            img_idx = idx[animacy][i] + e
            img, label = dataset[img_idx]
            img = image_transform(img)
            img.save(os.path.join(image_path, animacy,
                'img-{:04d}_imagenet_label-{:04d}.png'.format(count, label)))
            count += 1