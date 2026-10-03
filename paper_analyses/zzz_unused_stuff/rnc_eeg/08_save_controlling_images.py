"""Save the controlling images selected by applying multivariate RNC on the
in silico multivariate EEG responses averaged over all subjects (i.e., with no
subject cross-validation).

Parameters
----------
time_pair : str
    Used pairwise time point combination.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import h5py
from PIL import Image
import torchvision
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--time_pair', type=str, default='0.1-0.2')
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Save the multivariate RNC controlling images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Time point names
# =============================================================================
idx = args.time_pair.find('-')
time_1 = args.time_pair[:idx]
time_2 = args.time_pair[idx+1:]


# =============================================================================
# Load the multivariate RNC stats
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc', 'stats',
    'cv-0', args.time_pair, 'stats.npy')

data_dict = np.load(data_dir, allow_pickle=True).item()

alignment_images = data_dict['best_generation_image_batches']['align'][-1]
disentanglement_images = data_dict['best_generation_image_batches']\
    ['disentangle'][-1]
baseline_images = data_dict['baseline_images']


# =============================================================================
# Access the ImageNet validation images
# =============================================================================
images = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')

# Get the indices of the first 10 images for each ILSVRC-2012 category
idx_img = []
for c in range(1000):
    idx_img += list(range(c*50, c*50+10))
idx_img = np.array(idx_img)


# =============================================================================
# Save the aligning images
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'controlling_images', 'cv-0', args.time_pair)
os.makedirs(save_dir, exist_ok=True)

for i, img_num in enumerate(alignment_images):

    img, _ = images.__getitem__(idx_img[img_num])

    min_size = min(img.size)
    transform = trn.Compose([
        trn.CenterCrop(min_size),
        trn.Resize((425,425))
        ])
    img = transform(img)

    img_name = 'align_img-' + format(i+1, '03') + '_imagenet_val-' + \
        format(idx_img[img_num], '06') + '.png'
    img.save(os.path.join(save_dir, img_name))


# =============================================================================
# Save the disentangling images
# =============================================================================
for i, img_num in enumerate(disentanglement_images):

    img, _ = images.__getitem__(idx_img[img_num])

    min_size = min(img.size)
    transform = trn.Compose([
        trn.CenterCrop(min_size),
        trn.Resize((425,425))
        ])
    img = transform(img)

    img_name = 'disentangle_img-' + format(i+1, '03') + '_imagenet_val-' + \
        format(idx_img[img_num], '06') + '.png'
    img.save(os.path.join(save_dir, img_name))


# =============================================================================
# Save the baseline images
# =============================================================================
for i, img_num in enumerate(baseline_images):

    img, _ = images.__getitem__(idx_img[img_num])

    min_size = min(img.size)
    transform = trn.Compose([
        trn.CenterCrop(min_size),
        trn.Resize((425,425))
        ])
    img = transform(img)

    img_name = 'rnc_baseline_img-' + format(i+1, '03') + '_imagenet_val-' + \
        format(idx_img[img_num], '06') + '.png'
    img.save(os.path.join(save_dir, img_name))