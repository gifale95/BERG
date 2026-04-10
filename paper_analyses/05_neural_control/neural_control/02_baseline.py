"""Get the in silico neural responses for a randomly selected batch of N images
# (out of all images in the image set), and then average these univariate
# responses across images. This will result in one score indicating the mean in
# silico univariate fMRI response for that image batch.

# Repeating this step X times will create the null  distribution, from which
# the N images from the batch with score closest to the distribution's mean are
# selected. The in silico neural response score averaged across these N images
# provides the neural control baseline.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : int
    Subject identifier for the monkey encoding model. Since the used encoding
    models are trained on the TVSD data, valid subject identifiers are "N" and
    "F".
roi: str
    ROI used. Valid values are "V1", "V4", and "IT".
n_images: int
    Number of retained controlling or baseline images.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import h5py
import numpy as np
import random
import torchvision
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--n_images', default=50, type=int)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Baseline <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'insilico_responses', args.encoding_model)
file_name = f'insilico_responses_sub-{args.subject}_roi-{args.roi}.npy'

data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
insilico_resp = data['responses']
metadata = data['metadata']


# =============================================================================
# Compute the baseline images
# =============================================================================
# Create the baseline null distribution
images = len(insilico_resp)
null_distribution = []
null_distribution_images = []
for i in tqdm(range(args.n_iter), leave=False):
    sample = resample(np.arange(images), replace=False,
        n_samples=args.n_images)
    sample.sort()
    null_distribution_images.append(copy(sample))
    null_distribution.append(np.mean(
        insilico_resp[sample], 0).astype(np.float32))
null_distribution = np.array(null_distribution)
null_distribution_images = np.array(null_distribution_images)

# Compute the confidence intervals of the baseline null distribution
ci_low_null_distribution = np.percentile(null_distribution, 2.5, 0)
ci_high_null_distribution = np.percentile(null_distribution, 97.5, 0)

# Select the baseline images from the null distribution: these are the images
# closest to the null distribution mean, based on the responses averaged across
# the entire epoch (25-200ms)
times = metadata['utah_array']['times']
t_min = np.where(times == 25)[0][0]
t_max = np.where(times == 199)[0][0]
null_distribution_mean = np.mean(null_distribution, 0)
idx_best = np.argsort(abs(np.mean(null_distribution[:,t_min:t_max+1], 1) - \
    np.mean(null_distribution_mean[t_min:t_max+1])))[0]
img_baseline = null_distribution_images[idx_best]


# =============================================================================
# Get the neural response scores for the baseline images
# =============================================================================
baseline_resp = insilico_resp[img_baseline]


# =============================================================================
# Save the baseline results
# =============================================================================
results = {
    'ci_low_null_distribution': ci_low_null_distribution,
    'ci_high_null_distribution': ci_high_null_distribution,
    'img_baseline': img_baseline,
    'baseline_resp': baseline_resp
}

save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'sub-{args.subject}_roi-{args.roi}_baseline.npy'

np.save(os.path.join(save_dir, file_name), results)


# =============================================================================
# Save the baseline images
# =============================================================================
# Access the ILSVRC-2012 train split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train')

# Save directory
save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'controlling_images', args.encoding_model, f'subject-{args.subject}',
    f'roi-{args.roi}')
os.makedirs(save_dir, exist_ok=True)

# Loop across images
images = []
for i in tqdm(range(args.n_images)):

    # Get and preprocess the baseline images
    img, _ = imageset.__getitem__(img_baseline[i])
    min_size = min(img.size)
    transform = trn.Compose([
        trn.CenterCrop(min_size),
        trn.Resize((425,425))
        ])
    img = transform(img)
    images.append(np.array(img))

    # Save the baseline images as .png files
    file_name = (f'baseline_img-{i+1:03}'
        f'_imagenet_train-{img_baseline[i]:06}.png')
    # img.save(os.path.join(save_dir, file_name))

# Save the baseline images as h5py files
with h5py.File(os.path.join(save_dir, 'baseline_images.h5'), 'w') as f:
    f.create_dataset('images', data=np.array(images))