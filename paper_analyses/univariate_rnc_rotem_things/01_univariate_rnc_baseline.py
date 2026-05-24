"""Get the fMRI responses for a randomly selected batch of N images (out of
all THINGS images), and then average these univariate responses across images.
This will result in one score indicating the mean fMRI univariate fMRI response
for that image batch.

# Repeating this step X times will create the null  distribution, from which
# the N images from the batch with score closest to the distribution's mean are
# selected. The in silico neural response score averaged across these N images
# provides the neural control baseline.

Parameters
----------
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
    If cv==1, the left-out subject during cross-validation, out of the 8 NSD
    subjects.
roi: str
    Used ROI.
n_images: int
    Number of retained controlling or baseline images.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--n_images', default=100, type=int)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Univariate RNC baseline <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Get the total dataset subjects
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# Load the fMRI responses for the THINGS images
# =============================================================================
# Load the fMRI responses of all subjects
fmri = []
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'fmri_responses')
for sub in all_subjects:
    file_name = f'fmri_sub-{sub:02d}_roi-{args.roi}_things.npy'
    fmri.append(np.load(os.path.join(data_dir, file_name)))
fmri = np.array(fmri)
n_images = fmri.shape[1]

# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are
# used to select the baseline images, and the same images are also used as
# baseline for the test subjects.
if args.cv == 0:
    fmri = np.mean(fmri, 0)
elif args.cv == 1:
    fmri_train = np.mean(np.delete(fmri, args.cv_subject-1, 0), 0)
    fmri_test = fmri[args.cv_subject-1]
    del fmri


# =============================================================================
# Compute the baseline images
# =============================================================================
# Empty result dictionaries
baseline_img = {}
if args.cv == 0:
    baseline_resp = {}
elif args.cv == 1:
    baseline_resp_train = {}
    baseline_resp_test = {}

# Create the baseline null distribution
null_distribution_images = []
if args.cv == 0:
    null_distribution = []
elif args.cv == 1:
    null_distribution_train = []
    null_distribution_test = []
for i in tqdm(range(args.n_iter), leave=False):
    sample = resample(np.arange(n_images), replace=False,
        n_samples=args.n_images)
    sample.sort()
    null_distribution_images.append(copy(sample))
    if args.cv == 0:
        null_distribution.append(np.mean(
            fmri[sample]).astype(np.float32))
    elif args.cv == 1:
        null_distribution_train.append(np.mean(
            fmri_train[sample]).astype(np.float32))
        null_distribution_test.append(np.mean(
            fmri_test[sample]).astype(np.float32))
null_distribution_images = np.array(null_distribution_images)
if args.cv == 0:
    null_distribution = np.array(null_distribution)
elif args.cv == 1:
    null_distribution_train = np.array(null_distribution_train)
    null_distribution_test = np.array(null_distribution_test)

# Select the baseline images from the null distribution: these are the
# images closest to the null distribution mean
if args.cv == 0:
    idx_best = np.argsort(abs(
        null_distribution - np.mean(null_distribution)))[0]
    baseline_img = null_distribution_images[idx_best]
    baseline_resp = fmri[baseline_img]
elif args.cv == 1:
    null_distribution_mean = np.mean(null_distribution_train, 0)
    idx_best = np.argsort(abs(null_distribution_train - \
        np.mean(null_distribution_train)))[0]
    baseline_img = null_distribution_images[idx_best]
    baseline_resp_train = fmri_train[baseline_img]
    baseline_resp_test = fmri_test[baseline_img]


# =============================================================================
# Save the baseline scores
# =============================================================================
if args.cv == 0:
    results = {
        'baseline_img': baseline_img,
        'baseline_resp': baseline_resp
    }
elif args.cv == 1:
    results = {
        'baseline_img': baseline_img,
        'baseline_resp_train': baseline_resp_train,
        'baseline_resp_test': baseline_resp_test
    }

save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'things',
    'baseline', f'cv-{args.cv}')
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = f'baseline_roi-{args.roi}.npy'
elif args.cv == 1:
    file_name = f'baseline_cv_subject-{args.cv_subject}_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), results)