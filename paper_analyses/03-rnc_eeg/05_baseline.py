"""For each pairwise time point comparison, randomly select a batch of 50
images (out of all images from the chosen image set), pick the corresponding
RSM entries from the previously computed RSMs, and used representational
similarity analysis (RSA) to compare the RSMs of the two time points (through a
Pearson's correlation). This will results in one score indicating how
similar/dissimilar the multivariate responses for the two time points are for
that image batch.

Repeating this step 1 million times will create the multivariate RNC null 
distribution, from which the 50 images from the batch with score closest to
the distribution's mean are selected. The RSA score for these 50 images
provides the time points' RSA baseline.

Parameters
----------
cv : int
    If '1' multivariate RNC leaves the data of one subject out for
    cross-validation, if '0' multivariate RNC uses the data of all subjects.
cv_subject : int
    If cv==1, the left-out subject during cross-validation, out of the 10
    THINGS EEG2 subjects.
time_pair : str
    Used pairwise time point combination.
n_images_per_batch : int
    Amount of images per image batch.
null_dist_samples : int
    Amount of null distribution samples.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy
import random

from utils import load_rsms
from utils import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--time_pair', type=str, default='0.1-0.2')
parser.add_argument('--n_images_per_batch', type=int, default=50)
parser.add_argument('--null_dist_samples', type=int, default=1000000)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Multivariate RNC baseline <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Time point names
# =============================================================================
idx = args.time_pair.find('-')
time_1 = args.time_pair[:idx]
time_2 = args.time_pair[idx+1:]


# =============================================================================
# Load the RSMs
# =============================================================================
if args.cv == 0:
    # If not cross-validating, the RSMs averaged across all subjects is used.
    rsm_1, rsm_2 = load_rsms(args)

elif args.cv == 1:
    # If cross-validating, the RSMs of the train (N-1) subjects are used to
    # select the baseline images, and the same images are also used as baseline
    # for the test (remaining) subject.
    rsm_1_test, rsm_2_test = load_rsms(args, args.cv_subject, 'test')
    rsm_1_train, rsm_2_train = load_rsms(args, args.cv_subject, 'train')


# =============================================================================
# Create the null distribution
# =============================================================================
# If cross-validating, separate null distributions are created for the in
# silico EEG responses of the train and test subjects.

if args.cv == 0:
    null_distribution = np.zeros(args.null_dist_samples)
    null_distrubition_images = np.zeros((args.null_dist_samples,
        args.n_images_per_batch), dtype=np.int32)
    for s in tqdm(range(args.null_dist_samples)):
        sample = resample(np.arange(len(rsm_1)), replace=False,
            n_samples=args.n_images_per_batch)
        sample.sort()
        sample = np.reshape(sample, (1, -1))
        null_distrubition_images[s] = copy(sample)
        null_distribution[s] = evaluate(sample, rsm_1, rsm_2)[0]

elif args.cv == 1:
    null_distribution_test = np.zeros(args.null_dist_samples)
    null_distribution_train = np.zeros(args.null_dist_samples)
    null_distrubition_images = np.zeros((args.null_dist_samples,
        args.n_images_per_batch), dtype=np.int32)
    for s in tqdm(range(args.null_dist_samples)):
        sample = resample(np.arange(len(rsm_1_test)), replace=False,
            n_samples=args.n_images_per_batch)
        sample.sort()
        sample = np.reshape(sample, (1, -1))
        null_distrubition_images[s] = copy(sample)
        null_distribution_test[s] = evaluate(sample, rsm_1_test, rsm_2_test)[0]
        null_distribution_train[s] = evaluate(sample, rsm_1_train,
            rsm_2_train)[0]


# =============================================================================
# Choose the control images from the null distribution
# =============================================================================
# The control images are the bootstrap sample with the RSA correlation score
# closest to the null distribution mean.

if args.cv == 0:
    null_distribution_mean = np.mean(null_distribution)
    idx_best = np.argsort(abs(null_distribution - null_distribution_mean))[0]
    baseline_images = null_distrubition_images[idx_best]
    baseline_images_score = null_distribution[idx_best]

# The baseline images are chosen from the train subjects null distribution, and
# then also evaluated on the test subject (to see whether they generalize).
elif args.cv == 1:
    null_distribution_mean = np.mean(null_distribution_train)
    idx_best = np.argsort(abs(
        null_distribution_train - null_distribution_mean))[0]
    baseline_images = null_distrubition_images[idx_best]
    baseline_images_score_train = null_distribution_train[idx_best]
    baseline_images_score_test = null_distribution_test[idx_best]

# Convert the baseline images to integer
baseline_images = baseline_images.astype(np.int32)


# =============================================================================
# Save the results
# =============================================================================
if args.cv == 0:
    results = {
        'time_1': time_1,
        'time_2': time_2,
        'baseline_images': baseline_images,
        'baseline_images_score': baseline_images_score,
        }

elif args.cv == 1:
    results = {
        'time_1': time_1,
        'time_2': time_2,
        'baseline_images': baseline_images,
        'baseline_images_score_test': baseline_images_score_test,
        'baseline_images_score_train': baseline_images_score_train,
        'null_distribution_test': null_distribution_test
        }

save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'baseline', 'cv-'+format(args.cv), args.time_pair)
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = 'baseline.npy'
elif args.cv == 1:
    file_name = 'baseline_cv_subject-' + format(args.cv_subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)