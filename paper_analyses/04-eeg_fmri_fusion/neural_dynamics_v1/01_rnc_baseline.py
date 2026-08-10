"""Get the t-fMRI responses for a randomly selected batch of N images (out of
all 50k ILSVRC-2012 validation images), and then average these univariate
responses across images. This will result in one score indicating the mean
t-fMRI univariate fMRI response for that image batch.

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
time_window_pair: str
    A string specifying the two time windows of interest.
n_images: int
    Number of retained controlling or baseline images.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import h5py
import numpy as np
import random
from berg import BERG
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.05-0.10__0.10-0.15', type=str)
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
# Break down the time windows
# =============================================================================
time_window_1_start, time_window_1_end = map(
    float, args.time_window_pair.split('__')[0].split('-'))
time_window_2_start, time_window_2_end = map(
    float, args.time_window_pair.split('__')[1].split('-'))


# =============================================================================
# Get the total dataset subjects
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# Load the t-fMRI responses for the 50k ILSVRC-2012 validation images
# =============================================================================
# Load the t-fMRI responses of all subjects
tfmri = []
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'tfmri_responses')
for sub in all_subjects:
    file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}.h5'
    tfmri.append(h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'])
tfmri = np.array(tfmri)
n_images = tfmri.shape[1]

# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are
# used to select the baseline images, and the same images are also used as
# baseline for the test subjects. If not cross-validating, average over all
# subjects.
if args.cv == 0:
    tfmri_mean = np.mean(tfmri, 0)
elif args.cv == 1:
    tfmri_tr = np.mean(np.delete(tfmri, args.cv_subject-1, 0), 0)
    tfmri_te = tfmri[args.cv_subject-1]
del tfmri


# =============================================================================
# Average the t-fMRI responses within the two time windows of interest
# =============================================================================
# Get the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = np.round(metadata_eeg['eeg']['times'], 3)

# Get the time window indices
t_min_1 = np.where(times == time_window_1_start)[0][0]
t_max_1 = np.where(times == time_window_1_end)[0][0]
t_min_2 = np.where(times == time_window_2_start)[0][0]
t_max_2 = np.where(times == time_window_2_end)[0][0]

# Average the t-fMRI responses and baseline scores within the two time windows
# of interest
if args.cv == 0:
    tfmri = {}
    tfmri['time_window_1'] = np.mean(tfmri_mean[:,t_min_1:t_max_1], 1)
    tfmri['time_window_2'] = np.mean(tfmri_mean[:,t_min_2:t_max_2], 1)
elif args.cv == 1:
    tfmri_train = {}
    tfmri_test = {}
    tfmri_train['time_window_1'] = np.mean(tfmri_tr[:,t_min_1:t_max_1], 1)
    tfmri_train['time_window_2'] = np.mean(tfmri_tr[:,t_min_2:t_max_2], 1)
    tfmri_test['time_window_1'] = np.mean(tfmri_te[:,t_min_1:t_max_1], 1)
    tfmri_test['time_window_2'] = np.mean(tfmri_te[:,t_min_2:t_max_2], 1)
    del tfmri_tr, tfmri_te


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
for tw in ['time_window_1', 'time_window_2']:
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
                tfmri[tw][sample]).astype(np.float32))
        elif args.cv == 1:
            null_distribution_train.append(np.mean(
                tfmri_train[tw][sample]).astype(np.float32))
            null_distribution_test.append(np.mean(
                tfmri_test[tw][sample]).astype(np.float32))
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
        baseline_img[tw] = null_distribution_images[idx_best]
        baseline_resp[tw] = tfmri[tw][baseline_img[tw]]
    elif args.cv == 1:
        null_distribution_mean = np.mean(null_distribution_train, 0)
        idx_best = np.argsort(abs(null_distribution_train - \
            np.mean(null_distribution_train)))[0]
        baseline_img[tw] = null_distribution_images[idx_best]
        baseline_resp_train[tw] = tfmri_train[tw][baseline_img[tw]]
        baseline_resp_test[tw] = tfmri_test[tw][baseline_img[tw]]


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

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc', 'baseline',
    f'cv-{args.cv}', args.time_window_pair)
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = f'baseline_roi-{args.roi}.npy'
elif args.cv == 1:
    file_name = f'baseline_cv_subject-{args.cv_subject}_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), results)