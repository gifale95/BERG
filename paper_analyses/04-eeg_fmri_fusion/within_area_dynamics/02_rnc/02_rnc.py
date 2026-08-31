"""Apply univariate relational neural control (RNC) to find images that align
or disentangle the t-fMRI responses of an ROI between two time windows of
interest.

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
imageset : str
    The image set to use for the analysis. Possible values are: 'imagenet'
    (ILSVRC-2012 validation split) and 'coco' (MS COCO 2017 test split).
n_images: int
    Number of retained controlling or baseline images.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet', type=str)
parser.add_argument('--n_images', default=25, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Univariate RNC <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


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
# Load the univariate RNC baseline scores, and average them across images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'baseline', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')

if args.cv == 0:
    file_name = f'baseline_roi-{args.roi}.npy'
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    base_1 = np.mean(data['baseline_resp']['time_window_1'])
    base_2 = np.mean(data['baseline_resp']['time_window_2'])

elif args.cv == 1:
    file_name = f'baseline_cv_subject-{args.cv_subject}_roi-{args.roi}.npy'
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    base_1 = np.mean(data['baseline_resp_train']['time_window_1'])
    base_2 = np.mean(data['baseline_resp_train']['time_window_2'])


# =============================================================================
# Load the t-fMRI responses
# =============================================================================
# Load the t-fMRI responses of all subjects
tfmri = []
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'tfmri_responses')
for sub in all_subjects:
    file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}_imageset_{args.imageset}.h5'
    tfmri.append(h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'])
tfmri = np.array(tfmri)

# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are
# used to select the controlling images, and the controlling images will then
# be validated on the fMRI responses for the test subjects. If not
# cross-validating, average over all subjects.
if args.cv == 0:
    tfmri_mean = np.mean(tfmri, 0)
elif args.cv == 1:
    tfmri_mean = np.delete(tfmri, args.cv_subject-1, 0)
    tfmri_mean = np.mean(tfmri_mean, 0)
del tfmri


# =============================================================================
# Average the tfMRI responses within the two time windows of interest
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

# Average the t-fMRI responses within the two time windows of interest
tfmri_1 = np.mean(tfmri_mean[:,t_min_1:t_max_1], 1)
tfmri_2 = np.mean(tfmri_mean[:,t_min_2:t_max_2], 1)


# =============================================================================
# Rank the images based on their t-fMRI univariate responses
# =============================================================================
# Univariate response score margin used to constrain the selection of the
# control images. The margin is defined as half the standard deviation of the
# t-fMRI responses across all images for each time window. The margin is used
# to ignore images that have t-fMRI responses that are too close to the
# baseline scores, as these images may not be informative for aligning or
# disentangling the two time windows. 
margin_1 = np.std(tfmri_1) / 2 # np.std(tfmri_1) / 4 * 3
margin_2 = np.std(tfmri_2) / 2 # np.std(tfmri_2) / 4 * 3

# Select the top N images that align the t-fMRI univariate responses of the two
# time windows (i.e., that lead to both time windows having either high or low
# univariate responses).
# 1st ranking: images with high univariate responses for both time windows
response_sum = tfmri_1 + tfmri_2
high_1_high_2 = np.argsort(response_sum)[::-1].astype(np.float32)
# Ignore image conditions with t-fMRI responses below the baseline scores
idx_bad_1 = np.where(tfmri_1[high_1_high_2.astype(np.int32)] < base_1+margin_1)[0]
idx_bad_2 = np.where(tfmri_2[high_1_high_2.astype(np.int32)] < base_2+margin_2)[0]
high_1_high_2[idx_bad_1] = np.nan
high_1_high_2[idx_bad_2] = np.nan
high_1_high_2 = high_1_high_2[~np.isnan(high_1_high_2)].astype(np.int32)
# 2nd ranking: images with low univariate responses for both time windows
low_1_low_2 = np.argsort(response_sum).astype(np.float32)
# Ignore images conditions with t-fMRI responses above the baseline scores
idx_bad_1 = np.where(tfmri_1[low_1_low_2.astype(np.int32)] > base_1-margin_1)[0]
idx_bad_2 = np.where(tfmri_2[low_1_low_2.astype(np.int32)] > base_2-margin_2)[0]
low_1_low_2[idx_bad_1] = np.nan
low_1_low_2[idx_bad_2] = np.nan
low_1_low_2 = low_1_low_2[~np.isnan(low_1_low_2)].astype(np.int32)

# Select the top N images that disentangle the t-fMRI univariate responses of
# the two time windows (i.e., that lead one time window having high responses
# and the other time window low responses, or vice versa).
# 3rd ranking: images with high univariate responses for time window 1 and low
# univariate responses for time window 2
response_diff = tfmri_1 - tfmri_2
high_1_low_2 = np.argsort(response_diff)[::-1].astype(np.float32)
# Ignore images conditions with univariate responses below (time window 1) or
# above (time window 2) the baseline scores
idx_bad_1 = np.where(tfmri_1[high_1_low_2.astype(np.int32)] < base_1+margin_1)[0]
idx_bad_2 = np.where(tfmri_2[high_1_low_2.astype(np.int32)] > base_2-margin_2)[0]
high_1_low_2[idx_bad_1] = np.nan
high_1_low_2[idx_bad_2] = np.nan
high_1_low_2 = high_1_low_2[~np.isnan(high_1_low_2)].astype(np.int32)
# 4th ranking: images with low univariate responses for ROI 1 and high
# univariate responses for ROI 2
low_1_high_2 = np.argsort(response_diff).astype(np.float32)
# Ignore images conditions with univariate responses above (time window 1) or
# below (time window 2) the baseline scores
idx_bad_1 = np.where(tfmri_1[low_1_high_2.astype(np.int32)] > base_1-margin_1)[0]
idx_bad_2 = np.where(tfmri_2[low_1_high_2.astype(np.int32)] < base_2+margin_2)[0]
low_1_high_2[idx_bad_1] = np.nan
low_1_high_2[idx_bad_2] = np.nan
low_1_high_2 = low_1_high_2[~np.isnan(low_1_high_2)].astype(np.int32)


# =============================================================================
# Save the results
# =============================================================================
data_dict = {
    'times': times,
    'time_window_1_start': time_window_1_start,
    'time_window_1_end': time_window_1_end,
    'time_window_2_start': time_window_2_start,
    'time_window_2_end': time_window_2_end,
    'tfmri_1': tfmri_1,
    'tfmri_2': tfmri_2,
    'controlling_images': {
         'high_1_high_2': high_1_high_2,
         'low_1_low_2': low_1_low_2,
         'high_1_low_2': high_1_low_2,
         'low_1_high_2': low_1_high_2
    }
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'quantitative_results', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = f'image_ranking_roi-{args.roi}.npy'
elif args.cv == 1:
    file_name = f'image_ranking_cv_subject-{args.cv_subject:02d}_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), data_dict)