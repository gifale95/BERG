"""Test whether the controlling images found using the t-fMRI responses of the
train subjects generalize to the in t-fMRI responses of the left-out subject.
Stats include confidence intervals and significance.

Parameters
----------
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
roi: str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest.
imageset : str
    The image set to use for the analysis. Possible values are: 'imagenet'
    (ILSVRC-2012 validation split) and 'coco' (MS COCO 2017 test split).
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
import h5py
from berg import BERG
from scipy.stats import ttest_rel

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet', type=str)
parser.add_argument('--n_images', default=25, type=int)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
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
# Load the t-fMRI responses of all subjects
# =============================================================================
tfmri = []
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'tfmri_responses')

for sub in all_subjects:
    file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}_imageset_{args.imageset}.h5'
    tfmri.append(h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'])

tfmri = np.array(tfmri)


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

# Average the t-fMRI responses within the two time windows of interest
tfmri_1 = np.mean(tfmri[:,:,t_min_1:t_max_1], 2)
tfmri_2 = np.mean(tfmri[:,:,t_min_2:t_max_2], 2)


# =============================================================================
# Load the univariate RNC controlling images
# =============================================================================
control_types = ['high_1_low_2', 'low_1_high_2']
controlling_images = {}

data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'quantitative_results', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')

if args.cv == 0:
    file_name = f'image_ranking_roi-{args.roi}.npy'
    data_dict = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    times = data_dict['times']
    for ct in control_types:
        controlling_images[ct] = data_dict['controlling_images'][ct]\
            [:args.n_images]

elif args.cv == 1:
    for ct in control_types:
        controlling_images[ct] = []
    for s in all_subjects:
        file_name = f'image_ranking_cv_subject-{s:02d}_roi-{args.roi}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        times = data_dict['times']
        for ct in control_types:
            controlling_images[ct].append(
                data_dict['controlling_images'][ct][:args.n_images])
    for ct in control_types:
        controlling_images[ct] = np.asarray(controlling_images[ct])


# =============================================================================
# Load the univariate RNC baseline images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'baseline', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')
baseline_images = {}

if args.cv == 0:
    file_name = f'baseline_roi-{args.roi}.npy'
    data_dict = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    baseline_images['time_window_1'] = data_dict['baseline_img']['time_window_1']
    baseline_images['time_window_2'] = data_dict['baseline_img']['time_window_2']

elif args.cv == 1:
    baseline_images['time_window_1'] = np.zeros((len(all_subjects),
        args.n_images), dtype=np.int32)
    baseline_images['time_window_2'] = np.zeros((len(all_subjects),
        args.n_images), dtype=np.int32)
    for s, sub in enumerate(all_subjects):
        file_name = f'baseline_cv_subject-{sub}_roi-{args.roi}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        baseline_images['time_window_1'][s] = \
            data_dict['baseline_img']['time_window_1']
        baseline_images['time_window_2'][s] = \
            data_dict['baseline_img']['time_window_2']


# =============================================================================
# Validate the neural control conditions across subjects (only for cv==1)
# =============================================================================
# Get the test subject univariate t-fMRI responses for the controlling
# and baseline images.

if args.cv == 1:

    # Univariate t-fMRI response arrays of shape:
    # (8 Subjects × Target images)
    cv_resp_1 = {}
    cv_resp_2 = {}
    for ct in control_types:
        cv_resp_1[ct] = np.zeros((len(all_subjects), args.n_images),
            dtype=np.float32)
        cv_resp_2[ct] = np.zeros((len(all_subjects), args.n_images),
            dtype=np.float32)
    base_resp_1 = np.zeros((len(all_subjects), args.n_images), dtype=np.float32)
    base_resp_2 = np.zeros((len(all_subjects), args.n_images), dtype=np.float32)

    # Get the t-fMRI responses for the controlling and baseline images
    for s in range(len(all_subjects)):
        for ct in control_types:
            cv_resp_1[ct][s] = tfmri_1[s,controlling_images[ct][s]]
            cv_resp_2[ct][s] = tfmri_2[s,controlling_images[ct][s]]
        base_resp_1[s] = tfmri_1[s,baseline_images['time_window_1'][s]]
        base_resp_2[s] = tfmri_2[s,baseline_images['time_window_2'][s]]


# =============================================================================
# Neural control statistical significance (only for cv==1)
# =============================================================================
# For each time window, compute the significance of the difference in
# univariate t-fMRI responses between the two disentangling control conditions
# (high_1_low_2 vs low_1_high_2). For each time window, the significance is
# computed on responses of all subjects for all images from the same neural
# control condition, using a paired samples t-test.

if args.cv == 1:

    # All subjects
    # Time window 1
    p_val_tw_1_all_sub = ttest_rel(cv_resp_1['high_1_low_2'].flatten(),
        cv_resp_1['low_1_high_2'].flatten(), alternative='greater')[1]
    # Time window 2
    p_val_tw_2_all_sub = ttest_rel(cv_resp_2['high_1_low_2'].flatten(),
        cv_resp_2['low_1_high_2'].flatten(), alternative='less')[1]

    # Single subjects
    p_val_tw_1_single_sub = np.zeros(len(all_subjects))
    p_val_tw_2_single_sub = np.zeros(len(all_subjects))
    for s in range(len(all_subjects)):
        # Time window 1
        p_val_tw_1_single_sub[s] = ttest_rel(cv_resp_1['high_1_low_2'][s],
            cv_resp_1['low_1_high_2'][s], alternative='greater')[1]
        # Time window 2
        p_val_tw_2_single_sub[s] = ttest_rel(cv_resp_2['high_1_low_2'][s],
            cv_resp_2['low_1_high_2'][s], alternative='less')[1]


# =============================================================================
# Save the results
# =============================================================================
if args.cv == 0:
    stats = {
        'times': times,
        'time_window_1_start': time_window_1_start,
        'time_window_1_end': time_window_1_end,
        'time_window_2_start': time_window_2_start,
        'time_window_2_end': time_window_2_end,
        'tfmri_1': tfmri_1,
        'tfmri_2': tfmri_2,
        'control_types': control_types,
        'controlling_images': controlling_images,
        'baseline_images': baseline_images,
        }

elif args.cv == 1:
    stats = {
        'times': times,
        'time_window_1_start': time_window_1_start,
        'time_window_1_end': time_window_1_end,
        'time_window_2_start': time_window_2_start,
        'time_window_2_end': time_window_2_end,
        'tfmri_1': tfmri_1,
        'tfmri_2': tfmri_2,
        'control_types': control_types,
        'controlling_images': controlling_images,
        'baseline_images': baseline_images,
        'cv_resp_1': cv_resp_1,
        'cv_resp_2': cv_resp_2,
        'base_resp_1': base_resp_1,
        'base_resp_2': base_resp_2,
        'p_val_tw_1_all_sub': p_val_tw_1_all_sub,
        'p_val_tw_2_all_sub': p_val_tw_2_all_sub,
        'p_val_tw_1_single_sub': p_val_tw_1_single_sub,
        'p_val_tw_2_single_sub': p_val_tw_2_single_sub
        }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), stats)