"""Test whether the controlling images found using the in silico fMRI responses
of the train subjects generalize to the in silico fMRI responses of the
left-out subject. Stats include confidence intervals and significance.

Parameters
----------
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
roi_pair : str
    Used pairwise ROI combination.
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
n_categories: int
    Number of retained image categories.
n_exemplars: int
    Number of retained image exemplars for each category and neural control
    condition.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--roi_pair', default='V1-ventral', type=str)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--n_categories', default=10, type=int)
parser.add_argument('--n_exemplars', default=4, type=int)
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
# Get the total dataset subjects
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]
rois = [roi_1, roi_2]


# =============================================================================
# Load the univariate RNC baseline images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'baseline', f'cv-{args.cv}')
baseline_images = {}

for roi in rois:

    if args.cv == 0:
        file_name = f'baseline_roi-{args.roi}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        baseline_images[roi] = data_dict['baseline_img']

    elif args.cv == 1:
        baseline_images[roi] = []
        for s, sub in enumerate(all_subjects):
            file_name = f'baseline_cv_subject-{sub}_roi-{args.roi}.npy'
            data_dict = np.load(os.path.join(data_dir, file_name),
                allow_pickle=True).item()
            baseline_images[roi].append(data_dict['baseline_img'])


# =============================================================================
# Load the univariate RNC controlling images
# =============================================================================
control_types = ['high_1_high_2', 'low_1_low_2', 'high_1_low_2',
    'low_1_high_2']
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'quantitative_results',
    f'cv-{args.cv}')

if args.cv == 0:
    file_name = f'image_ranking_roi-{args.roi_pair}.npy'
    data_dict = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    controlling_images = data_dict['controlling_images']

elif args.cv == 1:
    controlling_images = []
    for s in all_subjects:
        file_name = f'image_ranking_cv_subject-{s:02d}_roi-{args.roi_pair}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        controlling_images.append(data_dict['controlling_images'])


# =============================================================================
# Load the fMRI responses of all subjects for the ILSVRC-2012 images
# =============================================================================
fmri = {}
for roi in rois:

    fmri_roi = []
    data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
        'fmri_responses')
    for sub in all_subjects:
        file_name = (f'fmri_sub-{sub:02d}_roi-{roi}_imagenet_split-'
            f'{args.imagenet_split}.npy')
        fmri_roi.append(np.load(os.path.join(data_dir, file_name)))
    fmri[roi] = np.array(fmri_roi)
    del fmri_roi


# =============================================================================
# Validate the neural control conditions across subjects (only for cv==1) # !!!
# =============================================================================
# Get the test subject univariate t-fMRI responses for the controlling
# images from the four neural control conditions, as well as for the baseline
# images.

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
# Correlate the in silico univariate fMRI responses of thw two ROIs, across
# all images (only for cv==1)
# =============================================================================
elif args.cv == 1:

    roi_pair_corr = np.zeros((len(all_subjects)))

    for s in range(len(all_subjects)):
        roi_pair_corr[s] = pearsonr(fmri[roi][s], fmri[roi][s])[0]


# =============================================================================
# Save the results
# =============================================================================
if args.cv == 0:
    stats = {
        'fmri': fmri,
        'control_types': control_types,
        'controlling_images': controlling_images,
        'baseline_images': baseline_images
        }

elif args.cv == 1: # !!!
    stats = {
        'fmri': fmri,
        'control_types': control_types,
        'controlling_images': controlling_images,
        'baseline_images': baseline_images,
        'cv_resp_1': cv_resp_1,
        'cv_resp_2': cv_resp_2,
        'base_resp_1': base_resp_1,
        'base_resp_2': base_resp_2,
        'roi_pair_corr': roi_pair_corr
        }

save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'stats', f'cv-{args.cv}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_{args.roi_pair}.npy'

np.save(os.path.join(save_dir, file_name), stats)