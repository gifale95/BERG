"""Apply univariate relational neural control (RNC) to find images that align
or disentangle the in silico fMRI responses of two ROIs.

Parameters
----------
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
    If cv==1, the left-out subject during cross-validation, out of the 8 NSD
    subjects.
roi_pair : str
    Used pairwise ROI combination.
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
n_categories: int
    Number of retained image categories.
n_exemplars: int
    Number of retained image exemplars for each category and neural control
    condition.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi_pair', default='V1-ventral', type=str)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--n_categories', default=10, type=int)
parser.add_argument('--n_exemplars', default=4, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Univariate RNC <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


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
# Load the univariante RNC baseline scores, and average them across images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'baseline', f'cv-{args.cv}')

base = {}
for roi in rois:

    if args.cv == 0:
        file_name = f'baseline_roi-{roi}.npy'
        data = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        base[roi] = np.mean(data['baseline_resp'])

    elif args.cv == 1:
        file_name = f'baseline_cv_subject-{args.cv_subject}_roi-{roi}.npy'
        data = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        base[roi] = np.mean(data['baseline_resp_train'])


# =============================================================================
# Load the fMRI responses for the ILSVRC-2012 images
# =============================================================================
fmri_mean = {}
for roi in rois:

    # Load the fMRI responses of all subjects
    fmri = []
    data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
        'fmri_responses')
    for sub in all_subjects:
        file_name = (f'fmri_sub-{sub:02d}_roi-{roi}_imagenet_split-'
            f'{args.imagenet_split}.npy')
        fmri.append(np.load(os.path.join(data_dir, file_name)))
    fmri = np.array(fmri)

    # If cross-validating, remove the CV (test) subject, and average over the
    # remaining (train) subjects. The fMRI responses for the train subjects
    # are used to select the controlling images, and the controlling images
    # will then be validated on the fMRI responses for the test subjects.
    if args.cv == 0:
        fmri_mean[roi] = np.mean(fmri, 0)
    elif args.cv == 1:
        fmri_mean[roi] = np.mean(np.delete(fmri, args.cv_subject-1, 0), 0)
    del fmri


# =============================================================================
# Rank the images based on their fMRI univariate responses
# =============================================================================
# Univariate response score margin used to constrain the selection of the
# control images
if args.cv == 1:
    margin = 0.15
elif args.cv == 0:
    margin = 0.15

# Rank the images based on their alignment of univariate fMRI responses of the
# two ROIs (i.e., that lead to both ROIs having either high or low univariate
# responses).
# 1st ranking: images with high univariate responses for both ROIs
response_sum = fmri_mean[roi_1] + fmri_mean[roi_2]
high_1_high_2_rank = np.argsort(np.argsort(
    response_sum)[::-1]).astype(np.float32)
# Ignore image conditions with fMRI responses below the baseline scores
idx_bad_1 = np.where(fmri_mean[roi_1] < base[roi_1]+margin)[0]
idx_bad_2 = np.where(fmri_mean[roi_2] < base[roi_2]+margin)[0]
high_1_high_2_rank[idx_bad_1] = np.nan
high_1_high_2_rank[idx_bad_2] = np.nan
# 2nd ranking: images with low univariate responses for both ROIs
low_1_low_2_rank = np.argsort(np.argsort(response_sum)).astype(np.float32)
# Ignore image conditions with fMRI responses above the baseline scores
idx_bad_1 = np.where(fmri_mean[roi_1] > base[roi_1]-margin)[0]
idx_bad_2 = np.where(fmri_mean[roi_2] > base[roi_2]-margin)[0]
low_1_low_2_rank[idx_bad_1] = np.nan
low_1_low_2_rank[idx_bad_2] = np.nan

# Rank the images based on their disentanglement of univariate fMRI responses
# of the two ROIs (i.e., that lead to one ROI having high responses and the
# other ROI having low responses, or vice versa).
# 3rd ranking: images with high univariate responses for ROI 1 and low
# univariate responses for ROI 2
response_diff = fmri_mean[roi_1] - fmri_mean[roi_2]
high_1_low_2_rank = np.argsort(np.argsort(
    response_diff)[::-1]).astype(np.float32)
# Ignore image conditions with univariate responses below (ROI 1) or above
# (ROI 2) the baseline scores
idx_bad_1 = np.where(fmri_mean[roi_1] < base[roi_1]+margin)[0]
idx_bad_2 = np.where(fmri_mean[roi_2] > base[roi_2]-margin)[0]
high_1_low_2_rank[idx_bad_1] = np.nan
high_1_low_2_rank[idx_bad_2] = np.nan
# 4th ranking: images with low univariate responses for ROI 1 and high
# univariate responses for ROI 2
low_1_high_2_rank = np.argsort(np.argsort(response_diff)).astype(np.float32)
# Ignore image conditions with univariate responses above (ROI 1) or below
# (ROI 2) the baseline scores
idx_bad_1 = np.where(fmri_mean[roi_1] > base[roi_1]-margin)[0]
idx_bad_2 = np.where(fmri_mean[roi_2] < base[roi_2]+margin)[0]
low_1_high_2_rank[idx_bad_1] = np.nan
low_1_high_2_rank[idx_bad_2] = np.nan


# =============================================================================
# Select the image categories that rank best across all four neural control
# conditions
# =============================================================================
# Get the image category labels
images = torchvision.datasets.ImageNet(root=args.imagenet_dir,
    split=args.imagenet_split)
targets = np.array(images.targets)
classes_white_spaces = images.classes
classes = [c[0].replace(" ", "_") for c in classes_white_spaces]

# Loop across categories
scores_high_1_high_2 = []
scores_low_1_low_2 = []
scores_high_1_low_2 = []
scores_low_1_high_2 = []
for cat in np.unique(targets):

    # Get the indices of the images from the current category
    idx_cat = np.where(targets == cat)[0]

    # Average the ranks across the best N image examplars from each category
    scores_high_1_high_2.append(np.mean(np.sort(
        high_1_high_2_rank[idx_cat])[:args.n_exemplars]))
    scores_low_1_low_2.append(np.mean(np.sort(
        low_1_low_2_rank[idx_cat])[:args.n_exemplars]))
    scores_high_1_low_2.append(np.mean(np.sort(
        high_1_low_2_rank[idx_cat])[:args.n_exemplars]))
    scores_low_1_high_2.append(np.mean(np.sort(
        low_1_high_2_rank[idx_cat])[:args.n_exemplars]))

# Sum the category scores across the four neural control conditions
scores_all = np.array(scores_high_1_high_2) + np.array(scores_low_1_low_2) + \
    np.array(scores_high_1_low_2) + np.array(scores_low_1_high_2)

# Select the N categories with lowest scores across all four neural control
# conditions
idx_cat_best = np.argsort(scores_all)[:args.n_categories]


# =============================================================================
# Select the image exemplars from the best categories
# =============================================================================
controlling_images = {}

for i, cat in enumerate(idx_cat_best):

    # Get the indices of the images from the current category
    idx_cat = np.where(targets == cat)[0]

    # Select the best N image exemplars from each category and neural control
    # condition
    controlling_images[f'{i+1:02d}_{classes[cat]}'] = {
        'high_1_high_2': idx_cat[np.argsort(
            high_1_high_2_rank[idx_cat])[:args.n_exemplars]],
        'low_1_low_2': idx_cat[np.argsort(
            low_1_low_2_rank[idx_cat])[:args.n_exemplars]],
        'high_1_low_2': idx_cat[np.argsort(
            high_1_low_2_rank[idx_cat])[:args.n_exemplars]],
        'low_1_high_2': idx_cat[np.argsort(
            low_1_high_2_rank[idx_cat])[:args.n_exemplars]]
    }


# =============================================================================
# Save the results
# =============================================================================
data_dict = {
    'roi_1': roi_1,
    'roi_2': roi_2,
    'controlling_images': controlling_images
}

save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'quantitative_results',
    f'cv-{args.cv}')
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = f'image_ranking_{args.roi_pair}.npy'
elif args.cv == 1:
    file_name = f'image_ranking_cv_subject-{args.cv_subject:02d}_{args.roi_pair}.npy'

np.save(os.path.join(save_dir, file_name), data_dict)