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
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--roi_pair', default='V1-ventral', type=str)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--n_categories', default=40, type=int)
parser.add_argument('--n_exemplars', default=4, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
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
# Load the univariate RNC baseline images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'baseline', f'cv-{args.cv}')
baseline_images = {}

for roi in rois:

    if args.cv == 0:
        file_name = f'baseline_roi-{roi}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        baseline_images[roi] = data_dict['baseline_img']

    elif args.cv == 1:
        baseline_images[roi] = []
        for s, sub in enumerate(all_subjects):
            file_name = f'baseline_cv_subject-{sub}_roi-{roi}.npy'
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
    file_name = f'image_ranking_{args.roi_pair}.npy'
    data_dict = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()
    controlling_images = data_dict['controlling_images']

elif args.cv == 1:
    controlling_images = []
    for s in all_subjects:
        file_name = f'image_ranking_cv_subject-{s:02d}_{args.roi_pair}.npy'
        data_dict = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        controlling_images.append(data_dict['controlling_images'])


# =============================================================================
# Load the in silico fMRI responses of all subjects for the ILSVRC-2012 images
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
# Validate the neural control conditions across subjects
# =============================================================================
# Get the test subject in silico univariate fMRI responses for the controlling
# images from the four neural control conditions, as well as for the baseline
# images.

if args.cv == 0:

    # Get the in silico univariate fMRI responses for the controlling images
    cv_resp_roi_1 = []
    cv_resp_roi_2 = []
    for s in range(len(all_subjects)):
        cv_resp_roi_1_sub = {}
        cv_resp_roi_2_sub = {}
        for key, val in controlling_images.items():
            cv_resp_roi_1_sub[key] = {}
            cv_resp_roi_2_sub[key] = {}
            for ct in control_types:
                cv_resp_roi_1_sub[key][ct] = \
                    fmri[roi_1][s,controlling_images[key][ct]]
                cv_resp_roi_2_sub[key][ct] = \
                    fmri[roi_2][s,controlling_images[key][ct]]
        cv_resp_roi_1.append(cv_resp_roi_1_sub)
        cv_resp_roi_2.append(cv_resp_roi_2_sub)
        del cv_resp_roi_1_sub, cv_resp_roi_2_sub

    # Get the in silico univariate fMRI responses for the baseline images
    base_resp = {}
    for roi in rois:
        base_resp[roi] = []
        for s in range(len(all_subjects)):
            base_resp[roi].append(fmri[roi][s,baseline_images[roi]])

elif args.cv == 1:

    # Get the in silico univariate fMRI responses for the controlling images
    cv_resp_roi_1 = []
    cv_resp_roi_2 = []
    for s in range(len(all_subjects)):
        cv_resp_roi_1_sub = {}
        cv_resp_roi_2_sub = {}
        for key, val in controlling_images[s].items():
            cv_resp_roi_1_sub[key] = {}
            cv_resp_roi_2_sub[key] = {}
            for ct in control_types:
                cv_resp_roi_1_sub[key][ct] = \
                    fmri[roi_1][s,controlling_images[s][key][ct]]
                cv_resp_roi_2_sub[key][ct] = \
                    fmri[roi_2][s,controlling_images[s][key][ct]]
        cv_resp_roi_1.append(cv_resp_roi_1_sub)
        cv_resp_roi_2.append(cv_resp_roi_2_sub)
        del cv_resp_roi_1_sub, cv_resp_roi_2_sub

    # Get the in silico univariate fMRI responses for the baseline images
    base_resp = {}
    for roi in rois:
        base_resp[roi] = []
        for s in range(len(all_subjects)):
            base_resp[roi].append(fmri[roi][s,baseline_images[roi][s]])


# =============================================================================
# Correlate the in silico univariate fMRI responses of the two ROIs, across
# all images
# =============================================================================
roi_pair_corr_all_img = np.zeros((len(all_subjects)))

for s in range(len(all_subjects)):
    roi_pair_corr_all_img[s] = pearsonr(fmri[roi_1][s], fmri[roi_2][s])[0]


# =============================================================================
# Correlate the in silico univariate fMRI responses of the two ROIs, across
# the controlling images (only for cv-0)
# =============================================================================
if args.cv == 0:

    roi_pair_corr_control_img = np.zeros((len(all_subjects)))

    for s in range(len(all_subjects)):
        idx = []
        for key in controlling_images.keys():
            for ct in control_types:
                idx.append(controlling_images[key][ct])
        idx = np.array(idx).flatten()
        idx.sort()
        roi_pair_corr_control_img[s] = pearsonr(fmri[roi_1][s][idx],
            fmri[roi_2][s][idx])[0]


# =============================================================================
# Filter the univariate resposnes of the images selected by Rotem
# =============================================================================
if args.cv == 0:

    # Kept images
    cat_rotem = {
        '01_Granny_Smith': {
            'high_1_high_2': np.array([0, 1, 3]),
            'high_1_low_2': np.array([0, 1, 3]),
            'low_1_high_2': np.array([0, 2, 3]),
            'low_1_low_2': np.array([1, 2, 3])
        },
        '04_lemon': {
            'high_1_high_2': np.array([0, 1, 2]),
            'high_1_low_2': np.array([0, 1, 3]),
            'low_1_high_2': np.array([0, 1, 3]),
            'low_1_low_2': np.array([0, 1, 2])
        },
        '08_butternut_squash': {
            'high_1_high_2': np.array([1, 2, 3]),
            'high_1_low_2': np.array([0, 1, 3]),
            'low_1_high_2': np.array([0, 1, 3]),
            'low_1_low_2': np.array([1, 2, 3])
        },
        '10_mountain_tent': {
            'high_1_high_2': np.array([1, 2, 3]),
            'high_1_low_2': np.array([0, 1, 3]),
            'low_1_high_2': np.array([0, 1, 2]),
            'low_1_low_2': np.array([0, 1, 2])
        },
        '16_pot': {
            'high_1_high_2': np.array([1, 2, 3]),
            'high_1_low_2': np.array([0, 1, 3]),
            'low_1_high_2': np.array([0, 1, 2]),
            'low_1_low_2': np.array([0, 2, 3])
        },
        '11_baseball': {
            'high_1_high_2': np.array([0, 1, 3]),
            'high_1_low_2': np.array([0, 1, 2]),
            'low_1_high_2': np.array([0, 1, 2]),
            'low_1_low_2': np.array([1, 2, 3])
        },
        '06_weasel': {
            'high_1_high_2': np.array([0, 2, 3]),
            'high_1_low_2': np.array([0, 1, 2]),
            'low_1_high_2': np.array([0, 1, 2]),
            'low_1_low_2': np.array([0, 1, 2])
        },
        '20_house_finch': {
            'high_1_high_2': np.array([0, 2, 3]),
            'high_1_low_2': np.array([0, 1, 2]),
            'low_1_high_2': np.array([0, 2, 3]),
            'low_1_low_2': np.array([0, 1, 2])
        },
        '17_hognose_snake': {
            'high_1_high_2': np.array([0, 2, 3]),
            'high_1_low_2': np.array([0, 1, 2]),
            'low_1_high_2': np.array([0, 1, 2]),
            'low_1_low_2': np.array([0, 1, 3])
        }
    }

    # Filter the univariate fMRI responses
    cv_resp_roi_1_rotem = []
    cv_resp_roi_2_rotem = []
    for s in range(len(all_subjects)):
        cv_resp_roi_1_sub = {}
        cv_resp_roi_2_sub = {}
        for key, val in cat_rotem.items():
            cv_resp_roi_1_sub[key] = {}
            cv_resp_roi_2_sub[key] = {}
            for ct in control_types:
                cv_resp_roi_1_sub[key][ct] = cv_resp_roi_1[s][key][ct][val[ct]]
                cv_resp_roi_2_sub[key][ct] = cv_resp_roi_2[s][key][ct][val[ct]]
        cv_resp_roi_1_rotem.append(cv_resp_roi_1_sub)
        cv_resp_roi_2_rotem.append(cv_resp_roi_2_sub)
        del cv_resp_roi_1_sub, cv_resp_roi_2_sub

    # Correlate the fMRI resposnes of the two ROIs for the selected images
    roi_pair_corr_control_img_rotem = np.zeros((len(all_subjects)))
    for s in range(len(all_subjects)):
        fmri_roi_1 = []
        fmri_roi_2 = []
        for key in cat_rotem.keys():
            for ct in control_types:
                fmri_roi_1.append(cv_resp_roi_1_rotem[s][key][ct])
                fmri_roi_2.append(cv_resp_roi_2_rotem[s][key][ct])
        fmri_roi_1 = np.array(fmri_roi_1).flatten()
        fmri_roi_2 = np.array(fmri_roi_2).flatten()
        roi_pair_corr_control_img_rotem[s] = pearsonr(fmri_roi_1, fmri_roi_2)[0]


# =============================================================================
# Save the results
# =============================================================================
stats = {
    'fmri': fmri,
    'control_types': control_types,
    'controlling_images': controlling_images,
    'baseline_images': baseline_images,
    'cv_resp_roi_1': cv_resp_roi_1,
    'cv_resp_roi_2': cv_resp_roi_2,
    'base_resp': base_resp,
    'roi_pair_corr_all_img': roi_pair_corr_all_img
    }

if args.cv == 0:
    stats['roi_pair_corr_control_img'] = roi_pair_corr_control_img
    stats['cv_resp_roi_1_rotem'] = cv_resp_roi_1_rotem
    stats['cv_resp_roi_2_rotem'] = cv_resp_roi_2_rotem
    stats['roi_pair_corr_control_img_rotem'] = roi_pair_corr_control_img_rotem

save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'stats', f'cv-{args.cv}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_{args.roi_pair}.npy'

np.save(os.path.join(save_dir, file_name), stats)