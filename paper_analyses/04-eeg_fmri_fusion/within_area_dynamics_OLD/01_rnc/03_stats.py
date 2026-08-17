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
from tqdm import tqdm
from berg import BERG
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--roi', default='hV4', type=str)
parser.add_argument('--time_window_pair', default='0.10-0.15__0.20-0.25', type=str)
parser.add_argument('--n_images', default=100, type=int)
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
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'tfmri_responses')
for sub in all_subjects:
    file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}.h5'
    tfmri.append(h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'])
tfmri = np.array(tfmri)


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
tfmri_1 = np.mean(tfmri[:,:,t_min_1:t_max_1], 2)
tfmri_2 = np.mean(tfmri[:,:,t_min_2:t_max_2], 2)


# =============================================================================
# Load the univariate RNC controlling images
# =============================================================================
control_types = ['high_1_high_2', 'low_1_low_2', 'high_1_low_2',
    'low_1_high_2']
controlling_images = {}

data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_results', f'cv-{args.cv}', args.time_window_pair)

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
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc', 'baseline',
    f'cv-{args.cv}', args.time_window_pair)
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
# Compute the 95% confidence intervals (only for cv==1)
# =============================================================================
# Compute the confidence intervals of the cross-validated t-fMRI univariate
# responses for the controlling images (averaged across the N best
# controlling images), across the 8 NSD subjects.

if args.cv == 1:

    # CI arrays of shape:
    # (2 CI percentiles)
    ci_cv_resp_1 = {}
    ci_cv_resp_2 = {}
    for ct in control_types:
        ci_cv_resp_1[ct] = np.zeros((2))
        ci_cv_resp_2[ct] = np.zeros((2))
    ci_base_resp_1 = np.zeros((2))
    ci_base_resp_2 = np.zeros((2))

    # Empty CI distribution arrays
    resp_dist_1 = {}
    resp_dist_2 = {}
    for ct in control_types:
        resp_dist_1[ct] = np.zeros((args.n_iter), dtype=np.float32)
        resp_dist_2[ct] = np.zeros((args.n_iter), dtype=np.float32)
    base_dist_1 = np.zeros((args.n_iter), dtype=np.float32)
    base_dist_2 = np.zeros((args.n_iter), dtype=np.float32)

    # Compute the CI distributions
    for i in tqdm(range(args.n_iter)):
        idx_resample = resample(np.arange(len(all_subjects)))
        for ct in control_types:
            resp_dist_1[ct][i] = np.mean(cv_resp_1[ct][idx_resample])
            resp_dist_2[ct][i] = np.mean(cv_resp_2[ct][idx_resample])
        base_dist_1[i] = np.mean(base_resp_1[idx_resample])
        base_dist_2[i] = np.mean(base_resp_2[idx_resample])

    # Get the 5th and 95th CI distributions percentiles
    for ct in control_types:
        ci_cv_resp_1[ct][0] = np.percentile(resp_dist_1[ct], 2.5)
        ci_cv_resp_2[ct][0] = np.percentile(resp_dist_2[ct], 2.5)
        ci_cv_resp_1[ct][1] = np.percentile(resp_dist_1[ct], 97.5)
        ci_cv_resp_2[ct][1] = np.percentile(resp_dist_2[ct], 97.5)
    ci_base_resp_1[0] = np.percentile(base_dist_1, 2.5)
    ci_base_resp_2[0] = np.percentile(base_dist_2, 2.5)
    ci_base_resp_1[1] = np.percentile(base_dist_1, 97.5)
    ci_base_resp_2[1] = np.percentile(base_dist_2, 97.5)


# =============================================================================
# Compute the significance (only for cv==1)
# =============================================================================
# Compute whether the t-fMRI responses for the controlling images are
# significantly higher or lower than the t-fMRI responses for the baseline
# images, across the 8 NSD subjects, using paired-samples t-tests, and
# correcting for multiple comparisons across the four control conditions and
# two time windows.

if args.cv == 1:

    pval_1 = {}
    pval_2 = {}

    # 1st ranking: images with high univariate responses for both time windows
    pval_1['high_1_high_2'] = ttest_rel(np.mean(cv_resp_1['high_1_high_2'], 1),
        np.mean(base_resp_1, 1), alternative='greater')[1]
    pval_2['high_1_high_2'] = ttest_rel(np.mean(cv_resp_2['high_1_high_2'], 1),
        np.mean(base_resp_2, 1), alternative='greater')[1]

    # 2st ranking: images with low univariate responses for both time windows
    pval_1['low_1_low_2'] = ttest_rel(np.mean(cv_resp_1['low_1_low_2'], 1),
        np.mean(base_resp_1, 1), alternative='less')[1]
    pval_2['low_1_low_2'] = ttest_rel(np.mean(cv_resp_2['low_1_low_2'], 1),
        np.mean(base_resp_2, 1), alternative='less')[1]

    # 3rd ranking: images with high univariate responses for time window 1 and
    # low univariate responses for time window 2
    pval_1['high_1_low_2'] = ttest_rel(np.mean(cv_resp_1['high_1_low_2'], 1),
        np.mean(base_resp_1, 1), alternative='greater')[1]
    pval_2['high_1_low_2'] = ttest_rel(np.mean(cv_resp_2['high_1_low_2'], 1),
        np.mean(base_resp_2, 1), alternative='less')[1]

    # 4th ranking: images with low univariate responses for time window 1 and
    # high univariate responses for time window 2
    pval_1['low_1_high_2'] = ttest_rel(np.mean(cv_resp_1['low_1_high_2'], 1),
        np.mean(base_resp_1, 1), alternative='less')[1]
    pval_2['low_1_high_2'] = ttest_rel(np.mean(cv_resp_2['low_1_high_2'], 1),
        np.mean(base_resp_2, 1), alternative='greater')[1]

    # Correct for multiple comparisons
    pval_corrected_1 = {}
    pval_corrected_2 = {}
    sig_1 = {}
    sig_2 = {}
    pvals = np.append(np.array(list(pval_1.values())),
        np.array(list(pval_2.values())))
    sig, pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    pval_corrected_1['high_1_high_2'] = pvals[0]
    pval_corrected_1['low_1_low_2'] = pvals[1]
    pval_corrected_1['high_1_low_2'] = pvals[2]
    pval_corrected_1['low_1_high_2'] = pvals[3]
    pval_corrected_2['high_1_high_2'] = pvals[4]
    pval_corrected_2['low_1_low_2'] = pvals[5]
    pval_corrected_2['high_1_low_2'] = pvals[6]
    pval_corrected_2['low_1_high_2'] = pvals[7]
    sig_1['high_1_high_2'] = sig[0]
    sig_1['low_1_low_2'] = sig[1]
    sig_1['high_1_low_2'] = sig[2]
    sig_1['low_1_high_2'] = sig[3]
    sig_2['high_1_high_2'] = sig[4]
    sig_2['low_1_low_2'] = sig[5]
    sig_2['high_1_low_2'] = sig[6]
    sig_2['low_1_high_2'] = sig[7]


# =============================================================================
# Correlate the time window responses across all images
# =============================================================================
# This will provide the correlation scores between the t-fMRI
# univariate responses of the two time windows.

if args.cv == 0:
    time_window_pair_corr = pearsonr(np.mean(tfmri_1, 0),
        np.mean(tfmri_2, 0))[0]

elif args.cv == 1:
    # Correlation arrays of shape: (Subjects)
    time_window_pair_corr = np.zeros((len(all_subjects)))
    for s in range(len(all_subjects)):
        time_window_pair_corr[s] = pearsonr(tfmri_1[s], tfmri_2[s])[0]


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
        'time_window_pair_corr': time_window_pair_corr
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
        'ci_cv_resp_1': ci_cv_resp_1,
        'ci_cv_resp_2': ci_cv_resp_2,
        'ci_base_resp_1': ci_base_resp_1,
        'ci_base_resp_2': ci_base_resp_2,
        'pval_1': pval_1,
        'pval_2': pval_2,
        'pval_corrected_1': pval_corrected_1,
        'pval_corrected_2': pval_corrected_2,
        'sig_1': sig_1,
        'sig_2': sig_2,
        'time_window_pair_corr': time_window_pair_corr
        }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'stats', f'cv-{args.cv}', args.time_window_pair)
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), stats)