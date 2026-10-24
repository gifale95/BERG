"""Load the t-fMRI prediction accuracies, average the vertex results within
selected ROIs, and compute the ROI-wise confidence intervals and significance.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'all' (all trials), 'even' (even trials), and 'odd' (odd
    trials).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution and for the prediction accuracy null distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--eeg_train_trials', default='all', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result arrays
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# LOnly select time points between 0-400 ms
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']
t_idx = np.intersect1d(np.where(times >= 0)[0], np.where(times <= 0.4)[0])
times_new = times[t_idx]

# Analysis parameters
metadata = []
n_fsub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times_new)
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA']

# Empty correlation array of shape:
# (N fMRI subjects, 2 hemispheres, 163842 fMRI vertices, N EEG time points)
corr_tfmri_fmri = np.zeros((n_fsub, n_hemi, n_vertex, n_time),
    dtype=np.float32)
corr_tfmri_fmri[:] = np.nan


# =============================================================================
# Load the correlation results
# =============================================================================
# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=fsub
        ))

    # Loop across fMRI hemispheres
    for h, hemi in enumerate(args.hemispheres):

        # Only select vertices falling within the NSD visual streams
        idx_v = np.zeros(n_vertex, dtype=int)
        streams = ['early', 'midventral', 'midlateral', 'midparietal',
            'ventral', 'lateral', 'parietal']
        for stream in streams:
            idx_v[metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
        idx_v = np.where(idx_v == 1)[0]

        # Load and store the correlation scores
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'encoding_fusion_accuracy')
        file_name = (f'corr_fmri_sub-{fsub:02d}_hemi-{hemi}_eeg_train_trials-'
            f'{args.eeg_train_trials}.npy')
        corr_tfmri_fmri[fs,h,idx_v] = np.load(os.path.join(
            data_dir, file_name))[:,t_idx]

        # NCSNR vertex selection
        ncsnr = metadata[fs]['fmri'][hemi+'_ncsnr']
        idx_nan = ncsnr < args.ncsnr_threshold
        corr_tfmri_fmri[fs,h,idx_nan] = np.nan


# =============================================================================
# Get the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# Empty result dictionary
corr_tfmri_fmri_roi = {}

# Loop across ROIs
for r, roi in enumerate(rois):

    # Empty ROI correlation array of shape:
    # (N fMRI subjects, 140 EEG time points)
    corr_tfmri_fmri_roi[roi] = np.zeros((len(args.fmri_subjects), n_time),
        dtype=np.float32)

    # Loop across subjects and hemispheres
    for fs, fsub in enumerate(args.fmri_subjects):
        for h, hemi in enumerate(args.hemispheres):

            # Get the indices of the ROI vertices
            if roi in ['V1', 'V2', 'V3']:
                idx_roi = np.append(
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
                idx_roi.sort()
            elif roi in ['FFA', 'VWFA', 'FBA']:
                idx_roi = np.append(
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
                idx_roi.sort()
            elif roi in ['intermediate']:
                idx_roi = np.append(
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
                idx_roi = np.append(idx_roi,
                    metadata[fs]['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
                idx_roi.sort()
            else:
                idx_roi = metadata[fs]['fmri'][f'{hemi}_fsaverage_rois'][roi]

            # Get the correlation scores of the above NCSNR threshold vertices
            # of the chosen ROI
            if h == 0:
                corr = corr_tfmri_fmri[fs,h,idx_roi]
            else:
                corr = np.append(corr, corr_tfmri_fmri[fs,h,idx_roi], 0)
        
        # Store the mean correlation across ROI vertices
        corr_tfmri_fmri_roi[roi][fs] = np.nanmean(corr, 0)
        del corr


# =============================================================================
# Bootstrap the confidence intervals (ROI-wise in silico fMRI vs t-fMRI
# correlation scores)
# =============================================================================
ci_corr_tfmri_fmri_roi = {}
ci_corr_tfmri_fmri_roi_peak_lat = {}

for key, val in tqdm(corr_tfmri_fmri_roi.items()):

    ci_corr_tfmri_fmri_roi[key] = np.zeros((2, n_time))
    ci_corr_tfmri_fmri_roi_peak_lat[key] = np.zeros((2))
    corr_dist = np.zeros((args.n_iter, n_time))
    peak_lat_dist = np.zeros((args.n_iter))

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.fmri_subjects)))
        corr_dist[i] = np.mean(val[idx], 0)
        peak_lat_dist[i] = times_new[np.argmax(np.mean(val[idx], 0))]

    ci_corr_tfmri_fmri_roi[key][0] = np.percentile(corr_dist, 2.5, axis=0)
    ci_corr_tfmri_fmri_roi[key][1] = np.percentile(corr_dist, 97.5, axis=0)
    ci_corr_tfmri_fmri_roi_peak_lat[key][0] = np.percentile(peak_lat_dist, 2.5)
    ci_corr_tfmri_fmri_roi_peak_lat[key][1] = np.percentile(peak_lat_dist, 97.5)


# =============================================================================
# Load the ROI-wise prediction null distribution results
# =============================================================================
# Empty null distribution arrays of shape:
# (8 subjects, N EEG time points, N iterations)
null_dist = {}
for roi in rois:
    null_dist[roi] = np.zeros((len(args.fmri_subjects), len(times_new),
        args.n_iter), dtype=np.float32)

# Loop across fMRI subjects, ROIs, and hemispheres
for fs, fsub in enumerate(args.fmri_subjects):
    for roi in rois:
        for h, hemi in enumerate(args.hemispheres):

            # Load the null distribution
            data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
                'encoding_fusion_accuracy_null_distribution')
            file_name = (f'corr_null_dist_fmri_sub-{fsub:02d}_hemi-{hemi}_roi-'
                f'{roi}_eeg_train_trials-{args.eeg_train_trials}.npy')

            if h == 0:
                null_dist_hemi = np.load(os.path.join(data_dir, file_name))
            else:
                null_dist_hemi = np.append(null_dist_hemi,
                    np.load(os.path.join(data_dir, file_name)), 0)

        # Store the mean null distribution across hemispheres
        null_dist[roi][fs] = np.mean(null_dist_hemi, 0)[t_idx]


# =============================================================================
# Compute the ROI-wise significance
# =============================================================================
# Empty p-value and significance arrays of shape:
# (8 subjects, N EEG time points)
within_subj_pval = {}
within_subj_sig = {}
for roi in rois:
    within_subj_pval[roi] = np.zeros((len(args.fmri_subjects), len(times_new)))
    within_subj_sig[roi] = np.zeros((len(args.fmri_subjects), len(times_new)))

# Compute the within-subject ROI-wise p-values based on the null distribution
# permutations tests
for roi in rois:
    for fs in range(len(args.fmri_subjects)):
        for t in range(len(times_new)):
            within_subj_pval[roi][fs,t] = \
                (np.sum(null_dist[roi][fs,t] >= corr_tfmri_fmri_roi[roi][fs,t]) + 1) / \
                (args.n_iter + 1)

# Multiple comparison correction across time points
for roi in rois:
    for fs in range(len(args.fmri_subjects)):
        within_subj_sig[roi][fs] = multipletests(within_subj_pval[roi][fs],
            0.05, 'fdr_bh')[0]

# Compute the between-subject significance based on population prevalence:
# this is the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject
# significances.
n = len(args.fmri_subjects) # Total number of subjects
p = 0.05 # probability of success in each trial
between_subj_pval = {}
for roi in rois:
    between_subj_pval[roi] = np.zeros((len(times_new)))
    for t in range(len(times_new)):
        k = sum(within_subj_sig[roi][:,t]) # Number of significant subjects
        # We use "k-1" because otherwise we would get the probability of observing
        # k+1 or more significant results by chance
        between_subj_pval[roi][t] = 1 - binom.cdf(k-1, n, p)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'times': times,
    'times_new': times_new,
    'corr_tfmri_fmri': corr_tfmri_fmri,
    'corr_tfmri_fmri_roi': corr_tfmri_fmri_roi,
    'ci_corr_tfmri_fmri_roi': ci_corr_tfmri_fmri_roi,
    'ci_corr_tfmri_fmri_roi_peak_lat': ci_corr_tfmri_fmri_roi_peak_lat,
    'between_subj_pval': between_subj_pval
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_eeg_train_trials-{args.eeg_train_trials}.npy'

np.save(os.path.join(save_dir, file_name), results)