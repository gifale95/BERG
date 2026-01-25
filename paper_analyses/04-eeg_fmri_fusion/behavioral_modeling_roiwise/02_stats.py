"""Aggregate the behavioral modeling RSA scores across fMRI subjects and
hemispheres, and compute the ROI-wise RSA scores.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
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

# Load the EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']

# Analysis parameters
n_fsub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = len(times)
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']


# =============================================================================
# Load the ROI-wise RSA results
# =============================================================================
rsa_roi = {}
metadata = []

# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'behavioral_modeling_roiwise', 'rsa', f'rsa_fmri_sub-{fsub:02d}.npy')

    data = np.load(data_dir, allow_pickle=True).item()

    metadata.append(data['metadata_fmri'])

    for roi in data['rsa_roi'].keys():
        if fs == 0:
            rsa_roi[roi] = np.zeros((n_fsub, n_time))
        rsa_roi[roi][fs] = data['rsa_roi'][roi]


# =============================================================================
# Bootstrap the confidence intervals (ROI-wise in silico fMRI vs t-fMRI
# correlation scores)
# =============================================================================
ci_rsa_roi = {}
ci_rsa_roi_peak_lat = {}

for key, val in tqdm(rsa_roi.items()):

    ci_rsa_roi[key] = np.zeros((2, n_time))
    ci_rsa_roi_peak_lat[key] = np.zeros((2))
    corr_dist = np.zeros((args.n_iter, n_time))
    peak_lat_dist = np.zeros((args.n_iter))

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.fmri_subjects)))
        corr_dist[i] = np.mean(val[idx], 0)
        peak_lat_dist[i] = times[np.argmax(np.mean(val[idx], 0))]

    ci_rsa_roi[key][0] = np.percentile(corr_dist, 2.5, axis=0)
    ci_rsa_roi[key][1] = np.percentile(corr_dist, 97.5, axis=0)
    ci_rsa_roi_peak_lat[key][0] = np.percentile(peak_lat_dist, 2.5)
    ci_rsa_roi_peak_lat[key][1] = np.percentile(peak_lat_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'times': times,
    'rsa_roi': rsa_roi,
    'ci_rsa_roi': ci_rsa_roi,
    'ci_rsa_roi_peak_lat': ci_rsa_roi_peak_lat
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling_roiwise', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)