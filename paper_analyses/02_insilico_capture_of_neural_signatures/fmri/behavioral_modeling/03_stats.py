"""Compute the difference in RSA scores between high-level visual areas
(ventral, lateral, and dorsal streams) and early visual cortex.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
nsd_dir : str
	Directory of the Natural Scenes Dataset (NSD).
	https://naturalscenesdataset.org/
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Initiate BERG, empty results lists, and loop across fMRI subjects
# =============================================================================

metadata = []
lh_rsa = []
rh_rsa = []
diff_rsa_high_early = []

for s, sub in enumerate(tqdm(args.fmri_subjects)):


# =============================================================================
# Load the RSA results
# =============================================================================
    data_dir = os.path.join(args.berg_dir,
        'insilico_capture_of_neural_signatures', 'fmri', 'behavioral_modeling',
        'rsa', args.encoding_model)

    lh_data = np.load(os.path.join(data_dir, f'rsa_sub-{sub:02d}_lh.npy'),
        allow_pickle=True).item()
    rh_data = np.load(os.path.join(data_dir, f'rsa_sub-{sub:02d}_rh.npy'),
        allow_pickle=True).item()

    lh_rsa.append(lh_data['rsa'])
    rh_rsa.append(rh_data['rsa'])
    metadata.append(lh_data['metadata'])


# =============================================================================
# Vertex thresholding
# =============================================================================
    # Only retain vertices that have above threshold NCSNR
    # Left hemisphere
    lh_idx = metadata[s]['fmri']['lh_ncsnr'] >= \
        args.ncsnr_threshold
    rh_idx = metadata[s]['fmri']['rh_ncsnr'] >= \
        args.ncsnr_threshold

    # Threshold the retinotopic maps
    lh = lh_rsa[s]
    lh[~lh_idx] = np.nan
    rh = rh_rsa[s]
    rh[~rh_idx] = np.nan


# =============================================================================
# Compute the differences in RSA scores between high and early visual areas
# =============================================================================
    # Get the mean RSA scores for early visual areas
    early_visual = np.nanmean(np.append(
        lh[metadata[s]['fmri']['lh_fsaverage_rois']['early']],
        rh[metadata[s]['fmri']['rh_fsaverage_rois']['early']]))

    # Get the RSA scores for high-level visual areas (ventral, lateral, and
    # dorsal streams)
    lh_stream_idx = np.zeros(len(lh), dtype=int)
    rh_stream_idx = np.zeros(len(rh), dtype=int)
    streams = ['ventral', 'lateral', 'parietal']
    for stream in streams:
        lh_stream_idx[metadata[s]['fmri']['lh_fsaverage_rois'][stream]] = 1
        rh_stream_idx[metadata[s]['fmri']['rh_fsaverage_rois'][stream]] = 1
    lh_stream_idx = lh_stream_idx == 1
    rh_stream_idx = rh_stream_idx == 1
    high_visual = np.nanmean(np.append(lh[lh_stream_idx], rh[rh_stream_idx]))

    # Compute the difference in RSA scores between high and early visual areas
    diff_rsa_high_early.append(high_visual - early_visual)


# =============================================================================
# Compute the significance
# =============================================================================
p_val_diff_rsa_high_early = ttest_1samp(diff_rsa_high_early, 0,
    alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_rsa': lh_rsa,
    'rh_rsa': rh_rsa,
    'diff_rsa_high_early': diff_rsa_high_early,
    'p_val_diff_rsa_high_early': p_val_diff_rsa_high_early,
    'metadata': metadata
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'behavioral_modeling', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.save(os.path.join(save_dir, 'stats.npy'), results)