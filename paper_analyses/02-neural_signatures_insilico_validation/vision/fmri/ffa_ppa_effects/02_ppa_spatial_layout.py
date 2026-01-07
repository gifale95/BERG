"""Test the sensitivity to spatial layout of PPA. The hypothesis is that images
of empty rooms will drive PPA more than images of the same room surfaces
rearranged, and more than single or multiple pieces of furniture.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
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
import random
from sklearn.utils import resample
from scipy.stats import ttest_rel

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> PPA spatial layout <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in silico fMRI responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa_effects', 'insilico_fmri_responses',
    args.encoding_model, 'insilico_fmri_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

insilico_fmri = {}
insilico_fmri['PPA'] = {}
insilico_fmri['PPA']['lh'] = data['lh_insilico_fmri_ppa']['ppa_spatial_layout']
insilico_fmri['PPA']['rh'] = data['rh_insilico_fmri_ppa']['ppa_spatial_layout']
metadata = data['metadata']
del data


# =============================================================================
# NCSNR and prediction accuracy thresholding
# =============================================================================
# Retain vertices that have above threshold (i) NCSNR AND (ii) encoding
# prediction accuracy.

# Loop across hemispheres
hemispheres = ['lh', 'rh']
for hem in hemispheres:

    # Loop across subjects
    for s in range(len(metadata)):

        # Get the vertex indices of the ROI of interest
        roi_idx = metadata[s]['fmri'][hem+'_fsaverage_rois']['PPA']

        # NCSNR and noise ceiling vertex selection
        ncsnr = metadata[s]['fmri'][hem+'_ncsnr'][roi_idx]
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = metadata[s]['encoding_models']\
            [hem+'_explained_variance_nsdcore'][roi_idx]
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

        # Loop across image types and threshold vertices
        for key in insilico_fmri['PPA'][hem].keys():
            insilico_fmri['PPA'][hem][key][s][idx_nan] = np.nan


# =============================================================================
# Get the mean ROI response across vertices for each image type
# =============================================================================
# Empty results dictionary
vertex_mean_resp = {}

# Loop across image types
image_types = ['Objects-MultFurniture','Objects-SingleFurniture',
    'Scenes-EmptyRooms', 'Scenes-Rearranged']
for itype in image_types:

    # Empty arrays of shape (n_subjects,)
    vertex_mean_resp[itype] = np.zeros((len(metadata)))

    # Loop across subjects
    for s in range(len(metadata)):

        # Empty subject response list
        vertex_mean_resp_sub = []

        # Loop across hemispheres
        for hem in hemispheres:

            # Get the vertex responses for PPA
            vertex_mean_resp_sub.append(
                insilico_fmri['PPA'][hem][itype][s])

        # Compute the mean ROI response across vertices
        vertex_mean_resp[itype][s] = np.nanmean(np.concatenate(
            vertex_mean_resp_sub))


# =============================================================================
# Compute significant difference between responses for different image types
# =============================================================================
# Empty results dictionary
pval_diff = {}

# Scenes-EmptyRooms > Scenes-Rearranged
pval_diff['Scenes-EmptyRooms>Scenes-Rearranged'] = ttest_rel(
    vertex_mean_resp['Scenes-EmptyRooms'],
    vertex_mean_resp['Scenes-Rearranged'],
    alternative='greater')[1]

# Scenes-EmptyRooms > Objects-SingleFurniture
pval_diff['Scenes-EmptyRooms>Objects-SingleFurniture'] = ttest_rel(
    vertex_mean_resp['Scenes-EmptyRooms'],
    vertex_mean_resp['Objects-SingleFurniture'],
    alternative='greater')[1]

# Scenes-EmptyRooms > Objects-MultFurniture
pval_diff['Scenes-EmptyRooms>Objects-MultFurniture'] = ttest_rel(
    vertex_mean_resp['Scenes-EmptyRooms'],
    vertex_mean_resp['Objects-MultFurniture'],
    alternative='greater')[1]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Empty result variables
ci_vertex_mean_resp = {}
dist = {}
for itype in image_types:
    ci_vertex_mean_resp[itype] = np.zeros((2))
    dist[itype] = np.zeros((args.n_iter))

# Create the bootstrap distribution
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(metadata)))
    for itype in image_types:
        dist[itype][i] = np.mean(vertex_mean_resp[itype][idx])

# Compute the CIs from the bootstrap distribution
for itype in image_types:
    ci_vertex_mean_resp[itype][0] = np.percentile(dist[itype], 2.5)
    ci_vertex_mean_resp[itype][1] = np.percentile(dist[itype], 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'vertex_mean_resp': vertex_mean_resp,
    'pval_diff': pval_diff,
    'ci_vertex_mean_resp': ci_vertex_mean_resp,
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa_effects', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'ppa_spatial_layout.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore