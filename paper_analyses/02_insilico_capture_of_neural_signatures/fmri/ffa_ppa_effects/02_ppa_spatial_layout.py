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
n_iter : int
    Amount of iterations for creating the null and confidence interval
    distributions.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.utils import resample
from scipy.stats import binom

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
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
rng = np.random.default_rng(seed=seed)


# =============================================================================
# Load the in silico fMRI responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'insilico_fmri_responses',
    args.encoding_model, 'insilico_fmri_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

insilico_fmri = {}
insilico_fmri['PPA'] = {}
insilico_fmri['PPA']['lh'] = data['lh_insilico_fmri_ppa']['ppa_spatial_layout']
insilico_fmri['PPA']['rh'] = data['rh_insilico_fmri_ppa']['ppa_spatial_layout']
metadata = data['metadata']
del data


# =============================================================================
# Retain vertices that have above threshold NCSNR
# =============================================================================
# Loop across hemispheres
hemispheres = ['lh', 'rh']
n_images = {}
n_sub = len(metadata)
for hem in hemispheres:

    # Loop across subjects
    for s in range(len(metadata)):

        # Get the vertex indices of the ROI of interest
        roi_idx = metadata[s]['fmri'][hem+'_fsaverage_rois']['PPA']

        # NCSNR vertex selection
        ncsnr = metadata[s]['fmri'][hem+'_ncsnr'][roi_idx]
        idx_nan = ncsnr < args.ncsnr_threshold

        # Loop across image types and threshold vertices
        for key in insilico_fmri['PPA'][hem].keys():
            insilico_fmri['PPA'][hem][key][s][:,idx_nan] = np.nan
            n_images[key] = insilico_fmri['PPA'][hem][key][s].shape[0]


# =============================================================================
# Get the mean PPA response across vertices for each image type
# =============================================================================
# Empty results dictionary
vertex_mean_resp = {}

# Loop across image types
image_types = ['Objects-MultFurniture','Objects-SingleFurniture',
    'Scenes-EmptyRooms', 'Scenes-Rearranged']
vertex_mean_resp = {}
for itype in image_types:

    # Empty arrays of shape (n_subjects, n_images)
    vertex_mean_resp[itype] = np.zeros((n_sub, n_images[itype]),
        dtype=np.float32)

    # Loop across subjects
    for s in range(len(metadata)):

        # Loop across hemispheres
        for h, hem in enumerate(hemispheres):

            # Get the vertex responses for PPA
            if h == 0:
                vertex_mean_resp_sub = insilico_fmri[hem][itype][s]
            else:
                vertex_mean_resp_sub = np.append(vertex_mean_resp_sub,
                    insilico_fmri[hem][itype][s], 1)

        # Compute the mean ROI response across vertices
        vertex_mean_resp[itype][s] = np.nanmean(vertex_mean_resp_sub, 1)
        del vertex_mean_resp_sub


# =============================================================================
# Compute significant difference between responses for different image types
# =============================================================================
# Get the contrast conditions
contrasts = {
    'Scenes-EmptyRooms>Scenes-Rearranged':
        ['Scenes-EmptyRooms', 'Scenes-Rearranged'],
    'Scenes-EmptyRooms>Objects-SingleFurniture':
        ['Scenes-EmptyRooms', 'Objects-SingleFurniture'],
    'Scenes-EmptyRooms>Objects-MultFurniture':
        ['Scenes-EmptyRooms', 'Objects-MultFurniture']
    }

# Get the actual difference between the in silico fMRI responses of the
# different conditions
diff = {}
for key in contrasts.keys():
    cond1 = contrasts[key][0]
    cond2 = contrasts[key][1]
    diff[key] = np.mean(vertex_mean_resp[cond1], 1) - \
        np.mean(vertex_mean_resp[cond2], 1)

# Define the permutation indices
n_images = n_images['Scenes-EmptyRooms']
idx_perm_1 = np.zeros((args.n_iter, n_images), dtype=np.int32)
idx_perm_2 = np.zeros((args.n_iter, n_images), dtype=np.int32)
for i in range(args.n_iter):
    idx = rng.choice(n_images*2, size=n_images*2, replace=False)
    idx_perm_1[i] = idx[:n_images]
    idx_perm_2[i] = idx[n_images:]
    del idx

# Create the null distributions using permuted in silico fMRI responses
diff_null_dist = {}
for key in contrasts.keys():
    cond1 = contrasts[key][0]
    cond2 = contrasts[key][1]
    diff_null_dist[key] = np.zeros((n_sub, args.n_iter), dtype=np.float32)
    data = np.append(vertex_mean_resp[cond1],
        vertex_mean_resp[cond2], 1)
    for i in range(args.n_iter):
        diff_null_dist[key][:,i] = np.mean(data[:,idx_perm_1[i]], 1) - \
            np.mean(data[:,idx_perm_2[i]], 1)

# Compute the within-subject significance
within_subj_sig = {}
for key in contrasts.keys():
    within_subj_sig[key] = np.zeros((n_sub))
    for s in range(n_sub):
        within_subj_pval = \
            (np.sum(diff_null_dist[key][s] >= diff[key][s]) + 1) / \
            (args.n_iter + 1)
        within_subj_sig[key][s] = within_subj_pval < 0.01

# Compute the population-level significance based on population
# prevalence: this is the probability of observing k or more
# significant results by chance, based on the CDF of the binomial
# distribution of within-subject significances.
pval_diff = {}
for key in contrasts.keys():
    n = n_sub # Total number of subjects
    p = 0.05 # probability of success in each trial
    k = sum(within_subj_sig[key]) # Number of significant subjects
    # We use "k-1" because otherwise we would get the probability of observing
    # k+1 or more significant results by chance
    pval_diff[key] = 1 - binom.cdf(k-1, n, p)


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

save_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'ppa_spatial_layout.npy'

np.save(os.path.join(save_dir, file_name), results)