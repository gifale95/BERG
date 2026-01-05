"""Test the categorical selectivity of high-level visual cortex ROIs on in
silico fMRI responses.

Parameters
----------
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
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> HVC selectivity - Stats <<<')
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
    'vision', 'fmri', 'hvc_selectivity_univariate_responses',
    'insilico_fmri_responses', 'insilico_fmri_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

insilico_fmri = {}
insilico_fmri['lh'] = data['lh_insilico_fmri']
insilico_fmri['rh'] = data['rh_insilico_fmri']
metadata = data['metadata']
del data


# =============================================================================
# NCSNR and prediction accuracy thresholding
# =============================================================================
# Only retain vertices that have above threshold (i) NCSNR AND (ii) encoding
# prediction accuracy.

# Loop across hemispheres
hemispheres = ['lh', 'rh']
for hem in hemispheres:

    # Loop across subjects
    for s in range(len(metadata)):

        ncsnr = metadata[s]['fmri'][hem+'_ncsnr']
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = metadata[s]['encoding_models'][hem+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
        for key in insilico_fmri[hem].keys():
            insilico_fmri[hem][key][s,idx_nan] = np.nan


# =============================================================================
# Get the mean ROI response across vertices for each category
# =============================================================================
# Empty results dictionary
vertex_mean_resp = {}

# Loop across categories and ROIs
categories = ['Bodies', 'Faces', 'Objects', 'Scenes']
rois = ['EBA', 'FBA', 'FFA', 'OFA', 'PPA', 'OPA', 'RSC']
for c, cat in enumerate(categories):
    for roi in rois:

        # Empty arrays of shape (n_subjects,)
        vertex_mean_resp[roi+'_'+cat] = np.zeros((len(metadata)))

        # Loop across subjects
        for s in range(len(metadata)):

            # Empty subject response list
            vertex_mean_resp_sub = []

            # Loop across hemispheres
            for hem in hemispheres:

                # Get the vertex indices for the ROI
                if roi == 'FFA' or roi == 'FBA': # type: ignore
                    # Get the vertex indices for both parts of the ROI
                    idx = np.append(
                        metadata[s]['fmri'][hem+'_fsaverage_rois'][f'{roi}-1'], # type: ignore
                        metadata[s]['fmri'][hem+'_fsaverage_rois'][f'{roi}-2']) # type: ignore
                    idx.sort()
                else:
                    # Get the vertex indices for the ROI
                    idx = metadata[s]['fmri'][hem+'_fsaverage_rois'][roi] # type: ignore

                # Store the responses across selected vertices
                vertex_mean_resp_sub.append(insilico_fmri[hem][cat][s,idx])

            # Compute the mean ROI response across vertices
            vertex_mean_resp[roi+'_'+cat][s] = np.nanmean(np.concatenate(
                vertex_mean_resp_sub))


# =============================================================================
# Compute significant difference between responses for different categories
# =============================================================================
# Empty results dictionary
pval_cat_diff = {}

# Loop across target categories and ROIs
rois = [['EBA', 'FBA'], ['FFA', 'OFA'], [], ['PPA', 'OPA', 'RSC']]
for c, cat in enumerate(categories):
    for roi in rois[c]:

        # Loop across non-target categories
        other_cat = [item for item in categories if item != cat]
        for oc, ocat in enumerate(other_cat):

            # Compute the significance
            pval_cat_diff[roi+'_'+cat+'_>'+ocat] = \
                ttest_rel(vertex_mean_resp[roi+'_'+cat],
                vertex_mean_resp[roi+'_'+ocat], alternative='greater')[1]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Empty result variables
rois = ['EBA', 'FBA', 'FFA', 'OFA', 'PPA', 'OPA', 'RSC']
ci_vertex_mean_resp = {}
dist = {}
for roi in rois:
    for cat in categories:
        ci_vertex_mean_resp[roi+'_'+cat] = np.zeros((2))
        dist[roi+'_'+cat] = np.zeros((args.n_iter))

# Create the bootstrap distribution
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(metadata)))
    for roi in rois:
        for cat in categories:
            dist[roi+'_'+cat][i] = np.mean(vertex_mean_resp[roi+'_'+cat][idx])

# Compute the CIs from the bootstrap distribution
for roi in rois:
    for cat in categories:
        ci_vertex_mean_resp[roi+'_'+cat][0] = \
            np.percentile(dist[roi+'_'+cat], 2.5)
        ci_vertex_mean_resp[roi+'_'+cat][1] = \
            np.percentile(dist[roi+'_'+cat], 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'insilico_fmri': insilico_fmri,
    'vertex_mean_resp': vertex_mean_resp,
    'pval_cat_diff': pval_cat_diff,
    'ci_vertex_mean_resp': ci_vertex_mean_resp,
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'hvc_selectivity_univariate_responses', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore