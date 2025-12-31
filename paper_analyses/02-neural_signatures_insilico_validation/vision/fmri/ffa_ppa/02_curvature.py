"""Test the curvature preferences of FFA and PPA. The hypothesis is that FFA
responds more to curved objects/textures, where PPA responds more to
rectilinear objects/textures.

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

print('>>> FFA-PPA curvature effect <<<')
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
    'vision', 'fmri', 'ffa_ppa', 'insilico_fmri_responses',
    'insilico_fmri_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

insilico_fmri = {}
insilico_fmri['FFA-1'] = {}
insilico_fmri['FFA-1']['lh'] = data['lh_insilico_fmri_ffa1']['curvature']
insilico_fmri['FFA-1']['rh'] = data['rh_insilico_fmri_ffa1']['curvature']
insilico_fmri['FFA-2'] = {}
insilico_fmri['FFA-2']['lh'] = data['lh_insilico_fmri_ffa2']['curvature']
insilico_fmri['FFA-2']['rh'] = data['rh_insilico_fmri_ffa2']['curvature']
insilico_fmri['PPA'] = {}
insilico_fmri['PPA']['lh'] = data['lh_insilico_fmri_ppa']['curvature']
insilico_fmri['PPA']['rh'] = data['rh_insilico_fmri_ppa']['curvature']
metadata = data['metadata']
del data


# =============================================================================
# NCSNR and prediction accuracy thresholding
# =============================================================================
# Retain vertices that have above threshold (i) NCSNR AND (ii) encoding
# prediction accuracy.

# Loop across ROIs and hemispheres
rois = ['FFA-1', 'FFA-2', 'PPA']
hemispheres = ['lh', 'rh']
for roi in rois:
    for hem in hemispheres:

        # Loop across subjects
        for s in range(len(metadata)):

            # Get the vertex indices of the ROI of interest
            roi_idx = metadata[s]['fmri'][hem+'_fsaverage_rois'][roi]

            # NCSNR and noise ceiling vertex selection
            ncsnr = metadata[s]['fmri'][hem+'_ncsnr'][roi_idx]
            idx_ncsnr = ncsnr > args.ncsnr_threshold
            encoding = metadata[s]['encoding_models']\
                [hem+'_explained_variance_nsdcore'][roi_idx]
            idx_encoding = encoding > args.encoding_threshold
            idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

            # Loop across image types and threshold vertices
            for key in insilico_fmri[roi][hem].keys():
                insilico_fmri[roi][hem][key][s][idx_nan] = np.nan


# =============================================================================
# Get the mean ROI response across vertices for each image type
# =============================================================================
# Empty results dictionary
vertex_mean_resp = {}

# Loop across ROIs and image types
rois = ['FFA', 'PPA']
image_types = ['Curvature-Obj-Curvy', 'Curvature-Shapes-Circles',
    'Curvature-Tex-Curvy', 'Curvature-Obj-Rectilinear',
    'Curvature-Shapes-Diamonds', 'Curvature-Tex-Rectilinear']
for roi in rois:

    vertex_mean_resp[roi] = {}

    for itype in image_types:

        # Empty arrays of shape (n_subjects,)
        vertex_mean_resp[roi][itype] = np.zeros((len(metadata)))

        # Loop across subjects
        for s in range(len(metadata)):

            # Empty subject response list
            vertex_mean_resp_sub = []

            # Loop across hemispheres
            for hem in hemispheres:

                # Get the vertex responses for the chosen ROI
                if roi == 'FFA':
                    # Get the vertex responses for both parts of the ROI
                    vertex_mean_resp_sub.append(np.append(
                        insilico_fmri['FFA-1'][hem][itype][s],
                        insilico_fmri['FFA-2'][hem][itype][s]))
                else:
                    # Get the vertex responses for the ROI
                    vertex_mean_resp_sub.append(
                        insilico_fmri[roi][hem][itype][s])

            # Compute the mean ROI response across vertices
            vertex_mean_resp[roi][itype][s] = np.nanmean(np.concatenate(
                vertex_mean_resp_sub))


# =============================================================================
# Compute significant difference between responses for different image types
# =============================================================================
# Empty results dictionary
pval_diff = {}

# FFA
# Obj-Curvy > Obj-Rectilinear
pval_diff['FFA_Obj-Curvy>Obj-Rectilinear'] = ttest_rel(
    vertex_mean_resp['FFA']['Curvature-Obj-Curvy'],
    vertex_mean_resp['FFA']['Curvature-Obj-Rectilinear'],
    alternative='greater')[1]
# Shapes-Circles > Shapes-Diamonds
pval_diff['FFA_Shapes-Circles>Shapes-Diamonds'] = ttest_rel(
    vertex_mean_resp['FFA']['Curvature-Shapes-Circles'],
    vertex_mean_resp['FFA']['Curvature-Shapes-Diamonds'],
    alternative='greater')[1]
# Tex-Curvy > Tex-Rectilinear
pval_diff['FFA_Tex-Curvy_>Tex-Rectilinear'] = ttest_rel(
    vertex_mean_resp['FFA']['Curvature-Tex-Curvy'],
    vertex_mean_resp['FFA']['Curvature-Tex-Rectilinear'],
    alternative='greater')[1]

# PPA
# Obj-Curvy < Obj-Rectilinear
pval_diff['PPA_Obj-Curvy<Obj-Rectilinear'] = ttest_rel(
    vertex_mean_resp['PPA']['Curvature-Obj-Curvy'],
    vertex_mean_resp['PPA']['Curvature-Obj-Rectilinear'],
    alternative='less')[1]
# Shapes-Circles < Shapes-Diamonds
pval_diff['PPA_Shapes-Circles<Shapes-Diamonds'] = ttest_rel(
    vertex_mean_resp['PPA']['Curvature-Shapes-Circles'],
    vertex_mean_resp['PPA']['Curvature-Shapes-Diamonds'],
    alternative='less')[1]
# Tex-Curvy < Tex-Rectilinear
pval_diff['PPA_Tex-Curvy_<Tex-Rectilinear'] = ttest_rel(
    vertex_mean_resp['PPA']['Curvature-Tex-Curvy'],
    vertex_mean_resp['PPA']['Curvature-Tex-Rectilinear'],
    alternative='less')[1]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Empty result variables
ci_vertex_mean_resp = {}
dist = {}
for roi in rois:
    ci_vertex_mean_resp[roi] = {}
    dist[roi] = {}
    for itype in image_types:
        ci_vertex_mean_resp[roi][itype] = np.zeros((2))
        dist[roi][itype] = np.zeros((args.n_iter))

# Create the bootstrap distribution
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(metadata)))
    for roi in rois:
        for itype in image_types:
            dist[roi][itype][i] = np.mean(vertex_mean_resp[roi][itype][idx])

# Compute the CIs from the bootstrap distribution
for roi in rois:
    for itype in image_types:
        ci_vertex_mean_resp[roi][itype][0] = \
            np.percentile(dist[roi][itype], 2.5)
        ci_vertex_mean_resp[roi][itype][1] = \
            np.percentile(dist[roi][itype], 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'vertex_mean_resp': vertex_mean_resp,
    'pval_diff': pval_diff,
    'ci_vertex_mean_resp': ci_vertex_mean_resp,
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'curvature.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore