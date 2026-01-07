"""Test the tripartite organization effect (Konkle & Caramazza, 2013) on in
silico fMRI responses.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
images : str
    Whether to use 'naturalistic' or 'texforms' images.
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
parser.add_argument('--images', type=str, default='naturalistic')
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Tripartite organization - Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in silico fMRI responses for the tripartite organization images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'insilico_fmri_responses',
    args.encoding_model, 'insilico_fmri_responses_images-'+args.images+'.npy')

data = np.load(data_dir, allow_pickle=True).item()

animals = {}
animals['lh'] = data['lh_animals']
animals['rh'] = data['rh_animals']
big_objects = {}
big_objects['lh'] = data['lh_big_objects']
big_objects['rh'] = data['rh_big_objects']
small_objects = {}
small_objects['lh'] = data['lh_small_objects']
small_objects['rh'] = data['rh_small_objects']
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
        animals[hem][s,idx_nan] = np.nan
        small_objects[hem][s,idx_nan] = np.nan
        big_objects[hem][s,idx_nan] = np.nan


# =============================================================================
# Compute the overlap of different ROIs with the categorical zones
# =============================================================================
# For each subject, compute the overlap of face/body-selective and
# scene-selective ROIs with the animal-selective and big object-selective
# cortical zones, respectively.

# Empty results dictionary
rois = ['FFA', 'OFA', 'EBA', 'FBA', 'PPA', 'OPA', 'RSC']
categories = ['animals', 'small_objects', 'big_objects']
vertex_overlap = {}

# Loop across ROIs and categories
for roi in rois:

    # Empty arrays of shape (n_subjects,) for face/body-selective ROIs
    for cat in categories:
        vertex_overlap[roi+'_'+cat] = np.zeros((len(metadata)))

    # Loop across subjects
    for s in range(len(metadata)):

        # Initialize counters
        tot_vertices = 0
        count_animals = 0
        count_big_objects = 0
        count_small_objects = 0

        # Loop across hemispheres
        for hem in ['lh', 'rh']:

            # Get the vertex indices for the ROI
            if roi == 'FFA' or roi == 'FBA':
                # Get the vertex indices for both parts of the ROI
                idx = np.append(
                    metadata[s]['fmri'][hem+'_fsaverage_rois'][f'{roi}-1'],
                    metadata[s]['fmri'][hem+'_fsaverage_rois'][f'{roi}-2'])
                idx.sort()
            else:
                # Get the vertex indices for the ROI
                idx = metadata[s]['fmri'][hem+'_fsaverage_rois'][roi]

            # Calculate the count of vertices selective for animals, big
            # objects, or small objects
            for v in idx:
                if np.isnan(animals[hem][s,v]):
                    continue
                else:
                    tot_vertices += 1
                    if animals[hem][s,v] > small_objects[hem][s,v] and animals[hem][s,v] > big_objects[hem][s,v]:
                        count_animals += 1
                    if big_objects[hem][s,v] > small_objects[hem][s,v] and big_objects[hem][s,v] > animals[hem][s,v]:
                        count_big_objects += 1
                    if small_objects[hem][s,v] > big_objects[hem][s,v] and small_objects[hem][s,v] > animals[hem][s,v]:
                        count_small_objects += 1

        # Store the vertex overlap results
        vertex_overlap[roi+'_animals'][s] = count_animals / tot_vertices * 100
        vertex_overlap[roi+'_small_objects'][s] = \
            count_small_objects / tot_vertices * 100
        vertex_overlap[roi+'_big_objects'][s] = \
            count_big_objects / tot_vertices * 100


# =============================================================================
# Compute the significance
# =============================================================================
# Empty results dictionary
pval_vertex_overlap = {}

# Compute the significance (animal preference ROIs)
animal_rois = ['FFA', 'OFA', 'EBA', 'FBA']
for a_roi in animal_rois:
    pval_vertex_overlap[a_roi+'_animals>small_objects'] = \
        ttest_rel(vertex_overlap[a_roi+'_animals'],
        vertex_overlap[a_roi+'_small_objects'], alternative='greater')[1]
    pval_vertex_overlap[a_roi+'_animals>big_objects'] = \
        ttest_rel(vertex_overlap[a_roi+'_animals'],
        vertex_overlap[a_roi+'_big_objects'], alternative='greater')[1]

# Compute the significance (big object preference ROIs)
bigobject_rois = ['PPA', 'OPA', 'RSC']
for bo_roi in bigobject_rois:
    pval_vertex_overlap[bo_roi+'_big_objects>animals'] = \
        ttest_rel(vertex_overlap[bo_roi+'_big_objects'],
        vertex_overlap[bo_roi+'_animals'], alternative='greater')[1]
    pval_vertex_overlap[bo_roi+'_big_objects>small_objects'] = \
        ttest_rel(vertex_overlap[bo_roi+'_big_objects'],
        vertex_overlap[bo_roi+'_small_objects'], alternative='greater')[1]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Empty result variables
ci_vertex_overlap = {}
dist = {}
for roi in rois:
    for cat in categories:
        ci_vertex_overlap[roi+'_'+cat] = np.zeros((2)) # type: ignore
        dist[roi+'_'+cat] = np.zeros((args.n_iter)) # type: ignore

# Create the bootstrap distribution
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(metadata)))
    for roi in rois:
        for cat in categories:
            dist[roi+'_'+cat][i] = np.mean(vertex_overlap[roi+'_'+cat][idx])

# Compute the CIs from the bootstrap distribution
for roi in rois:
    for cat in categories:
        ci_vertex_overlap[roi+'_'+cat][0] = \
            np.percentile(dist[roi+'_'+cat], 2.5)
        ci_vertex_overlap[roi+'_'+cat][1] = \
            np.percentile(dist[roi+'_'+cat], 97.5)


# =============================================================================
# Tripartite organization analysis (across-subjects)
# =============================================================================
# Perform the tripartite organization analysis on the fMRI responses averaged
# across subjects.

# Average the responses across subjects
lh_animals_avg = np.nanmean(animals['lh'], 0)
rh_animals_avg = np.nanmean(animals['rh'], 0)
lh_big_objects_avg = np.nanmean(big_objects['lh'], 0)
rh_big_objects_avg = np.nanmean(big_objects['rh'], 0)
lh_small_objects_avg = np.nanmean(small_objects['lh'], 0)
rh_small_objects_avg = np.nanmean(small_objects['rh'], 0)

# Append the in silico fMRI responses for the three conditions
lh_data = np.array([lh_animals_avg, lh_big_objects_avg, lh_small_objects_avg])
rh_data = np.array([rh_animals_avg, rh_big_objects_avg, rh_small_objects_avg])

# For each vertex, select the condition leading to highest response
lh_tripartite_organization = np.argsort(lh_data, axis=0)[-1].astype(np.float32)
rh_tripartite_organization = np.argsort(rh_data, axis=0)[-1].astype(np.float32)

# Threshold with univariate response magnitude
threshold_resp = -.25
lh_idx_nan = np.where(np.max(lh_data, 0) < threshold_resp)[0]
rh_idx_nan = np.where(np.max(rh_data, 0) < threshold_resp)[0]
lh_tripartite_organization[lh_idx_nan] = np.nan
rh_tripartite_organization[rh_idx_nan] = np.nan

# Threshold with ncsnr
lh_idx_nan = np.where(np.isnan(lh_animals_avg))[0]
rh_idx_nan = np.where(np.isnan(rh_animals_avg))[0]
lh_tripartite_organization[lh_idx_nan] = np.nan
rh_tripartite_organization[rh_idx_nan] = np.nan


# =============================================================================
# Save the results
# =============================================================================
results = {
    'vertex_overlap': vertex_overlap,
    'pval_vertex_overlap': pval_vertex_overlap,
    'ci_vertex_overlap': ci_vertex_overlap,
    'lh_tripartite_organization': lh_tripartite_organization,
    'rh_tripartite_organization': rh_tripartite_organization
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_images-' + args.images + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore