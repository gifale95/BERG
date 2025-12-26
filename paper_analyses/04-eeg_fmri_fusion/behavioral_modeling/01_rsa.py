"""Perform searchlight RSA between the t-fMRI responses and behavioral
embeddings.

To reduce computational load, the analysis is only performed for vertices
falling within the NSD visual streams.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
criterion : str
    Criterion to define the searchlight neighborhood: 'radius' for all vertices
    within a geodesic radius, 'nearest' for k-nearest neighbors.
radius_mm : float
    Geodesic radius in millimeters (default = 10 mm), if criterion is 'radius'.
k : int
    Number of nearest geodesic neighbors (default = 10), if criterion is
    'nearest'.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--criterion', default='radius', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define the vectorized correlation function
# =============================================================================
def corr_matrix(X):
    """
    Computes the correlation matrix of the input data.
    Parameters
    ----------
    X : (N, M) float array
        Input data matrix with N features and M samples.

    Returns
    -------
    corr : (M, M) float array
        Correlation matrix of the input data.
    """

    Xc = X - X.mean(axis=0)
    Xc /= np.sqrt((Xc**2).sum(axis=0))

    return (Xc.T @ Xc).astype(np.float32)


# =============================================================================
# Access the in t-fMRI responses and metadata
# =============================================================================
# Access the in t-fMRI
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'tfmri_responses',
    'things_eeg_2_test_images',
    f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}.h5')
data = np.load(data_dir, allow_pickle=True).item()
fmri = data['fmri'].astype(np.float32)
metadata = data['metadata']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )


# =============================================================================
# Create the behavioral RDM
# =============================================================================
# Load the THINGS EEG2 image metadata
# The THINGS EEG2 image metadata can be downloaded from: https://osf.io/y63gw/files/qkgtf
metadata_dir = os.path.join(args.berg_dir, args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'behavioral_modeling', 'image_metadata.npy')
metadata_things = np.load(metadata_dir, allow_pickle=True).item()
# Get the test image category number based on the original THINGS database
test_img_concepts_THINGS = metadata_things['test_img_concepts_THINGS']

# Load the behavioral embeddings (the behavioral emebddings can be downloaded
# from: https://osf.io/f5rn6/overview)
embedding_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'behavioral_modeling', 'spose_embedding_66d_sorted.txt')
beh_embeddings_all = np.array(pd.read_csv(embedding_dir, delim_whitespace=True,
    header=None)).astype(np.float32)

# Retain the embeddings from the 200 test image concepts
idx_test = np.zeros(len(test_img_concepts_THINGS), dtype=int)
for i, img in enumerate(test_img_concepts_THINGS):
    idx_test[i] = int(img[:5]) - 1
beh_embeddings = beh_embeddings_all[idx_test]

# Create the RDM
beh_rdm = 1 - corr_matrix(beh_embeddings.T)


# =============================================================================
# Perform searchlight RSA
# =============================================================================
# Empty RSA results array
rsa = np.empty((fmri.shape[1], fmri.shape[2]), dtype=np.float32)
rsa[:] = np.nan

# Take the lower triangle of the behavior RDM
idx_tril = np.tril_indices(len(beh_rdm), -1)
beh_rdm_tril = beh_rdm[idx_tril]

# Access the precomputed geodesic distances
data_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'behavioral_modeling', 'vertex_geodesic_distances',
    'vertex_geodesic_distances_'+args.hemisphere+'.h5')
geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances']

# Only use vertices falling within the NSD visual streams
idx_v = np.zeros(fmri.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1

# Loop across fMRI vertices and EEG time points
for v in tqdm(idx_v):

    # Select the neighborhood based on the chosen criterion
    if args.criterion == 'nearest':
        # Get k-smallest distances (including the target vertex)
        neighborhood = np.argsort(geodesic_distances[v])[:args.k]
    elif args.criterion == 'radius':
        # Select all vertices whose distance is within the radius
        mask = geodesic_distances[v] <= args.radius_mm
        neighborhood = np.where(mask)[0]

    # Loop across EEG time points
    for t in range(fmri.shape[2]):

        # Create the fMRI RDM
        fmri_rdm = 1 - corr_matrix(fmri[:,neighborhood,t].T)

        # Perform RSA
        rsa[v,t] = pearsonr(beh_rdm_tril, fmri_rdm[idx_tril])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = f'rsa_sub-{args.fmri_subject:02d}_{args.hemisphere}.npy'

np.save(os.path.join(save_dir, file_name), results)