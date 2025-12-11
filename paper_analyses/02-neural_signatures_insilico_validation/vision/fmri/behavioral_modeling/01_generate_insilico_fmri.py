"""Perform searchlight RSA between in silico fMRI responses and behavioral
embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses.
subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
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
from PIL import Image
from tqdm import tqdm
from berg import BERG
import pandas as pd
from scipy.stats import pearsonr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--criterion', default='radius', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=10, type=int)
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
# Load the THINGS EEG2 image metadata
# =============================================================================
# The THINGS EEG2 image metadata can be downloaded from: https://osf.io/y63gw/files/qkgtf

# Load the metadata
metadata_dir = os.path.join(args.berg_dir, args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'behavioral_modeling', 'image_metadata.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()

# Get the test image category number based on the original THINGS database
test_img_concepts_THINGS = metadata['test_img_concepts_THINGS']


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Select the vertices from the chosen hemisphere
n_vertices = 163842
if args.hemisphere == 'lh':
    lh_vertices = np.ones(n_vertices, dtype=int)
    rh_vertices = np.zeros(n_vertices, dtype=int)
    rh_vertices[0] = 1 # At least one vertex must be selected
elif args.hemisphere == 'rh':
    lh_vertices = np.zeros(n_vertices, dtype=int)
    lh_vertices[0] = 1 # At least one vertex must be selected
    rh_vertices = np.ones(n_vertices, dtype=int)

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={
        'lh_vertices': lh_vertices,
        'rh_vertices': rh_vertices
        }
    )


# =============================================================================
# Generate the in silico fMRI responses
# =============================================================================
fmri = []

# Loop across test object concepts
for cat in tqdm(test_img_concepts_THINGS):

    # Get the image exemplar file names for each concept
    image_list = os.listdir(os.path.join(args.things_dir,
        'image-database_things', cat[6:]))
    image_list.sort()

    # Loop across image exemplars
    images = []
    for ifile in image_list:

        # Load the images
        img_path = os.path.join(args.things_dir, 'image-database_things',
            cat[6:], ifile)
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
        img = np.array(img)
        images.append(img)
    
    # Format the images
    images = np.array(images)
    images = np.swapaxes(images, 1, 3)  # BHWC to BCHW

    # Generate the in silico fMRI responses
    fmri_cat, metadata = berg.encode(model, images, return_metadata=True)

    # Store the in silico fMRI responses averaged across image exemplars
    if args.hemisphere == 'lh':
        fmri.append(np.mean(fmri_cat[0], 0))
    if args.hemisphere == 'rh':
        fmri.append(np.mean(fmri_cat[1], 0))

    # Delete unused variables
    del fmri_cat, images

# Convert the fMRI responses to numpy arrays
fmri = np.array(fmri).astype(np.float32)


# =============================================================================
# Create the behavioral RDM
# =============================================================================
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
rsa = np.zeros(fmri.shape[1], dtype=np.float32)

# Take the lower triangle of the behavior RDM
idx_tril = np.tril_indices(len(beh_rdm), -1)
beh_rdm_tril = beh_rdm[idx_tril]

# Get info regarding the vertex splits of the geodesic distances
n_vertices = fmri.shape[1]
total_vertex_splits = 81
vertices_per_split = n_vertices // total_vertex_splits

# Loop across fMRI vertices
for v in tqdm(range(fmri.shape[1])):

    # Only load the precomputed geodesic distances for the first vertex of each
    # split
    idx = v % vertices_per_split
    if idx == 0:
        vertex_split = v // vertices_per_split # Get the split of the target vertex
        data_dir = np.load(os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation', 'vision', 'fmri',
            'behavioral_modeling', 'vertex_geodesic_distances',
            'vertex_geodesic_distances_'+args.hemisphere+'_split-'+
            format(vertex_split, '03')+'.h5py'))
        geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances'][:]

    # Select the neighborhood based on the chosen criterion
    if args.criterion == 'nearest':
        # Get k-smallest distances (including the target vertex)
        neighborhood = np.argsort(geodesic_distances[idx])[:args.k]
    elif args.criterion == 'radius':
        # Select all vertices whose distance is within the radius
        mask = geodesic_distances[idx] <= args.radius_mm
        neighborhood = np.where(mask)[0]

    # Create the fMRI RDM
    fmri_rdm = 1 - corr_matrix(fmri[:,neighborhood].T)

    # Take the lower triangle of the fMRI RDM
    fmri_rdm_tril = fmri_rdm[idx_tril]

    # Perform RSA
    rsa[v] = pearsonr(beh_rdm_tril, fmri_rdm_tril)[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'behavioral_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_' + args.hemisphere + \
    '.npy'

np.save(os.path.join(save_dir, file_name), results)