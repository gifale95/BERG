"""Perform searchlight RSA between in silico fMRI responses and layerwise DNN
features.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
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
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--criterion', default='nearest', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=100, type=int)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
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
# Load the in silico fMRI responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'insilico_fmri_responses',
    args.encoding_model, 'insilico_fmri_responses_sub-'+
    format(args.subject, '02')+'_'+args.hemisphere+'.npy')

data = np.load(data_dir, allow_pickle=True).item()
fmri = data['fmri'].astype(np.float32)
metadata = data['metadata']


# =============================================================================
# Load the DNN layerwise RDMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'dnn_rdms',
    'dnn_rdms_'+args.model+'.npy')

dnn_rdms = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Perform searchlight RSA
# =============================================================================
# Empty RSA results arrays
rsa = {}
for key in dnn_rdms.keys():
    rsa[key] = np.zeros(fmri.shape[1], dtype=np.float32)

# Take the lower triangle of the DNN RDMs
idx_tril = np.tril_indices(len(fmri), -1)
dnn_rdm_tril = {}
for key, val in dnn_rdms.items():
    dnn_rdm_tril[key] = val[idx_tril]

# Access the precomputed geodesic distances
data_dir = os.path.join(args.berg_dir, 'geodesic_vertex_distances',
    'geodesic_vertex_distances_'+args.hemisphere+'.h5')
geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances']

# Loop across fMRI vertices
for v in tqdm(range(fmri.shape[1])):

    # Select the neighborhood based on the chosen criterion
    if args.criterion == 'nearest':
        # Get k-smallest distances (including the target vertex)
        neighborhood = np.argsort(geodesic_distances[v])[:args.k]
    elif args.criterion == 'radius':
        # Select all vertices whose distance is within the radius
        mask = geodesic_distances[v] <= args.radius_mm
        neighborhood = np.where(mask)[0]

    # Create the fMRI RDM
    fmri_rdm = 1 - corr_matrix(fmri[:,neighborhood].T)

    # Perform RSA with each DNN layer
    for key, val in dnn_rdm_tril.items():
        rsa[key][v] = pearsonr(val, fmri_rdm[idx_tril])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'rsa', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_' + args.hemisphere + \
    '_model-' + args.model + '.npy'

np.save(os.path.join(save_dir, file_name), results)