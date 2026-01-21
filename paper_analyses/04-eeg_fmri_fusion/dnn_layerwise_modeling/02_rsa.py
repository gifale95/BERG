"""Perform searchlight RSA between t-fMRI responses and DNN layerwise features.

To reduce computational load, the EEG/fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 10.
criterion : str
    Criterion to define the searchlight neighborhood: 'radius' for all vertices
    within a geodesic radius, 'nearest' for k-nearest neighbors.
radius_mm : float
    Geodesic radius in millimeters (default = 10 mm), if criterion is 'radius'.
k : int
    Number of nearest geodesic neighbors (default = 10), if criterion is
    'nearest'.
dnn_model : str
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
from berg import BERG
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--criterion', default='radius', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--dnn_model', default='alexnet', type=str)
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
# Load the DNN layerwise RDMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'dnn_rdms', 'dnn_rdms_'+args.dnn_model+'.npy')

dnn_rdms = np.load(data_dir, allow_pickle=True).item()

# Take the lower triangle of the DNN RDMs
idx_tril = np.tril_indices(len(dnn_rdms[list(dnn_rdms.keys())[0]]), -1)
dnn_rdm_tril = {}
for key, val in dnn_rdms.items():
    dnn_rdm_tril[key] = val[idx_tril]


# =============================================================================
# Load and append the in vivo EEG test responses across subjects
# =============================================================================
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
    eeg_test_sub = np.mean(h5py.File(eeg_dir_test, 'r')['eeg'][:],
        1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_test = eeg_test_sub
    else:
        eeg_test = np.append(eeg_test, eeg_test_sub, 1)
    del eeg_test_sub

# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Load the fMRI metadata
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
n_vertices = 163842
idx_v = np.zeros(n_vertices, dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]

# Empty RSA result arrays of shape:
# (N fMRI vertices, 140 EEG time points)
rsa = {}
for key in dnn_rdms.keys():
    rsa[key] = np.empty((n_vertices, len(times)), dtype=np.float32)
    rsa[key][:] = np.nan

# Access the precomputed geodesic distances
data_dir = os.path.join(args.berg_dir, 'geodesic_vertex_distances',
    'geodesic_vertex_distances_'+args.hemisphere+'.h5')
geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances']

# Loop across EEG time points
for t in tqdm(range(len(times))):

    # Empty t-fMRI response array of shape:
    # (N Images, 163842 Vertices)
    tfmri = np.zeros((len(eeg_test), n_vertices), dtype=np.float32)

    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
        f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights', file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses for the test images with in vivo EEG
    tfmri[:,idx_v] = reg.predict(eeg_test[:,:,t])
    del reg_param, reg


# =============================================================================
# Perform searchlight RSA
# =============================================================================
    # Loop across fMRI vertices
    for v in idx_v:

        # Select the neighborhood based on the chosen criterion
        if args.criterion == 'nearest':
            # Get k-smallest distances (including the target vertex)
            neighborhood = np.argsort(geodesic_distances[v])[:args.k]
        elif args.criterion == 'radius':
            # Select all vertices whose distance is within the radius
            mask = geodesic_distances[v] <= args.radius_mm
            neighborhood = np.where(mask)[0]

        # Create the fMRI RDM
        fmri_rdm = 1 - corr_matrix(tfmri[:,neighborhood].T)

        # Perform RSA with each DNN layer
        for key, val in dnn_rdm_tril.items():
            rsa[key][v,t] = pearsonr(val, fmri_rdm[idx_tril])[0]
        del fmri_rdm
    del tfmri


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'metadata_fmri': metadata_fmri
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'rsa_fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}'
             f'_dnn_model-{args.dnn_model}.npy')

np.save(os.path.join(save_dir, file_name), results)