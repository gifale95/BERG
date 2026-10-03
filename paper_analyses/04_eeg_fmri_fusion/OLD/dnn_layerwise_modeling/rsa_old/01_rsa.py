"""Perform an encoding-based correlation analysis between t-fMRI responses and
layerwise vision DNN features.

Parameters
----------
fmri_subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
tot_time_splits : int
    The total number of splits in which the EEG time points are divided.
time_split : int
    The time split used, out of the total time splits.
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

"""

import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--tot_time_splits', default=10, type=int)
parser.add_argument('--time_split', default=0, type=int)
parser.add_argument('--criterion', default='nearest', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> DNN layerwise modeling <<<')
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
# Load the layerwise vision DNN features
# =============================================================================
# Load the RDMs
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'dnn_rdms',
    'dnn_rdms_alexnet.npy')
dnn_rdms = np.load(data_dir, allow_pickle=True).item()
model_layers = list(dnn_rdms.keys())

# Only retain the RDM upper triangle
idx_triu = np.triu_indices(dnn_rdms[model_layers[0]].shape[0], k=1)
for layer in dnn_rdms.keys():
    dnn_rdms[layer] = dnn_rdms[layer][idx_triu]


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
# Select the time points from the current time split
# =============================================================================
times_per_split = int(np.ceil(len(times) / args.tot_time_splits))
start_idx = args.time_split * times_per_split
end_idx = min((args.time_split + 1) * times_per_split, len(times))
times_new = times[start_idx:end_idx]


# =============================================================================
# Only select vertices falling within the NSD visual streams
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the fMRI metadata
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
n_vertices = 163842
idx_streams = np.zeros(n_vertices, dtype=bool)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_streams[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois']\
        [stream]] = 1
idx_v = np.where(idx_streams)[0]


# =============================================================================
# Access the precomputed geodesic distances
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'geodesic_vertex_distances',
    'geodesic_vertex_distances_'+args.hemisphere+'.h5')
geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances']


# =============================================================================
# Loop across EEG time points
# =============================================================================
# Empty result arrays of shape:
# (N fMRI vertices, EEG time points)
rsa = {}
for layer in model_layers:
    rsa[layer] = np.zeros((n_vertices, len(times_new)), dtype=np.float32)
    rsa[layer][:] = np.nan

for t, t_idx in tqdm(enumerate(range(start_idx, end_idx))):


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
    # Empty t-fMRI response arrays of shape:
    # (N Images, 163842 Vertices)
    tfmri_test = np.zeros((len(eeg_test), n_vertices), dtype=np.float32)

    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
        f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights', f'source_dataset-things_eeg_2',
        file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses for the test images with in vivo EEG
    tfmri_test[:,idx_v] = reg.predict(eeg_test[:,:,t_idx])
    del reg_param, reg


# =============================================================================
# Loop across fMRI vertices
# =============================================================================
    for v in tqdm(idx_v):


# =============================================================================
# Create the t-fMRI RDMs
# =============================================================================
        # Select the neighborhood based on the chosen criterion
        if args.criterion == 'nearest':
            # Get k-smallest distances (including the target vertex)
            neighborhood = np.argsort(geodesic_distances[v])[:args.k]
        elif args.criterion == 'radius':
            # Select all vertices whose distance is within the radius
            mask = geodesic_distances[v] <= args.radius_mm
            neighborhood = np.where(mask)[0]

        # Create the fMRI RDMs
        tfmri_rdm_test = 1 - corr_matrix(tfmri_test[:,neighborhood].T)

        # Take the upper triangle of the RDMs
        tfmri_rdm_test = tfmri_rdm_test[idx_triu]


# =============================================================================
# Perform searchlight RSA
# =============================================================================
        for layer in model_layers:
            rsa[layer][v,t] = pearsonr(dnn_rdms[layer], tfmri_rdm_test)[0]

    # Delete unused variables
    del tfmri_test, tfmri_rdm_test


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'times_new': times_new,
    'model_layers': model_layers,
    'rsa': rsa
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'rsa_fmri_sub-{args.fmri_subject:02d}_hemisphere-'
    f'{args.hemisphere}_time_split-{args.time_split:02d}.npy')

np.save(os.path.join(save_dir, file_name), results)