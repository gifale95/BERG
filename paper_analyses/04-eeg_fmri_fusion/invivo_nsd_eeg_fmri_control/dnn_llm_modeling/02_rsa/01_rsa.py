"""Perform searchlight RSA between t-fMRI responses and vision DNN features.

To reduce computational load, the M/EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

Parameters
----------
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
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--criterion', default='nearest', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=100, type=int)
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
# Load the stimulus features and transform them into RDMs
# =============================================================================
# Vision DNN
# Load the vision DNN test features
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'stimulus_features',
    f'vision_dnn_features_sub-{args.subject:02d}.npy')
vision_dnn_test = np.load(data_dir,
    allow_pickle=True).item()['vision_dnn_features_test']
# Tranform the vision DNN features into an RDM
vision_dnn_test = 1 - corr_matrix(vision_dnn_test.T)
# Take the lower triangle of the RDM
idx_tril = np.tril_indices(len(vision_dnn_test), -1)
vision_dnn_test = vision_dnn_test[idx_tril]

# LLM
# Load the LLM test embeddings
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'stimulus_features',
    f'llm_embeddings_sub-{args.subject:02d}.npy')
llm_test = np.load(data_dir, allow_pickle=True).item()['llm_embeddings_test']
# Tranform the LLM embeddings into an RDM
llm_test = 1 - corr_matrix(llm_test.T)
# Take the lower triangle of the RDM
llm_test = llm_test[idx_tril]


# =============================================================================
# Load the EEG test responses
# =============================================================================
# Load the EEG responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_data')
file_name = f'eeg_test_sub-{args.subject:02d}.npy'
eeg_test = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()['eeg_test']

# Average the EEG responses into two splits (for later cross-validation in the
# variance paritioning analysis)
idx_even = np.arange(0, eeg_test.shape[1], 2)
idx_odd = np.arange(1, eeg_test.shape[1], 2)
eeg_test_1 = np.mean(eeg_test[:,idx_even], 1)
eeg_test_2 = np.mean(eeg_test[:,idx_odd], 1)
del eeg_test

# Get the time points # !!! Use official time points
n_times = 615
times = np.round(np.linspace(-200, 1000, n_times)).astype(int)
# Account for the 50ms shift in the EEG responses # !!!
shift = -50
times = times + shift
# Only select time points between -100ms and 600ms
t_start = np.where(times == -100)[0][0]
t_end = np.where(times == 600)[0][0]
times = times[t_start:t_end+1]


# =============================================================================
# Only select vertices falling within the NSD visual streams
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the fMRI metadata
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.subject
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
# Empty RSA result array of shape:
# (N fMRI vertices, EEG time points)
rsa = np.zeros((n_vertices, len(times)), dtype=np.float32) # !!!
rsa[:] = np.nan

for t in tqdm(range(len(times))):


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
    # Empty t-fMRI response array of shape:
    # (N Images, 163842 Vertices)
    tfmri = np.zeros((2, len(eeg_test_1), n_vertices), dtype=np.float32)

    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_sub-{args.subject:02d}_hemi-{args.hemisphere}_'
        f'eeg_train_trials-all_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights',
        file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses for the test images with in vivo EEG
    tfmri[0,:,idx_v] = reg.predict(eeg_test_1[:,:,t])
    tfmri[1,:,idx_v] = reg.predict(eeg_test_2[:,:,t])
    del reg_param, reg


# =============================================================================
# Loop across fMRI vertices
# =============================================================================
    for v in idx_v:


# =============================================================================
# Create the t-fMRI RDM
# =============================================================================
        # Select the neighborhood based on the chosen criterion
        if args.criterion == 'nearest':
            # Get k-smallest distances (including the target vertex)
            neighborhood = np.argsort(geodesic_distances[v])[:args.k]
        elif args.criterion == 'radius':
            # Select all vertices whose distance is within the radius
            mask = geodesic_distances[v] <= args.radius_mm
            neighborhood = np.where(mask)[0]

        # Create the fMRI RDM
        tfmri_rdm = np.expand_dims(
            1 - corr_matrix(tfmri[0,:,neighborhood].T), 0)
        tfmri_rdm = np.append(np.expand_dims(
            1 - corr_matrix(tfmri[1,:,neighborhood].T), 0), 0)


# =============================================================================
# Perform searchlight RSA (variance partitioning) # !!!
# =============================================================================
        # RSA with the vision DNN RDM
        score_vision_dnn = np.zeros(len(tfmri_rdm))
        for i in range(len(tfmri_rdm)):
            # Train the regression
            X = np.reshape(vision_dnn_test, (-1, 1))
            y_train = np.reshape(tfmri_rdm[i], (-1, 1))
            reg = LinearRegression()
            reg.fit(X, y_train)
            # Test the regression (cross-validation)
            i_test = abs(i-1)
            y_test = np.reshape(tfmri_rdm[i_test], (-1, 1))
            score = np.mean((y_test - reg.predict(X)) ** 2)
            # Compute R2 adjusted # !!!
            score_vision_dnn[i] = 

        # RSA with the LLM RDM # !!!

        # RSA with both vision DNN and LLM RDMs # !!!
        

        # Compute the total unique and shared variance explained by the vision
        # DNN and LLM RDMs # !!!







# =============================================================================
# Save the results # !!!
# =============================================================================
results = {
    'rsa': rsa,
    'metadata_fmri': metadata_fmri
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'rsa', f'source_dataset-{args.source_dataset}')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'rsa_fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}'
             f'_dnn_model-{args.dnn_model}.npy')

np.save(os.path.join(save_dir, file_name), results)