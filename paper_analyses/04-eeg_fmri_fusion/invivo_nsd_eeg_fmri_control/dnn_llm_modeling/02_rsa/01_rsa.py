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
cv_split : int
    Integer indicating which of two EEG splits are used for training or testing
    the variance partitioning models. Possible values are 1 and 2.
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
from tqdm import tqdm
import h5py
from berg import BERG
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--cv_split', default=1, type=int)
parser.add_argument('--tot_time_splits', default=10, type=int)
parser.add_argument('--time_split', default=0, type=int)
parser.add_argument('--criterion', default='nearest', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=100, type=int)
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

# Format the RDMs
X_vision_dnn = np.reshape(vision_dnn_test, (-1, 1))
X_llm = np.reshape(llm_test, (-1, 1))
X_both = np.append(X_llm, X_vision_dnn, 1)


# =============================================================================
# Load the EEG responses
# =============================================================================
# Load the EEG responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_data')
file_name = f'eeg_test_sub-{args.subject:02d}.npy'
eeg = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()['eeg_test']

# Average the EEG responses into two splits (for later cross-validation in the
# variance paritioning analysis)
idx_even = np.arange(0, eeg.shape[1], 2)
idx_odd = np.arange(1, eeg.shape[1], 2)
if args.cv_split == 1:
    eeg_train = np.mean(eeg[:,idx_even], 1)
    eeg_test = np.mean(eeg[:,idx_odd], 1)
elif args.cv_split == 2:
    eeg_train = np.mean(eeg[:,idx_odd], 1)
    eeg_test = np.mean(eeg[:,idx_even], 1)
del eeg

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

# Select the time points from the current time split
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
# Empty result arrays of shape:
# (N fMRI vertices, EEG time points)
total_variance = np.zeros((n_vertices, len(times_new)), dtype=np.float32)
total_variance_vision_dnn = np.zeros(total_variance.shape, dtype=np.float32)
total_variance_llm = np.zeros(total_variance.shape, dtype=np.float32)
unique_variance_vision_dnn = np.zeros(total_variance.shape, dtype=np.float32)
unique_variance_llm = np.zeros(total_variance.shape, dtype=np.float32)
shared_variance = np.zeros(total_variance.shape, dtype=np.float32)
total_variance[:] = np.nan
total_variance_vision_dnn[:] = np.nan
total_variance_llm[:] = np.nan
unique_variance_vision_dnn[:] = np.nan
unique_variance_llm[:] = np.nan
shared_variance[:] = np.nan

for t in tqdm(range(start_idx, end_idx)):


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
    # Empty t-fMRI response arrays of shape:
    # (N Images, 163842 Vertices)
    tfmri_train = np.zeros((len(eeg_train), n_vertices), dtype=np.float32)
    tfmri_test = np.zeros((len(eeg_test), n_vertices), dtype=np.float32)

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
    tfmri_train[:,idx_v] = reg.predict(eeg_train[:,:,t])
    tfmri_test[:,idx_v] = reg.predict(eeg_test[:,:,t])
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
        tfmri_rdm_train = 1 - corr_matrix(tfmri_train[:,neighborhood].T)
        tfmri_rdm_test = 1 - corr_matrix(tfmri_test[:,neighborhood].T)

        # Take the lower triangle of the RDMs
        tfmri_rdm_train = tfmri_rdm_train[idx_tril]
        tfmri_rdm_test = tfmri_rdm_test[idx_tril]


# =============================================================================
# Perform searchlight RSA (variance partitioning)
# =============================================================================
        # Format the fMRI train/test RDMs
        y_train = np.reshape(tfmri_rdm_train, (-1, 1))
        y_test = np.reshape(tfmri_rdm_test, (-1, 1))

        # Compute the total variance explained by vision DNNs
        # Train the regression
        reg = LinearRegression()
        reg.fit(X_vision_dnn, y_train)
        # Test the regression (cross-validation)
        score = np.mean((y_test - reg.predict(X_vision_dnn)) ** 2)
        # Adjust the R2 scores for the number predictors in the model
        total_variance_vision_dnn[v,t] = score * (len(X_vision_dnn) - 1) / \
            (len(X_vision_dnn) - X_vision_dnn.shape[1] - 1)

        # Compute the total variance explained by LLMs
        # Train the regression
        reg = LinearRegression()
        reg.fit(X_llm, y_train)
        # Test the regression (cross-validation)
        score = np.mean((y_test - reg.predict(X_llm)) ** 2)
        # Adjust the R2 scores for the number predictors in the model
        total_variance_llm[v,t] = score * (len(X_llm) - 1) / \
            (len(X_llm) - X_llm.shape[1] - 1)

        # Compute the total variance explained by vision DNNs and LLMs together
        # Train the regression
        reg = LinearRegression()
        reg.fit(X_both, y_train)
        # Test the regression (cross-validation)
        score = np.mean((y_test - reg.predict(X_both)) ** 2)
        # Adjust the R2 scores for the number predictors in the model
        total_variance[v,t] = score * (len(X_both) - 1) / \
            (len(X_both) - X_both.shape[1] - 1)

        # Compute the unique and shared variance explained by the vision DNN
        # and LLM RDMs
        unique_variance_vision_dnn[v,t] = total_variance[v,t] - \
            total_variance_llm[v,t]
        unique_variance_llm[v,t] = total_variance[v,t] - \
            total_variance_vision_dnn[v,t]
        shared_variance[v,t] = total_variance[v,t] - \
            unique_variance_vision_dnn[v,t] - unique_variance_llm[v,t]
        # shared_variance[v,t] = total_variance_vision_dnn[v,t] + \
        #     total_variance_llm[v,t] - total_variance[v,t]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata_fmri': metadata_fmri,
    'times': times,
    'times_new': times_new,
    'total_variance': total_variance,
    'total_variance_vision_dnn': total_variance_vision_dnn,
    'total_variance_llm': total_variance_llm,
    'unique_variance_vision_dnn': unique_variance_vision_dnn,
    'unique_variance_llm': unique_variance_llm,
    'shared_variance': shared_variance
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'rsa', 'rsa_scores')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'rsa_sub-{args.subject:02d}_hemisphere-{args.hemisphere}_'
    f'cv_split-{args.cv_split}_time_split-{args.time_split:02d}.npy')

np.save(os.path.join(save_dir, file_name), results)