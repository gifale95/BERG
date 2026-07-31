"""Perform an ecoding-based partial correlation analysis between t-fMRI
responses and vision DNN features or LLM embeddings.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1 to 8.
rois : list
    List containing the ROIs used for the Granger Causality analysis. All ROIs
    are tested in a pairwise fashion.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'even' (even trials), and 'odd' (odd trials).
tot_time_splits : int
    The total number of splits in which the EEG time points are divided.
time_split : int
    The time split used, out of the total time splits.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_train_trials', default='even', type=str)
parser.add_argument('--tot_time_splits', default=10, type=int)
parser.add_argument('--time_split', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Partial correlation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus features
# =============================================================================
# Vision DNN
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'stimulus_features',
    f'vision_dnn_features_sub-{args.subject:02d}.npy')
data = np.load(data_dir, allow_pickle=True).item()
vision_dnn_test = data['vision_dnn_features_test']
vision_dnn_train = data['vision_dnn_features_train']

# LLM
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'stimulus_features',
    f'llm_embeddings_sub-{args.subject:02d}.npy')
data = np.load(data_dir, allow_pickle=True).item()
llm_test = data['llm_embeddings_test']
llm_train = data['llm_embeddings_train']
del data


# =============================================================================
# Load the train and test EEG responses
# =============================================================================
# Load the EEG train responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_data')
file_name_train = (f'eeg_train_sub-{args.subject:02d}_'
    f'trial_avg-{args.eeg_train_trials}.npy')
eeg_train = np.load(os.path.join(data_dir, file_name_train),
    allow_pickle=True).item()['eeg_train'].astype(np.float32)

# Load the EEG test responses, and average them across repeats
file_name_test = f'eeg_test_sub-{args.subject:02d}.npy'
eeg_test = np.mean(np.load(os.path.join(data_dir, file_name_test),
    allow_pickle=True).item()['eeg_test'].astype(np.float32), 1)


# =============================================================================
# Time point selection
# =============================================================================
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
# Get the vertex number
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
     'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights',
    f'weights_sub-{args.subject:02d}_hemi-{args.hemisphere}_'
    f'eeg_train_trials-all_eeg_time-000.npy')

reg_param = np.load(data_dir, allow_pickle=True).item()
n_vertices = len(reg_param['intercept_'])

del reg_param


# =============================================================================
# Empty result arrays
# =============================================================================
# Empty result arrays of shape:
# (N fMRI vertices, EEG time points)
total_variance_vision_dnn = np.zeros((n_vertices, len(times_new)),
    dtype=np.float32)
unique_variance_vision_dnn = np.zeros((n_vertices, len(times_new)),
    dtype=np.float32)
unique_variance_llm = np.zeros((n_vertices, len(times_new)), dtype=np.float32)
total_variance_llm = np.zeros((n_vertices, len(times_new)), dtype=np.float32)


# =============================================================================
# Loop across EEG time points
# =============================================================================
for t, t_idx in tqdm(enumerate(range(start_idx, end_idx))):


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
    # Load the EEG-fMRI encoding fusion models weights (if using the even EEG
    # training trials, then load the models trained on the odd EEG training
    # trials, and vice versa, to account for the fusion models using the noise
    # in the EEG responses to predict fMRI)
    if args.eeg_train_trials == 'even':
        weights_eeg_train_trials = 'odd'
    elif args.eeg_train_trials == 'odd':
        weights_eeg_train_trials = 'even'
    file_name = (f'weights_sub-{args.subject:02d}_hemi-{args.hemisphere}_'
        f'eeg_train_trials-{weights_eeg_train_trials}_eeg_time-{t_idx:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights',
        file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses
    tfmri_train = reg.predict(eeg_train[:,:,t_idx]).astype(dtype=np.float32)
    y = reg.predict(eeg_test[:,:,t_idx]).astype(dtype=np.float32)
    del reg_param, reg


# =============================================================================
# Train the encoding models using the vision DNN and LLM features
# =============================================================================
    # Train the encoding models
    reg_vision_dnn = LinearRegression()
    reg_vision_dnn.fit(vision_dnn_train, tfmri_train)
    reg_llm = LinearRegression()
    reg_llm.fit(llm_train, tfmri_train)

    # Compute the predictions of the encoding models on the test set
    X_vision_dnn = reg_vision_dnn.predict(vision_dnn_test)
    X_llm = reg_llm.predict(llm_test)
    del reg_vision_dnn, reg_llm

    # Add an extra dimension to the test data for the partial correlation
    # analysis
    y = np.expand_dims(y, 2)
    X_vision_dnn = np.expand_dims(X_vision_dnn, 2)
    X_llm = np.expand_dims(X_llm, 2)


# =============================================================================
# Partial correlation
# =============================================================================
    # Loop across vertices
    for v in range(n_vertices):

        # Compute the total variance explained by vision DNNs
        total_variance_vision_dnn[v,t] = pearsonr(y[:,v].flatten(),
            X_vision_dnn[:,v].flatten())[0]

        # Compute the total variance explained by LLMs
        total_variance_llm[v,t] = pearsonr(y[:,v].flatten(),
            X_llm[:,v].flatten())[0]

        # Compute the unique variance explained by vision DNNs
        # Remove the linear relationship with LLM embeddings from the t-fMRI
        # responses and vision DNN features
        reg = LinearRegression()
        reg.fit(X_llm[:,v], y[:,v])
        y_v = y[:,v] - reg.predict(X_llm[:,v])
        del reg
        # Remove the linear relationship with LLM embeddings from the vision
        # DNN features
        reg = LinearRegression()
        reg.fit(X_llm[:,v], X_vision_dnn[:,v])
        X_vision_dnn_v = X_vision_dnn[:,v] - reg.predict(X_llm[:,v])
        del reg
        # Correlate the residuals to get the unique variance explained by
        # vision DNNs
        unique_variance_vision_dnn[v,t] = pearsonr(y_v.flatten(),
            X_vision_dnn_v.flatten())[0]

        # Compute the unique variance explained by LLMs
        # Remove the linear relationship with vision DNN features from the
        # t-fMRI responses and LLM embeddings
        reg = LinearRegression()
        reg.fit(X_vision_dnn[:,v], y[:,v])
        y_v = y[:,v] - reg.predict(X_vision_dnn[:,v])
        del reg
        # Remove the linear relationship with vision DNN features from the LLM
        # embeddings
        reg = LinearRegression()
        reg.fit(X_vision_dnn[:,v], X_llm[:,v])
        X_llm_v = X_llm[:,v] - reg.predict(X_vision_dnn[:,v])
        del reg
        # Correlate the residuals to get the unique variance explained by LLMs
        unique_variance_llm[v,t] = pearsonr(y_v.flatten(),
            X_llm_v.flatten())[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'times_new': times_new,
    'partial_correlation': {
        'total_variance_vision_dnn': total_variance_vision_dnn,
        'total_variance_llm': total_variance_llm,
        'unique_variance_vision_dnn': unique_variance_vision_dnn,
        'unique_variance_llm': unique_variance_llm
    }
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding_partial_correlation',
    'partial_correlation')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'partial_correlation_sub-{args.subject:02d}_hemisphere-'
    f'{args.hemisphere}_eeg_train_trials-{args.eeg_train_trials}_'
    f'time_split-{args.time_split:02d}.npy')

np.save(os.path.join(save_dir, file_name), results)