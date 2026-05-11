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
from berg import BERG
from tqdm import tqdm
import h5py
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_train_trials', default='even', type=str)
parser.add_argument('--tot_time_splits', default=20, type=int)
parser.add_argument('--time_split', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> DNN layerwise modeling <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the layerwise vision DNN features
# =============================================================================
# Vision DNN
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stimulus_features',
    'alexnet_layerwise_stimulus_features.npy')
data = np.load(data_dir, allow_pickle=True).item()
ft_train = data['ft_train']
ft_test = data['ft_test']


# =============================================================================
# Load and append the in vivo EEG train responses across subjects
# =============================================================================
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_train_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-train.h5')
    eeg_train_sub = h5py.File(eeg_train_dir, 'r')['eeg']
    if args.eeg_train_trials == 'even':
        eeg_train_sub = np.mean(eeg_train_sub[:,::2], 1).astype(np.float32)
    elif args.eeg_train_trials == 'odd':
        eeg_train_sub = np.mean(eeg_train_sub[:,1::2], 1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_train = eeg_train_sub
    else:
        eeg_train = np.append(eeg_train, eeg_train_sub, 1)
    del eeg_train_sub


# =============================================================================
# Time point selection
# =============================================================================
# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']

# Select the time points from the current time split
times_per_split = int(np.ceil(len(times) / args.tot_time_splits))
start_idx = args.time_split * times_per_split
end_idx = min((args.time_split + 1) * times_per_split, len(times))
times_new = times[start_idx:end_idx]


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses', 'imageset-things_eeg_2')
file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_split-test.h5'
y = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
idx_v = np.zeros(y.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]
y = y[:,idx_v].astype(np.float32)

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
y_z = (y - y.mean(0)) /  (y.std(0) + eps)
del y


# =============================================================================
# Empty result arrays
# =============================================================================
model_layers = list(ft_train.keys())

# Empty result arrays of shape:
# (N fMRI vertices, EEG time points)
n_vertices = y_z.shape[1]
dnn_layerwise_correlation = {}
for layer in model_layers:
    dnn_layerwise_correlation[layer] = np.zeros(
        (n_vertices, len(times_new)), dtype=np.float32)


# =============================================================================
# Loop across EEG time points
# =============================================================================
for t, t_idx in enumerate(tqdm(range(start_idx, end_idx))):


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
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_hemi-'
        f'{args.hemisphere}_eeg_train_trials-{weights_eeg_train_trials}_'
        f'eeg_time-{t_idx:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights', file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses
    tfmri_train = reg.predict(eeg_train[:,:,t_idx]).astype(dtype=np.float32)
    del reg_param, reg


# =============================================================================
# Train the encoding models using the layerwise vision DNN features
# =============================================================================
    # Loop across DNN layers
    for layer in model_layers:

        # Train the DNN layerwise encoding models
        reg = LinearRegression()
        reg.fit(ft_train[layer], tfmri_train)

        # Compute the predictions of the encoding models on the test set
        X = reg.predict(ft_test[layer]).astype(dtype=np.float32)
        del reg

        # Normalize the predicted responses for later correlation
        X_z = (X - X.mean(0)) / X.std(0)
        del X


# =============================================================================
# Compute the encoding model's accuracy through correlation
# =============================================================================
        dnn_layerwise_correlation[layer][:,t] = np.mean(y_z * X_z, 0)
        del X_z


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'times_new': times_new,
    'model_layers': model_layers,
    'dnn_layerwise_correlation': dnn_layerwise_correlation
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'correlation_results')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'correlation_fmri_sub-{args.fmri_subject:02d}_hemisphere-'
    f'{args.hemisphere}_eeg_train_trials-{args.eeg_train_trials}_'
    f'time_split-{args.time_split:02d}.npy')

np.save(os.path.join(save_dir, file_name), results)