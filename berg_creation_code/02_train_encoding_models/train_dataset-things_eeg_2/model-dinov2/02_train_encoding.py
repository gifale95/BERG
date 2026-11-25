"""Fit a linear regression to predict EEG data using DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X). A separate model is trained for each EEG channel and
time point, and also for each of the four EEG train data repeats: in this way,
for each input image we can have four different instances of synthetic EEG
response.

The feature maps come from DINOv2, and are downsampled using PCA.

Parameters
----------
subject : int
    Number of the used THINGS EEG2 subject.
model : str
    Name of the used encoding model.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import numpy as np
import os
import h5py
import copy
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', type=str, default='dinov2')
parser.add_argument('--pcs', type=int, default=250)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, _ = parser.parse_known_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the PCA weights and image features
# =============================================================================
# Load the image features
data_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-eeg',
    'train_dataset-things_eeg_2', 'model-'+args.model,
    'encoding_models_weights', 'image_features.npy')
features = np.load(data_dir, allow_pickle=True).item()
fmaps_train = features['feat_train'][:,:args.pcs]
fmaps_test = features['feat_test'][:,:args.pcs]


# =============================================================================
# Train the encoding models
# =============================================================================
# Load the training EEG responses
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2')
eeg_dir = os.path.join(data_dir, 'eeg_sub-'+format(args.subject,'02')+
    '_split-train.h5')
eeg_train = h5py.File(eeg_dir, 'r')['eeg'][:]
n_repeats = eeg_train.shape[1]
n_channels = eeg_train.shape[2]
n_times = eeg_train.shape[3]

# Fit an encoding model at each EEG repeat, time-point and channel
reg_param = {}
eeg_test_pred = np.zeros((len(fmaps_test), n_repeats, n_channels, n_times),
    dtype=np.float32)

for r in range(eeg_train.shape[1]): # Loop over the 4 training EEG repeats
    # Reshape the EEG to (Samples x Features)
    eeg = eeg_train[:,r]
    eeg = np.reshape(eeg, (len(eeg), -1))
    # Fit the linear regression
    reg = LinearRegression()
    reg.fit(fmaps_train, eeg)

    # Store the linear regression weights
    reg_dict = {
        'coef_': reg.coef_,
        'intercept_': reg.intercept_,
        'n_features_in_': reg.n_features_in_
        }
    reg_param['rep-'+str(r+1)] = copy.deepcopy(reg_dict)

    # Use the learned weights to generate in silico EEG responses for the test
    # images
    eeg_test_pred_rep = reg.predict(fmaps_test)
    eeg_test_pred[:,r] = np.reshape(eeg_test_pred_rep, (-1, n_channels,
        n_times))
    del reg_dict

# Save the in silico EEG responses for the test images
save_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-eeg', 'train_dataset-things_eeg_2', f'model-{args.model}')
file_name = f'eeg_test_pred_subject-{args.subject:02d}_pcs-{args.pcs:04d}.npy'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), eeg_test_pred)


# =============================================================================
# Save the trained encoding models weights
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-eeg',
    'train_dataset-things_eeg_2', f'model-{args.model}',
    'encoding_models_weights')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

file_name = f'weights_subject-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), reg_param)
