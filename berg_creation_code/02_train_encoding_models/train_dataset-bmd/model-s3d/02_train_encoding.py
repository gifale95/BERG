"""Fit a linear regression to predict fMRI responses using DNN feature maps as
predictors. The linear regression is trained using the training image fMRI data
(Y) and feature maps (X). A separate model is trained for each fMRI voxel.

Parameters
----------
subject : int
    Number of the used BMD subject.
model : str
    Name of the used encoding model.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG
bmd_dir : str
    Directory of the BOLD Moments Dataset (BMD).
    https://openneuro.org/datasets/ds005165

"""

import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', type=str, default='s3d')
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--bmd_dir', default='/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the training and test stimulus features and fMRI responses
# =============================================================================
# Train stimulus features
feature_dir = os.path.join(args.berg_dir, 'results', 'stimulus_features',
    'modality-fmri', 'train_dataset-bmd', 'model-'+args.model,
    'pca_stimulus_features_train.npy')
fmaps_train = np.load(feature_dir)

# Test stimulus features
feature_dir = os.path.join(args.berg_dir, 'results', 'stimulus_features',
    'modality-fmri', 'train_dataset-bmd', 'model-'+args.model,
    'pca_stimulus_features_test.npy')
fmaps_test = np.load(feature_dir)


# =============================================================================
# Load the training fMRI responses
# =============================================================================
# Load the fMRI responses for the training images
fmri_file = os.path.join(args.bmd_dir, 'derivatives', 'versionB', 'MNI152',
    'GLM', f'sub-{args.subject:02d}', 'prepared_betas',
    f'sub-{args.subject:02d}_organized_betas_task-train_normalized.pkl')
with open(fmri_file, "rb") as f:
    fmri_train = pickle.load(f)[0]

# Average the responses across repeats
fmri_train = np.mean(fmri_train, 1)


# =============================================================================
# Train the encoding models
# =============================================================================
# Set NaN values (missing fMRI data) to zero
fmri_train = np.nan_to_num(fmri_train)

# Train encoding models using the NSD-core subject-unique images: fit the
# regression models at each fMRI vertex
reg = LinearRegression().fit(fmaps_train, fmri_train)

# Use the learned weights to generate in silico fMRI responses for the test
# images
betas_test_pred = reg.predict(fmaps_test)

# Save the in silico fMRI responses for the test images
save_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-fmri', 'train_dataset-bmd', 'model-'+args.model)
file_name = f'betas_test_pred_sub-{args.subject:02d}.npy'
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, file_name), betas_test_pred)


# =============================================================================
# Save the trained encoding models
# =============================================================================
weights = {
    'coef_': reg.coef_,
    'intercept_': reg.intercept_,
    'n_features_in_': reg.n_features_in_
    }

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-bmd', 'model-'+args.model, 'encoding_models_weights')
os.makedirs(save_dir, exist_ok=True)

file_name = f'weights_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), weights)
