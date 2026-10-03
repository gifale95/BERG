"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : int
    Number of the used THINGS MEG1 subject.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the MEG metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_meg_1')
metadata_dir = os.path.join(data_dir, 'meg_P'+str(args.subject)+
    '_metadata.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()


# =============================================================================
# Load and format the in vivo MEG responses for the test images
# =============================================================================
# Load the in vivo MEG responses
meg_dir = os.path.join(data_dir, 'meg_P'+str(args.subject)+
    '_split-test.h5')
meg_test_all_rep = h5py.File(meg_dir, 'r')['neural_data'][:]
n_sensors = metadata['sensors']['n_sensors']
n_times = len(metadata['meg']['times'])

# Get the test image IDs
test_img_ids = metadata['encoding_model']['test_things_img_ids'].astype(int)
unique_test_img_ids = np.unique(test_img_ids)

# Average the MEG responses across repetitions
meg_test = np.zeros((len(unique_test_img_ids), n_sensors, n_times))
for i, img_id in enumerate(unique_test_img_ids):
    meg_test[i] = np.mean(meg_test_all_rep[test_img_ids == img_id], 0)


# =============================================================================
# Compute the NCSNR and noise ceiling on the test MEG responses
# =============================================================================
# Reshape the MEG data to (Samples, Features)
meg_test_all_rep = np.reshape(meg_test_all_rep, (len(meg_test_all_rep), -1))

# Estimate the noise standard deviation (calculate the variance of the
# responses across the 12 presentations of each test image).
var = []
for img in unique_test_img_ids:
    idx = np.where(test_img_ids == img)[0]
    var.append(np.nanvar(meg_test_all_rep[idx], axis=0, ddof=1))
# Average the variance across images and compute the square root of the
# result
sigma_noise = np.sqrt(np.nanmean(var, 0))

# Estimate the signal standard deviation (total variance - noise variance)
tot_var_data = np.nanvar(meg_test_all_rep, axis=0, ddof=1)
sigma_signal = tot_var_data - (sigma_noise ** 2)
sigma_signal[sigma_signal<0] = 0
sigma_signal = np.sqrt(sigma_signal)

# Compute the ncsnr
ncsnr = sigma_signal / sigma_noise

# Convert the ncsnr to noise ceiling (the noise ceiling is in r² explained
# variance units)
img_reps = 12
noise_ceiling = 100 * (ncsnr ** 2) / ((ncsnr ** 2) + (1 / img_reps))

# Reshape the scores to (n_sensors, n_times)
ncsnr = ncsnr.reshape(n_sensors, n_times)
noise_ceiling = noise_ceiling.reshape(n_sensors, n_times)


# =============================================================================
# Load the in silico MEG responses for the test images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-meg', 'train_dataset-things_meg_1', 'model-vit_b_32',
    'meg_test_pred_P' + str(args.subject) + '.npy')

meg_test_pred = np.load(data_dir, allow_pickle=True).item()
meg_test_pred_single_splits = meg_test_pred['split-single']
meg_test_pred_all_splits = meg_test_pred['split-all']
n_splits = 4
del meg_test_pred


# =============================================================================
# Compute the encoding accuracy (single split models)
# =============================================================================
# Correlate the in vivo and in silico fMRI responses (averaged across
# repetitions)
correlation_single_splits_avg_rep = np.zeros((n_sensors, n_times),
    dtype=np.float32)
for t in range(n_times):
    for s in range(n_sensors):
        correlation_single_splits_avg_rep[s,t] = pearsonr(meg_test[:,s,t],
            np.mean(meg_test_pred_single_splits[:,:,s,t], 1))[0]

# Correlate the in vivo and in silico fMRI responses (for single repetitions)
correlation_single_splits_single_rep = np.zeros((n_splits, n_sensors,
    n_times), dtype=np.float32)
for r in range(n_splits):
    for s in range(n_sensors):
        for t in range(n_times):
            correlation_single_splits_single_rep[r,s,t] = pearsonr(
                meg_test[:,s,t], meg_test_pred_single_splits[:,r,s,t])[0]


# =============================================================================
# Compute the encoding accuracy (all split models)
# =============================================================================
correlation_all_splits = np.zeros((n_sensors, n_times), dtype=np.float32)
for t in range(n_times):
    for s in range(n_sensors):
        correlation_all_splits[s,t] = pearsonr(meg_test[:,s,t],
            meg_test_pred_all_splits[:,s,t])[0]


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
encoding_accuracy_new = {
    'correlation_single_splits_avg_rep': correlation_single_splits_avg_rep,
    'correlation_single_splits_single_rep': correlation_single_splits_single_rep,
    'correlation_all_splits': correlation_all_splits,
    'ncsnr': ncsnr,
    'noise_ceiling': noise_ceiling
    }
metadata['encoding_accuracy_new'] = encoding_accuracy_new

# Save the metadata
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', 'model-vit_b_32', 'metadata')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
file_name = 'metadata_P' + str(args.subject) + '.npy'
np.save(os.path.join(save_dir, file_name), metadata)