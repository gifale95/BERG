"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : str
    Which subject's data to use ('P1', 'P2', 'P3', 'P4').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=str, required=True, choices=['P1', 'P2', 'P3', 'P4'])
parser.add_argument('--berg_dir', required=True, type=str)
parser.add_argument('--train_split', type=str, default='all_training_splits',
                   choices=['all_training_splits', 'single_training_split_1', 'single_training_split_2', 'single_training_split_3', 'single_training_split_4'],
                   help='Which training split to test')
args = parser.parse_args()

print('>>> Test THINGS MEG encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the responses metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_meg_1')

metadata_path = os.path.join(data_dir, f'meg_{args.subject}_metadata.npy')
metadata_meg = np.load(metadata_path, allow_pickle=True).item()


# =============================================================================
# Load the in vivo neural responses for the test images
# =============================================================================
neural_test_path = os.path.join(data_dir, f'meg_{args.subject}_split-test_averaged.h5')
with h5py.File(neural_test_path, 'r') as f:
    neural_test = f['neural_data'][:]


# =============================================================================
# Load the in silico neural responses for the test images
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-meg', 'train_dataset-things_meg_1', 'vit_b_32')

pred_path = os.path.join(results_dir, f'meg_test_pred_{args.subject}_{args.train_split}.npy')
neural_test_pred = np.load(pred_path, allow_pickle=True)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
correlation_results = np.zeros((neural_test.shape[1], neural_test.shape[2]))

for c in range(neural_test.shape[1]):  # channels
    for t in range(neural_test.shape[2]):  # timepoints
        correlation_results[c, t] = pearsonr(neural_test[:, c, t],
            neural_test_pred[:, c, t])[0]


# =============================================================================
# Compute percent noise ceiling
# =============================================================================
noise_ceiling = metadata_meg['encoding_model']['noise_ceiling']

noise_ceiling_r2 = noise_ceiling / 100 

# Clip negative correlations to 0 before squaring
correlation_results_clipped = np.clip(correlation_results, 0, None)
percent_noise_ceiling = (correlation_results_clipped**2 / noise_ceiling_r2) * 100

# If noise ceiling = 0
percent_noise_ceiling[~np.isfinite(percent_noise_ceiling)] = np.nan

print(f"Percent noise ceiling shape: {percent_noise_ceiling.shape}")


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', 'model-vit_b_32', 'metadata')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'metadata_{args.subject}.npy'
file_path = os.path.join(save_dir, file_name)

# Load existing metadata or create from base
if os.path.exists(file_path):
    metadata = np.load(file_path, allow_pickle=True).item()
else:
    metadata = metadata_meg.copy()

# Add results to the appropriate split
if args.train_split not in metadata['encoding_model']:
    metadata['encoding_model'][args.train_split] = {}

metadata['encoding_model'][args.train_split]['correlation_results'] = correlation_results
metadata['encoding_model'][args.train_split]['percent_noise_ceiling'] = percent_noise_ceiling

np.save(file_path, metadata)