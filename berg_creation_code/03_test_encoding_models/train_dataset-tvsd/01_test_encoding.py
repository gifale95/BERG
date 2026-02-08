"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
monkey : str
	Which monkey's data to use ('monkeyN' or 'monkeyF').
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG
train_split : str
	Which training split to test (default: 'all_training_splits').
"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--monkey', type=str, required=True, choices=['monkeyN', 'monkeyF'])
parser.add_argument('--berg_dir', required=True, type=str)
parser.add_argument('--train_split', type=str, default='all_training_splits',
                   choices=['all_training_splits', 'single_training_split_1', 'single_training_split_2', 'single_training_split_3', 'single_training_split_4'],
                   help='Which training split to test')

args = parser.parse_args()

print('>>> Test TVSD encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the responses metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
	'train_dataset-tvsd')

metadata_path = os.path.join(data_dir, f'tvsd_{args.monkey}_metadata.npy')
metadata_tvsd = np.load(metadata_path, allow_pickle=True).item()


# =============================================================================
# Load the in vivo neural responses for the test images
# =============================================================================
neural_test_path = os.path.join(data_dir, f'tvsd_{args.monkey}_split-test_averaged.h5')
with h5py.File(neural_test_path, 'r') as f:
	neural_test = f['neural_data'][:]

print(f"Actual neural test data shape: {neural_test.shape}")


# =============================================================================
# Load the in silico neural responses for the test images
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
	'modality-utah_array', 'train_dataset-tvsd', 'vit_b_32')

pred_path = os.path.join(results_dir, f'utah_array_test_pred_{args.monkey}_{args.train_split}.npy')
neural_test_pred = np.load(pred_path)

print(f"Predicted neural test data shape: {neural_test_pred.shape}")


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
correlation_results = np.zeros((neural_test.shape[1], neural_test.shape[2]))  

for e in range(neural_test.shape[1]):
	for t in range(neural_test.shape[2]):
		correlation_results[e, t] = pearsonr(neural_test[:, e, t],
			neural_test_pred[:, e, t])[0]

print(f"Correlation results shape: {correlation_results.shape}")


# =============================================================================
# Compute percent noise ceiling
# =============================================================================
noise_ceiling = metadata_tvsd['encoding_model']['noise_ceiling']

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
# Load the existing metadata file (which contains all splits)
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-utah_array',
	'train_dataset-tvsd', 'model-vit_b_32', 'metadata')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = f'metadata_monkey{args.monkey}.npy'
metadata_file_path = os.path.join(save_dir, file_name)

# Load existing metadata if it exists, otherwise use the one we loaded earlier
if os.path.exists(metadata_file_path):
	metadata = np.load(metadata_file_path, allow_pickle=True).item()
else:
	metadata = metadata_tvsd.copy()

# Update the specific split's metadata with encoding results
metadata['encoding_model'][args.train_split].update({
    'correlation_results': correlation_results,
    'percent_noise_ceiling': percent_noise_ceiling
})

# Save back to the single metadata file
np.save(metadata_file_path, metadata)

print(f"Metadata saved to: {metadata_file_path}")