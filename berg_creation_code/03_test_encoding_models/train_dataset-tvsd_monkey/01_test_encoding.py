"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
monkey : str
	Which monkey's data to use ('monkeyN' or 'monkeyF').
model : str
	Name of the used encoding model.
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG
 
python berg_creation_code/03_test_encoding_models/train_dataset-tvsd_monkey/01_test_encoding.py --monkey monkeyF --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator'


"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--monkey', type=str, required=True, choices=['monkeyN', 'monkeyF'])
parser.add_argument('--model', type=str, default='clip_vit_b_32')
parser.add_argument('--berg_dir', required=True, type=str)
args = parser.parse_args()

print('>>> Test TVSD encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the responses metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
	'train_dataset-tvsd_monkey')

metadata_path = os.path.join(data_dir, f'tvsd_{args.monkey}_metadata.npz')
metadata_tvsd = np.load(metadata_path)


# =============================================================================
# Load the in vivo neural responses for the test images
# =============================================================================
neural_test_path = os.path.join(data_dir, f'tvsd_{args.monkey}_split-test_averaged.h5')
with h5py.File(neural_test_path, 'r') as f:
	neural_test = f['neural_data_averaged'][:]

print(f"Actual neural test data shape: {neural_test.shape}")


# =============================================================================
# Load the in silico neural responses for the test images
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
	'modality-spike', 'train_dataset-tvsd_monkey', args.model)
pred_path = os.path.join(results_dir, f'spike_test_pred_{args.monkey}.npy')

neural_test_pred = np.load(pred_path)

print(f"Predicted neural test data shape: {neural_test_pred.shape}")


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
# Correlate the in vivo and in silico neural responses
correlation_results = np.zeros((neural_test.shape[2], neural_test.shape[1]))

for e in range(neural_test.shape[2]):  # electrodes
	for t in range(neural_test.shape[1]):  # timepoints
		correlation_results[e, t] = pearsonr(neural_test[:, t, e],
			neural_test_pred[:, t, e])[0]

print(f"Correlation results shape: {correlation_results.shape}")


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
metadata = {}

# Neural-related metadata
neural = {
	'times': metadata_tvsd['times'],
	'n_electrodes': int(metadata_tvsd['n_electrodes']),
	'monkey_id': str(metadata_tvsd['monkey_id']),
	'SNR': metadata_tvsd['SNR'],
	'SNR_max': metadata_tvsd['SNR_max'],
	'oracle': metadata_tvsd['oracle']
}
metadata['neural'] = neural

# Encoding-models-related metadata
encoding_models = {
	'correlation_results': correlation_results,
	'train_img_ids': metadata_tvsd['train_img_ids'],
	'train_img_files': metadata_tvsd['train_img_files'],
	'train_img_concepts': metadata_tvsd['train_img_concepts'],
	'test_avg_img_ids': metadata_tvsd['test_avg_img_ids'],
	'test_avg_img_files': metadata_tvsd['test_avg_img_files'],
	'test_avg_img_concepts': metadata_tvsd['test_avg_img_concepts']
}
metadata['encoding_models'] = encoding_models

# Save the metadata
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
	'train_dataset-tvsd_monkey', f'model-{args.model}', 'metadata')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = f'metadata_{args.monkey}.npy'
np.save(os.path.join(save_dir, file_name), metadata)

print(f"Metadata saved to: {os.path.join(save_dir, file_name)}")