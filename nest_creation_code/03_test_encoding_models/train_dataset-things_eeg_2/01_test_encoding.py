"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : int
	Number of the used NSD subject.
model : str
	Name of the used encoding model.
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the responses metadata
# =============================================================================
data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
	'train_dataset-things_eeg_2')

metadata_things_eeg2 = np.load(os.path.join(data_dir, 'metadata_subject-'+
	str(args.subject)+'.npy'), allow_pickle=True).item()


# =============================================================================
# Load the in vivo EEG responses for the test images
# =============================================================================
eeg_dir = os.path.join(data_dir, 'eeg_sub-'+format(args.subject,'02')+
	'_split-test.h5')
eeg_test = h5py.File(eeg_dir, 'r')['eeg'][:]

# Average the EEG responses for the test images across repetitions
eeg_test = np.mean(eeg_test, 1)


# =============================================================================
# Load the in silico EEG responses for the test images
# =============================================================================
data_dir = os.path.join(args.nest_dir, 'results', 'test_encoding_models',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-'+args.model)
pred_dir = os.path.join(data_dir, 'eeg_test_pred_subject-'+
	str(args.subject)+'.npy')

eeg_test_pred = np.load(pred_dir)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
# Correlate the in vivo and in silico fMRI responses (averaged across
# repetitions)
correlation_averaged_repetitions = np.zeros((eeg_test_pred.shape[2],
	eeg_test_pred.shape[3]))
for t in range(eeg_test_pred.shape[3]):
	for c in range(eeg_test_pred.shape[2]):
		correlation_averaged_repetitions[c,t] = pearsonr(eeg_test[:,c,t],
			np.mean(eeg_test_pred[:,:,c,t], 1))[0]

# Correlate the in vivo and in silico fMRI responses (for single repetitions)
correlation_single_repetitions = np.zeros((eeg_test_pred.shape[1],
	eeg_test_pred.shape[2], eeg_test_pred.shape[3]))
for r in range(eeg_test_pred.shape[1]):
	for t in range(eeg_test_pred.shape[3]):
		for c in range(eeg_test_pred.shape[2]):
			correlation_single_repetitions[r,c,t] = pearsonr(eeg_test[:,c,t],
				eeg_test_pred[:,r,c,t])[0]


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
metadata = {}

# EEG-related metadata
eeg = {
	'ch_names': metadata_things_eeg2['ch_names'],
	'times': metadata_things_eeg2['times']
	}
metadata['eeg'] = eeg

# Encoding-models-related metadata
encoding_models = {
	'correlation_averaged_repetitions': correlation_averaged_repetitions,
	'correlation_single_repetitions': correlation_single_repetitions,
	'train_img_info': metadata_things_eeg2['train_img_info'],
	'test_img_info': metadata_things_eeg2['test_img_info']
	}
metadata['encoding_models'] = encoding_models

# Save the metadata
save_dir = os.path.join(args.nest_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-'+args.model, 'metadata')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
file_name = 'metadata_subject-' + format(args.subject, '02') + '.npy'
np.save(os.path.join(save_dir, file_name), metadata)
