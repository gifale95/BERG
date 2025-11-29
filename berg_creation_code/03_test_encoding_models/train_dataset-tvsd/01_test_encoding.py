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
 


python berg_creation_code/03_test_encoding_models/train_dataset-tvsd_monkey/01_test_encoding.py \
    --monkey monkeyN \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --only_cls True \
    --regression linear \
    --model clip.vit_b_32

"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--monkey', type=str, required=True, choices=['monkeyN', 'monkeyF'])
parser.add_argument('--model', required=True, choices=["vit_b_32", "clip.vit_b_32"],
                   help="Selecting which model to use")
parser.add_argument('--only_cls', required=True, choices=["True", "False"],
                    help='If we should only use CLS token or all patches')
parser.add_argument('--regression', required=True, choices=["ridge", "linear"],
                   help="Select type of regression")
parser.add_argument('--berg_dir', required=True, type=str)

args = parser.parse_args()

args.only_cls = args.only_cls == "True"

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
	'modality-utah_array', 'train_dataset-tvsd', args.model)


cls_suffix = 'cls' if args.only_cls else 'all'
pred_path = os.path.join(results_dir, f'utah_array_test_pred_{args.regression}_{cls_suffix}_{args.monkey}.npy')
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
percent_noise_ceiling = (correlation_results**2 / noise_ceiling_r2) * 100

print(f"Percent noise ceiling shape: {percent_noise_ceiling.shape}")


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================


metadata = metadata_tvsd.copy()
metadata['encoding_model'].update({
    'correlation_results': correlation_results,
    'percent_noise_ceiling': percent_noise_ceiling
})

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-utah_array',
	'train_dataset-tvsd', f'model-{args.model}', 'metadata')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = f'metadata_{args.monkey}.npy'
np.save(os.path.join(save_dir, file_name), metadata)

print(f"Metadata saved to: {os.path.join(save_dir, file_name)}")