"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : str
    Which subject's data to use ('P1', 'P2', 'P3', 'P4').
model : str
    Name of the used encoding model.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
only_cls : str
    If we should only use CLS token or all patches ('True' or 'False').
regression : str
    Type of regression used ('ridge' or 'linear').

Example usage:
python berg_creation_code/03_test_encoding_models/train_dataset-things_meg/01_test_encoding.py \
    --subject P1 \
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
parser.add_argument('--subject', type=str, required=True, choices=['P1', 'P2', 'P3', 'P4'])
parser.add_argument('--model', required=True, choices=["vit_b_32", "clip.vit_b_32"],
                   help="Selecting which model to use")
parser.add_argument('--only_cls', required=True, choices=["True", "False"],
                    help='If we should only use CLS token or all patches')
parser.add_argument('--regression', required=True, choices=["ridge", "linear"],
                   help="Select type of regression")
parser.add_argument('--berg_dir', required=True, type=str)

args = parser.parse_args()

args.only_cls = args.only_cls == "True"

print('>>> Test THINGS MEG encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the responses metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_meg')

metadata_path = os.path.join(data_dir, f'meg_{args.subject}_metadata.npz')
metadata_meg = np.load(metadata_path, allow_pickle=True)


# =============================================================================
# Load the in vivo neural responses for the test images
# =============================================================================
neural_test_path = os.path.join(data_dir, f'meg_{args.subject}_split-test_averaged.h5')
with h5py.File(neural_test_path, 'r') as f:
    neural_test = f['neural_data_averaged'][:]

print(f"Actual neural test data shape: {neural_test.shape}")


# =============================================================================
# Load the in silico neural responses for the test images
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-meg', 'train_dataset-things_meg', args.model)

cls_suffix = 'cls' if args.only_cls else 'all'
pred_path = os.path.join(results_dir, f'meg_test_pred_{args.regression}_{cls_suffix}_{args.subject}.npy')
neural_test_pred = np.load(pred_path, allow_pickle=True)

print(f"Predicted neural test data shape: {neural_test_pred.shape}")


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
# Correlate the in vivo and in silico neural responses
# Shape: (channels, timepoints)
correlation_results = np.zeros((neural_test.shape[1], neural_test.shape[2]))

for c in range(neural_test.shape[1]):  # channels
    for t in range(neural_test.shape[2]):  # timepoints
        correlation_results[c, t] = pearsonr(neural_test[:, c, t],
            neural_test_pred[:, c, t])[0]

print(f"Correlation results shape: {correlation_results.shape}")


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
metadata = {
    'correlation_results': correlation_results,
    **{key: metadata_meg[key] for key in metadata_meg.files}
}

# Save the metadata
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg', f'model-{args.model}', 'metadata')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'metadata_{args.regression}_{cls_suffix}_{args.subject}.npy'
np.save(os.path.join(save_dir, file_name), metadata)

print(f"Metadata saved to: {os.path.join(save_dir, file_name)}")