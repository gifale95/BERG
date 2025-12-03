"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : str
    Which subject's data to use (e.g., 'sub-01').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).

Example usage:
python berg_creation_code/03_test_encoding_models/train_dataset-things_fmri_1/01_test_encoding.py \
    --subject sub-01 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator'
"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=str, required=True,
                   help="Subject ID (e.g., 'sub-01')")
parser.add_argument('--berg_dir', required=True, type=str)

args = parser.parse_args()

print('>>> Test THINGS fMRI encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the responses metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_fmri_1')

metadata_path = os.path.join(data_dir, f'fmri_{args.subject}_metadata.npy')
metadata_fmri = np.load(metadata_path, allow_pickle=True).item()


# =============================================================================
# Load the in vivo neural responses for the test images
# =============================================================================
neural_test_path = os.path.join(data_dir, f'fmri_{args.subject}_split-test_averaged.h5')
with h5py.File(neural_test_path, 'r') as f:
    neural_test = f['neural_data'][:]

print(f"Actual neural test data shape: {neural_test.shape}")


# =============================================================================
# Load the in silico neural responses for the test images
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-fmri', 'train_dataset-things_fmri_1', 'vit_b_32')

pred_path = os.path.join(results_dir, f'fmri_test_pred_{args.subject}.npy')
neural_test_pred = np.load(pred_path, allow_pickle=True)

print(f"Predicted neural test data shape: {neural_test_pred.shape}")


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
n_voxels = neural_test.shape[1]
correlation_results = np.zeros(n_voxels)

print("Computing voxel-wise correlations...")
for v in range(n_voxels):
    if v % 10000 == 0:
        print(f"  Processing voxel {v}/{n_voxels}")
    correlation_results[v] = pearsonr(neural_test[:, v], neural_test_pred[:, v])[0]

print(f"Correlation results shape: {correlation_results.shape}")
print(f"Correlation stats: mean={correlation_results.mean():.4f}, std={correlation_results.std():.4f}")
print(f"Correlation range: [{correlation_results.min():.4f}, {correlation_results.max():.4f}]")


# =============================================================================
# Compute percent noise ceiling 
# =============================================================================
noise_ceiling_testset = metadata_fmri['encoding_model']['noise_ceiling_singletrial']
noise_ceiling_r2 = noise_ceiling_testset / 100



# Clip negative correlations to 0 before squaring
correlation_results_clipped = np.clip(correlation_results, 0, None)
percent_noise_ceiling = (correlation_results_clipped**2 / noise_ceiling_r2) * 100

print(f"Percent noise ceiling shape: {percent_noise_ceiling.shape}")

# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
metadata = metadata_fmri.copy()
metadata['encoding_model'].update({
    'correlation_results': correlation_results,
    'percent_noise_ceiling': percent_noise_ceiling
})

# Save the metadata
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri_1', 'model-vit_b_32', 'metadata')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'metadata_{args.subject}.npy'
np.save(os.path.join(save_dir, file_name), metadata)

print(f"\nMetadata saved to: {os.path.join(save_dir, file_name)}")