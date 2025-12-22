"""Test encoding fusion models: THINGS invivo EEG → insilico fMRI.

Tests linear regression models that predict fMRI responses from EEG data.
Concatenates all 10 EEG subjects (630 channels), applies StandardScaler and PCA
transformations learned during training, then uses trained linear models to predict fMRI.


Required Data we already have:
------------------------------
1. THINGS invivo EEG: {nest_dir}/model_training_datasets/train_dataset-things_eeg_2/
   Files: eeg_sub-{01-10}_split-test.h5
-> Using berg_creation_code/01_prepare_data/train_dataset-things_eeg_2/prepare_things_eeg_2.py

2. EEG metadata (times): {nest_dir}/model_training_datasets/train_dataset-things_eeg_2/
   Files: metadata_subject-01.npy
-> Using berg_creation_code/01_prepare_data/train_dataset-things_eeg_2/prepare_things_eeg_2.py

3. THINGS insilico fMRI: {nest_dir}/results/paper_analyses/encoding_fusion/in_silico_fmri_responses/things/ 
-> Using 001_things_insilico_fmri.py

"""

import argparse
import numpy as np
import h5py
import os
from tqdm import tqdm
from berg import BERG
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--tot_vertex_splits', type=int, default=14)
parser.add_argument('--model_name', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--experiment_name', type=str, default='default_experiment')
parser.add_argument('--nest_dir', default='/scratch/giffordale95/projects/neural_encoding_simulation_toolkit', type=str)
args = parser.parse_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Select the test image conditions
# =============================================================================
# Its just the first 200 images
test_img = list(range(0, 200))
print(f"THINGS test images: {len(test_img)}")

# =============================================================================
# Load EEG times from metadata
# =============================================================================

# For in silico EEG, get times from BERG model metadata
nest_object = BERG(args.nest_dir)
metadata = nest_object.get_model_metadata("eeg-things_eeg_2-vit_b_32", subject=1) # We can use any subject since its always the same
times = metadata['eeg']['times']
print(f"EEG timepoints: {len(times)}")



# =============================================================================
# Load and concatenate invivo EEG test data from all 10 subjects
# =============================================================================
print("Loading invivo THINGS EEG test data from all 10 subjects...")

# Load first subject to get dimensions
data_dir = os.path.join(args.nest_dir, 'model_training_datasets', 
    'train_dataset-things_eeg_2', 'eeg_sub-01_split-test.h5')

with h5py.File(data_dir, 'r') as f:
    first_eeg = f['eeg'][:]
    # Average across repeat dimension
    first_eeg = np.mean(first_eeg, axis=1).astype(np.float32)

first_eeg = first_eeg[test_img]
n_images, n_channels, n_timepoints = first_eeg.shape
print(f"EEG dimensions: {n_images} images, {n_channels} channels, {n_timepoints} timepoints")

# Pre-allocate array for concatenated EEG: (n_images, 630 channels, n_timepoints)
eeg = np.zeros((n_images, n_channels * 10, n_timepoints), dtype=np.float32)

# Fill in first subject
eeg[:, :n_channels, :] = first_eeg
del first_eeg

# Load and concatenate remaining subjects
for i, subj in enumerate(range(2, 11), 1):
    start_idx = i * n_channels
    end_idx = (i + 1) * n_channels
    
    data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{subj:02d}_split-test.h5')
    
    with h5py.File(data_dir, 'r') as f:
        eeg_subj = f['eeg'][:]
        # Average across repeat dimension
        eeg_subj = np.mean(eeg_subj, axis=1).astype(np.float32)
    
    eeg[:, start_idx:end_idx, :] = eeg_subj[test_img]
    del eeg_subj

print(f"Concatenated EEG shape: {eeg.shape}")

# =============================================================================
# Load in silico fMRI ground truth for THINGS test images
# =============================================================================
print("Loading in silico fMRI ground truth...")

data_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'in_silico_fmri_responses', 'things',
    args.hemisphere+'_sub-'+format(args.fmri_subject, '02')+'_test.h5')

with h5py.File(data_dir, 'r') as f:
    fmri_ground_truth = f['insilico_responses'][test_img].astype(np.float32)

print(f"fMRI ground truth shape: {fmri_ground_truth.shape}")

# =============================================================================
# Generate encoding fusion fMRI predictions
# =============================================================================
print("Generating fMRI predictions from EEG...")

# Directory where trained models are stored
model_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'trained_models', args.experiment_name,
    str(args.model_name), 'aggr-append', 'regression-linear')

# Loop across EEG time points
fmri_predictions = []
n_vertices = 163842

for t in tqdm(range(n_timepoints), desc="Predicting per timepoint"):
    # Get EEG data for this timepoint: (n_images, 630)
    eeg_t = eeg[:, :, t]
    
    # Load scaler and PCA parameters from first vertex split
    # -> transformations are same across all vertices and only regression weights differ
    weights_file = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_' + \
                   f'vertices-001_eeg_append_time-{t+1:03d}.npy'
    weights_path = os.path.join(model_dir, weights_file)
    
    weights = np.load(weights_path, allow_pickle=True).item()
    
    # Apply StandardScaler transformation
    scaler = StandardScaler()
    scaler.scale_ = weights['scaler_param']['scale_']
    scaler.mean_ = weights['scaler_param']['mean_']
    scaler.var_ = weights['scaler_param']['var_']
    scaler.n_features_in_ = weights['scaler_param']['n_features_in_']
    scaler.n_samples_seen_ = weights['scaler_param']['n_samples_seen_']
    eeg_t = scaler.transform(eeg_t).astype(np.float32)
    
    # Apply PCA transformation
    pca = PCA(n_components=weights['pca_param']['n_components_'])
    pca.components_ = weights['pca_param']['components_']
    pca.mean_ = weights['pca_param']['mean_']
    pca.explained_variance_ = weights['pca_param']['explained_variance_']
    pca.explained_variance_ratio_ = weights['pca_param']['explained_variance_ratio_']
    pca.singular_values_ = weights['pca_param']['singular_values_']
    pca.n_components_ = weights['pca_param']['n_components_']
    pca.n_samples_ = weights['pca_param']['n_samples_']
    pca.noise_variance_ = weights['pca_param']['noise_variance_']
    pca.n_features_in_ = weights['pca_param']['n_features_in_']
    eeg_t = pca.transform(eeg_t).astype(np.float32)
    
    # Predict fMRI for all vertices
    fmri_pred_vertices = np.zeros((n_images, n_vertices), dtype=np.float32)
    
    # Loop through vertex splits
    for v in range(args.tot_vertex_splits):
        # Load regression weights for this vertex split
        weights_file = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_' + \
                       f'vertices-{v+1:03d}_eeg_append_time-{t+1:03d}.npy'
        weights_path = os.path.join(model_dir, weights_file)
        
        weights = np.load(weights_path, allow_pickle=True).item()
        
        # Reconstruct linear regression model
        reg = LinearRegression()
        reg.coef_ = weights['reg_param']['coef_']
        reg.intercept_ = weights['reg_param']['intercept_']
        reg.n_features_in_ = weights['reg_param']['n_features_in_']
        
        # Predict fMRI for this vertex split
        pred = reg.predict(eeg_t).astype(np.float32)
        
        # Store predictions in correct vertex range
        vertices_per_split = int(np.ceil(n_vertices / args.tot_vertex_splits))
        vertex_start = vertices_per_split * v
        vertex_end = vertex_start + vertices_per_split
        fmri_pred_vertices[:, vertex_start:vertex_end] = pred
        
        del pred, weights, reg
    
    # Store predictions for this timepoint
    fmri_predictions.append(fmri_pred_vertices)
    del fmri_pred_vertices

# Convert to numpy array: (timepoints, images, vertices)
fmri_predictions = np.array(fmri_predictions, dtype=np.float32)
print(f"Final predictions shape: {fmri_predictions.shape}")

# =============================================================================
# Save the predicted fMRI responses
# =============================================================================
save_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'test_results', args.experiment_name, 
    str(args.model_name), 'aggr-append', 'regression-linear')

os.makedirs(save_dir, exist_ok=True)

# Save predictions in HDF5 format
h5_file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_predictions.h5'
h5_save_path = os.path.join(save_dir, h5_file_name)

with h5py.File(h5_save_path, 'w') as h5f:
    h5f.create_dataset('predicted_responses', data=fmri_predictions, 
                       compression='gzip', compression_opts=9)

print(f"Predicted responses saved to: {h5_save_path}")
print(f"Shape: {fmri_predictions.shape} (timepoints × images × vertices)")

# =============================================================================
# Correlate predictions with ground truth
# =============================================================================
print("Computing correlations with in silico fMRI ground truth...")

# Correlation matrix: (timepoints, vertices)
correlations = np.zeros((n_timepoints, n_vertices), dtype=np.float32)

# Loop over EEG time points and fMRI vertices
for t in tqdm(range(n_timepoints), desc="Computing correlations"):
    for v in range(n_vertices):
        correlations[t, v] = pearsonr(fmri_predictions[t, :, v],
                                      fmri_ground_truth[:, v])[0]

print(f"Correlations shape: {correlations.shape}")

# =============================================================================
# Save the correlation results
# =============================================================================
results = {
    'correlations': correlations,
    'times': times
}

file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}.npy'
results_path = os.path.join(save_dir, file_name)

np.save(results_path, results)