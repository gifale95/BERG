"""Train the encoding fusion models by linearly mapping in vivo EEG responses
onto in silico fMRI responses for the 16,540 THINGS EEG2 train images.

One regression model is trained for each fMRI vertex and EEG time point,
using the EEG channel responses appended across the 10 THINGS EEG2 subjects.

Required Data:
--------------
1. THINGS invivo EEG: {nest_dir}/model_training_datasets/train_dataset-things_eeg_2/
   Files: eeg_sub-{01-10}_split-train.h5
-> Using berg_creation_code/01_prepare_data/train_dataset-things_eeg_2/prepare_things_eeg_2.py

2. THINGS insilico fMRI: {nest_dir}/results/paper_analyses/encoding_fusion/in_silico_fmri_responses/things/ 
-> Using 001_things_insilico_fmri.py


"""

import argparse
import numpy as np
import h5py
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--tot_vertex_splits', type=int, default=14)
parser.add_argument('--vertex_split', type=int, default=1)
parser.add_argument('--tot_time_splits', type=int, default=1)
parser.add_argument('--time_split', type=int, default=1)
parser.add_argument('--model_name', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--experiment_name', type=str, default='default_experiment')
parser.add_argument('--nest_dir', default='/scratch/giffordale95/projects/neural_encoding_simulation_toolkit', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset/', type=str)
args = parser.parse_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Select the train image conditions
# =============================================================================
train_img = np.arange(16540)
train_img.sort()
print(f"Training images: {train_img.shape}")

# =============================================================================
# Define the fMRI vertex split being processed
# =============================================================================
n_vertices = 163842
vertices_per_split = int(np.ceil(n_vertices / args.tot_vertex_splits))
vertex_start = vertices_per_split * (args.vertex_split - 1)
vertex_end = vertex_start + vertices_per_split

print(f"Vertex range: {vertex_start} to {vertex_end}")

# =============================================================================
# Load the insilico fMRI responses for THINGS
# =============================================================================
data_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'in_silico_fmri_responses', 'things',
    args.hemisphere+'_sub-'+format(args.fmri_subject, '02')+'_train.h5')

with h5py.File(data_dir, 'r') as f:
    fmri_all = f['insilico_responses'][:]  # Load all data

# Select training images and vertices
fmri = fmri_all[train_img, vertex_start:vertex_end].astype(np.float32)
print(f"fMRI data shape: {fmri.shape}")

# =============================================================================
# Load and concatenate invivo EEG data from all 10 subjects
# =============================================================================
print("Loading invivo THINGS EEG data from all 10 subjects...")

# Load first subject to get dimensions
data_dir = os.path.join(args.nest_dir, 'model_training_datasets', 
    'train_dataset-things_eeg_2', 'eeg_sub-01_split-train.h5')

with h5py.File(data_dir, 'r') as f:
    first_eeg = f['eeg'][:]
    # Average across repeat dimension
    first_eeg = np.mean(first_eeg, axis=1).astype(np.float32)

first_eeg = first_eeg[train_img]
n_images, n_channels, n_timepoints = first_eeg.shape
print(f"EEG dimensions: {n_images} images, {n_channels} channels, {n_timepoints} timepoints")

# Pre-allocate array for concatenated EEG: (n_images, 630 channels, n_timepoints)
eeg_concat = np.zeros((n_images, n_channels * 10, n_timepoints), dtype=np.float32)

# Fill in first subject
eeg_concat[:, :n_channels, :] = first_eeg
del first_eeg

# Load and concatenate remaining subjects
for i, subj in enumerate(range(2, 11), 1):
    start_idx = i * n_channels
    end_idx = (i + 1) * n_channels
    
    data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{subj:02d}_split-train.h5')
    
    with h5py.File(data_dir, 'r') as f:
        eeg_subj = f['eeg'][:]
        # Average across repeat dimension
        eeg_subj = np.mean(eeg_subj, axis=1).astype(np.float32)
    
    eeg_concat[:, start_idx:end_idx, :] = eeg_subj[train_img]
    del eeg_subj
    
eeg = eeg_concat

print(f"Concatenated EEG shape: {eeg.shape}")


# =============================================================================
# Create the saving directory
# =============================================================================
save_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'trained_models', args.experiment_name,
    str(args.model_name), 'aggr-append', 'regression-linear')

os.makedirs(save_dir, exist_ok=True)
print(f"Saving to: {save_dir}")

# =============================================================================
# Define the EEG time split being processed
# =============================================================================
times_per_split = int(np.ceil(eeg.shape[2] / args.tot_time_splits))
time_start = times_per_split * (args.time_split - 1)
time_end = time_start + times_per_split

print(f"Processing timepoints {time_start} to {time_end}")

# =============================================================================
# Train linear regression models per timepoint
# =============================================================================
print("Training linear regression models...")

for t in tqdm(range(time_start, time_end), desc="Training per timepoint"):
    # Get EEG data for this timepoint
    eeg_t = eeg[:, :, t]
    
    # Apply StandardScaler
    scaler = StandardScaler()
    eeg_t = scaler.fit_transform(eeg_t).astype(np.float32)
    
    # Apply PCA (no reduction)
    pca = PCA(n_components=eeg_t.shape[1]) 
    eeg_t = pca.fit_transform(eeg_t).astype(np.float32)  # Shape: (72,000, 63)

    reg = LinearRegression().fit(eeg_t, fmri)

    weights = {
        'scaler_param': {
            'scale_': scaler.scale_.astype(np.float32),
            'mean_': scaler.mean_.astype(np.float32),
            'var_': scaler.var_.astype(np.float32),
            'n_features_in_': scaler.n_features_in_,
            'n_samples_seen_': scaler.n_samples_seen_
        },
        'pca_param': {
            'components_': pca.components_.astype(np.float32),
            'explained_variance_': pca.explained_variance_.astype(np.float32),
            'explained_variance_ratio_': pca.explained_variance_ratio_.astype(np.float32),
            'singular_values_': pca.singular_values_.astype(np.float32),
            'mean_': pca.mean_.astype(np.float32),
            'n_components_': pca.n_components_,
            'n_samples_': pca.n_samples_,
            'noise_variance_': pca.noise_variance_ if pca.noise_variance_ is None else np.float32(pca.noise_variance_),
            'n_features_in_': pca.n_features_in_
        },
        'reg_param': {
            'coef_': reg.coef_.astype(np.float32),
            'intercept_': reg.intercept_.astype(np.float32),
            'n_features_in_': reg.n_features_in_
        }
    }

    
    # Save the trained model
    file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_' + \
                f'vertices-{args.vertex_split:03d}_eeg_append_' + \
                f'time-{t+1:03d}.npy'
    
    np.save(os.path.join(save_dir, file_name), weights)
    
    del weights, reg, scaler