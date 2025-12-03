"""Train a regression model to predict THINGS fMRI neural data using vision transformer feature maps as predictors. The regression is trained 
using the training images neural data (Y) and feature maps (X).

The feature maps come from a vision transformer, and are downsampled to 
250 principal components using PCA.

Pipeline steps:
1. Model setup: Load ViT-B/32, wrap with torchextractor for multi-layer access
2. Feature extraction: Extract tokens from 12 transformer layers (0-11)
   - Batch processing (512 images) to handle training + test images
3. Preprocessing: StandardScaler normalization + PCA reduction to 250 components
   - Decision: PCA reduces features for model stability
4. Neural data loading: Load all training data at once
5. Model training: Single Linear model for all voxels
6. Prediction: Generate test predictions
7. Output: Neural predictions (100 test images × 211,339 voxels) + saved models

Parameters
----------
subject : str
    Which subject's data to use (e.g., 'sub-01').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
things_dir : str
    Directory of the THINGS images.
feature_batch_size : int
    Batch size for feature extraction (default: 512).
n_pca_components : int
    Number of PCA components (default: 250).

Example usage:
python berg_creation_code/02_train_encoding_models/train_dataset-things_fmri_1/train_encoding.py \
    --subject sub-01 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --things_dir '/Volumes/Extreme SSD/Datasets/THINGS/things_images'
"""

import argparse
import torch
import numpy as np
import os
import h5py
from tqdm import tqdm
import copy
from PIL import Image
import torchextractor as tx
from torchvision.models import vit_b_32, ViT_B_32_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True, type=str,
                   help="Select which subject's data to use (e.g., 'sub-01').")
parser.add_argument('--berg_dir', required=True, type=str,
                   help="Directory of the BERG framework.")
parser.add_argument('--things_dir', required=True, type=str,
                   help="Directory of the things images.")
parser.add_argument('--feature_batch_size', type=int, default=512,
                   help='Batch size for feature extraction')
parser.add_argument('--n_pca_components', type=int, default=250,
                   help='Number of PCA components')
args = parser.parse_args()

print('>>> Train THINGS fMRI encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
torch.manual_seed(seed)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# =============================================================================
# Vision model
# =============================================================================
# Load the model
weights = ViT_B_32_Weights.DEFAULT
model = vit_b_32(weights=weights)
model.to(device)
model.eval()

layer_names = ['encoder.layers.encoder_layer_0',
                'encoder.layers.encoder_layer_1',
                'encoder.layers.encoder_layer_2',
                'encoder.layers.encoder_layer_3',
                'encoder.layers.encoder_layer_4',
                'encoder.layers.encoder_layer_5',
                'encoder.layers.encoder_layer_6',
                'encoder.layers.encoder_layer_7',
                'encoder.layers.encoder_layer_8',
                'encoder.layers.encoder_layer_9',
                'encoder.layers.encoder_layer_10',
                'encoder.layers.encoder_layer_11']

visual = tx.Extractor(model, layer_names)

# Use official preprocessing
preprocess = weights.transforms()



# =============================================================================
# Load metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-things_fmri_1')
metadata_path = os.path.join(data_dir, f'fmri_{args.subject}_metadata.npy')
metadata = np.load(metadata_path, allow_pickle=True).item()


# =============================================================================
# Extract the THINGS fMRI training image features
# =============================================================================
n_train_images = len(metadata["encoding_model"]['train_stimuli']) 
fmaps_train = []

for start_idx in tqdm(range(0, n_train_images, args.feature_batch_size), leave=False):
    end_idx = min(start_idx + args.feature_batch_size, n_train_images)
    batch_images = []
    
    # Load batch of training images
    for i in range(start_idx, end_idx):
        concept = metadata["encoding_model"]['train_concepts'][i]
        stimulus = metadata["encoding_model"]['train_stimuli'][i]
        full_path = os.path.join(args.things_dir, concept, stimulus)
        img = Image.open(full_path).convert('RGB')
        img_tensor = preprocess(img)
        batch_images.append(img_tensor)
    
    # Process batch
    batch_tensor = torch.stack(batch_images).to(device)
    
    with torch.no_grad():
        # Extract features using torchextractor
        _, features = visual(batch_tensor)
        
        # Extract all tokens from each layer and concatenate
        batch_features = []
        for layer_name in layer_names:
            layer_features = features[layer_name]
            # Shape: (batch_size, seq_len=50, hidden_dim=768)
            # Flatten: (batch_size, seq_len * hidden_dim)
            model_features = layer_features.flatten(1, 2)
            batch_features.append(model_features)
        
        # Concatenate features from all layers
        ft = torch.cat(batch_features, dim=-1)  # Shape: (batch_size, 12*50*768)
        fmaps_train.append(ft.detach().cpu().numpy())

# Concatenate all training batches
fmaps_train = np.concatenate(fmaps_train, axis=0)

# Standardize the image features
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)

# Downsample the image features using PCA
pca = PCA(n_components=args.n_pca_components, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_train = fmaps_train.astype(np.float32)


# =============================================================================
# Extract the THINGS fMRI test image features
# =============================================================================
test_images = []
for i in range(len(metadata['encoding_model']['test_avg_stimuli'])):
    category = metadata['encoding_model']['test_avg_concepts'][i]
    image_file = metadata['encoding_model']['test_avg_stimuli'][i]
    full_path = f"{args.things_dir}/{category}/{image_file}"
    
    img = Image.open(full_path).convert('RGB')
    img_tensor = preprocess(img)
    test_images.append(img_tensor)

# Process all test images at once
test_tensor = torch.stack(test_images).to(device)

with torch.no_grad():
    _, features = visual(test_tensor)
    
    batch_features = []
    for layer_name in layer_names:
        layer_features = features[layer_name]
        model_features = layer_features.flatten(1, 2)
        batch_features.append(model_features)
    
    fmaps_test = torch.cat(batch_features, dim=-1).detach().cpu().numpy()

# Standardize the image features
fmaps_test = scaler.transform(fmaps_test)

# Downsample the image features using PCA
fmaps_test = pca.transform(fmaps_test)
fmaps_test = fmaps_test.astype(np.float32)


# =============================================================================
# Train the encoding model
# =============================================================================
# Load neural data
neural_train_path = os.path.join(data_dir, f'fmri_{args.subject}_split-train.h5')
with h5py.File(neural_train_path, 'r') as f:
    neural_train = f['neural_data'][:]

# Fit the linear regression
reg = LinearRegression()
reg.fit(fmaps_train, neural_train)

# Store the linear regression weights
reg_param = {
    'coef_': reg.coef_,
    'intercept_': reg.intercept_,
    'n_features_in_': reg.n_features_in_
}

# Generate test predictions
neural_test_pred = reg.predict(fmaps_test)

# Save the in silico neural responses for the test images
save_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-fmri', 'train_dataset-things_fmri_1', 'vit_b_32')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'fmri_test_pred_{args.subject}.npy'
np.save(os.path.join(save_dir, file_name), neural_test_pred)


# =============================================================================
# Save the trained encoding models weights
# =============================================================================
weights = {
    'scaler_param': {
        'scale_': scaler.scale_,
        'mean_': scaler.mean_,
        'var_': scaler.var_,
        'n_features_in_': scaler.n_features_in_,
        'n_samples_seen_': scaler.n_samples_seen_
    },
    'pca_param': {
        'components_': pca.components_,
        'explained_variance_': pca.explained_variance_,
        'explained_variance_ratio_': pca.explained_variance_ratio_,
        'singular_values_': pca.singular_values_,
        'mean_': pca.mean_,
        'n_components_': pca.n_components_,
        'n_samples_': pca.n_samples_,
        'noise_variance_': pca.noise_variance_,
        'n_features_in_': pca.n_features_in_
    },
    'reg_param': reg_param
}

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri_1', 'model-vit_b_32', 'encoding_models_weights')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'weights_{args.subject}.npy'
np.save(os.path.join(save_dir, file_name), weights)