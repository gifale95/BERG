"""Train a regression model to predict TVSD neural data using vision
transformer feature maps as predictors. The regression is trained  using the
training images neural data (Y) and feature maps (X).

The feature maps come from a vision transformer, and are downsampled to 
250 principal components using PCA.


Pipeline steps:
1. Model setup: Load ViT-B/32, wrap with torchextractor for multi-layer access
2. Feature extraction: Extract tokens from 12 transformer layers (0-11)
   - Batch processing (512 images) to handle 22,248 training + 100 test images
3. Preprocessing: StandardScaler normalization + PCA reduction to 250 components
4. Neural data loading: Load all training data and reshape
5. Model training: Single Linear model (1,024 electrodes × 300 timepoints)
6. Prediction: Generate test predictions
7. Output: Neural predictions (1,024 electrodes × 300 timepoints) + saved models

Parameters
----------
monkey : str
    Which monkey's data to use ('monkeyN' or 'monkeyF').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
things_dir : str
    Directory of the THINGS images.
train_chunk_size : int
    Number of trials per chunk for loading neural data (default: 2000).
feature_batch_size : int
    Batch size for feature extraction (default: 512).
n_pca_components : int
    Number of PCA components (default: 250).
train_split : str
    Which training split to use (default: 'all_training_splits').
"""

import argparse
import hashlib
import torch
import numpy as np
import os
import h5py
from tqdm import tqdm
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
parser.add_argument("--monkey", required=True, choices=["monkeyN", "monkeyF"],
                   help="Select which monkey's data to use.")
parser.add_argument('--berg_dir', required=True, type=str,
                   help="Directory of the BERG framework.")
parser.add_argument('--things_dir', required=True, type=str,
                   help="Directory of the things images.")
parser.add_argument('--train_chunk_size', type=int, default=2000,
                   help='Number of trials per chunk for loading neural data')
parser.add_argument('--feature_batch_size', type=int, default=512,
                   help='Batch size for feature extraction')
parser.add_argument('--n_pca_components', type=int, default=250,
                   help='Number of PCA components')
parser.add_argument('--train_split', type=str, default='all_training_splits',
                   choices=['all_training_splits', 'single_training_split_1', 'single_training_split_2', 'single_training_split_3', 'single_training_split_4'],
                   help='Which training split to use')
args = parser.parse_args()


print('>>> Train TVSD encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
torch.manual_seed(seed)

# Derive a unique PCA random seed for each monkey × training split combination,
# so that each encoding model starts from a different PCA basis
pca_seed_key = f"{args.monkey}_{args.train_split}"
pca_seed = int(hashlib.md5(pca_seed_key.encode()).hexdigest(), 16) % (2**31)
print(f'\nPCA seed:        {pca_seed} (from "{pca_seed_key}")')

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
data_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-tvsd')
metadata_path = os.path.join(data_dir, f'tvsd_{args.monkey}_metadata.npy')
metadata = np.load(metadata_path, allow_pickle=True).item()


# =============================================================================
# Extract the TVSD training image features
# =============================================================================
train_img_ids = metadata['encoding_model'][args.train_split]['train_img_ids']
train_concepts = metadata['encoding_model'][args.train_split]['train_concepts']
train_stimuli = metadata['encoding_model'][args.train_split]['train_stimuli']

n_train_images = len(train_img_ids)
fmaps_train = []

for start_idx in tqdm(range(0, n_train_images, args.feature_batch_size), leave=False):
    end_idx = min(start_idx + args.feature_batch_size, n_train_images)
    batch_images = []
    
    # Load batch of training images
    for i in range(start_idx, end_idx):
        img_path = train_concepts[i] + "/" + train_stimuli[i]
        full_path = os.path.join(args.things_dir, img_path)
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
pca = PCA(n_components=args.n_pca_components, random_state=pca_seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_train = fmaps_train.astype(np.float32)


# =============================================================================
# Extract the TVSD test image features
# =============================================================================
# Get unique test images (100 unique images from the 3000 test trials)
test_img_ids = metadata['encoding_model']['test_img_ids']
unique_test_ids = np.unique(test_img_ids)

test_images = []
for test_id in unique_test_ids:
    # Find first occurrence of this test image
    test_idx = np.where(test_img_ids == test_id)[0][0]
    
    category = metadata['encoding_model']['test_concepts'][test_idx]
    image_file = metadata['encoding_model']['test_stimuli'][test_idx]
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
# Train the encoding models
# =============================================================================
# Load neural data shape info
if args.train_split == 'all_training_splits':
    neural_train_path = os.path.join(data_dir, f'tvsd_{args.monkey}_all_training_splits.h5')
else:
    neural_train_path = os.path.join(data_dir, f'tvsd_{args.monkey}_{args.train_split}.h5')

with h5py.File(neural_train_path, 'r') as f:
    n_trials, n_electrodes, n_times = f['neural_data'].shape

# Load full neural data in chunks to prevent memory overflow
neural_data = np.empty((n_trials, n_electrodes * n_times), dtype=np.float32)

print(f"Loading neural data in chunks of {args.train_chunk_size} trials...")
with h5py.File(neural_train_path, 'r') as f:
    for batch_start in tqdm(range(0, n_trials, args.train_chunk_size), desc="Loading neural data"):
        batch_end = min(batch_start + args.train_chunk_size, n_trials)
        
        # Load batch
        batch_data = f['neural_data'][batch_start:batch_end, :, :]
        # Reshape to (batch_size, n_electrodes * n_times)
        batch_reshaped = batch_data.reshape(batch_data.shape[0], -1)
        neural_data[batch_start:batch_end] = batch_reshaped

# Fit the linear regression
print("Fitting linear regression model...")
reg = LinearRegression()
reg.fit(fmaps_train, neural_data)

# Store the linear regression weights
reg_param = {
    'coef_': reg.coef_,
    'intercept_': reg.intercept_,
    'n_features_in_': reg.n_features_in_
}

# Use the learned weights to generate in silico neural responses for the test images
neural_test_pred = reg.predict(fmaps_test)
neural_test_pred = neural_test_pred.reshape(-1, n_electrodes, n_times).astype(np.float32)

# Save the in silico neural responses for the test images
save_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-utah_array', 'train_dataset-tvsd', 'vit_b_32')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'utah_array_test_pred_{args.monkey}_{args.train_split}.npy'
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

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-utah_array',
    'train_dataset-tvsd', 'model-vit_b_32', 'encoding_models_weights')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'weights_{args.monkey}_{args.train_split}.npy'
np.save(os.path.join(save_dir, file_name), weights)