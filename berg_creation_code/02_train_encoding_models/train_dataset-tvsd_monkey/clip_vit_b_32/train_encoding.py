"""Train a Ridge regression model to predict TVSD neural data using CLIP 
vision transformer feature maps as predictors. The Ridge regression is trained 
using the training images neural data (Y) and feature maps (X).

The feature maps come from a CLIP vision transformer, and are downsampled to 
250 principal components using PCA.

Parameters
----------
monkey : str
    Which monkey's data to use ('monkeyN' or 'monkeyF').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
things_dir : str
    Directory of the THINGS images.
train_chunk_size : int
    Number of trials per training chunk (default: 2000).
feature_batch_size : int
    Batch size for feature extraction (default: 512).
n_pca_components : int
    Number of PCA components (default: 250).
cv_folds : int
    Cross-validation folds for Ridge alpha (default: 5).

python berg_creation_code/02_train_encoding_models/train_dataset-tvsd_monkey/clip_vit_b_32/train_encoding.py \
    --monkey monkeyF \
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
import clip
import torchextractor as tx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV


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
                   help='Number of trials per training chunk')
parser.add_argument('--feature_batch_size', type=int, default=512,
                   help='Batch size for feature extraction')
parser.add_argument('--n_pca_components', type=int, default=250,
                   help='Number of PCA components')
parser.add_argument('--cv_folds', type=int, default=5,
                   help='Cross-validation folds for Ridge alpha')
args = parser.parse_args()

print('>>> Train TVSD encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
torch.manual_seed(seed)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


# =============================================================================
# Helper function to map trial to image
# =============================================================================
def map_trial_to_image(trial_idx, metadata, things_dir=None):
	"""Map a training trial index to its corresponding THINGS image information."""
	if trial_idx < 0 or trial_idx >= len(metadata['train_img_files']):
		raise ValueError(f"Trial index {trial_idx} out of range")
	
	image_info = {
		'trial_idx': trial_idx,
		'stimulus_id': metadata['train_img_ids'][trial_idx],
		'image_file': metadata['train_img_files'][trial_idx],
		'object_category': metadata['train_img_concepts'][trial_idx],
		'recording_day': metadata['train_days'][trial_idx],
		'sequence_position': metadata['train_sequence_pos'][trial_idx]
	}
	
	if things_dir:
		category = metadata['train_img_concepts'][trial_idx]
		image_info['full_path'] = f"{things_dir}/{category}/{image_info['image_file']}"
	
	return image_info


# =============================================================================
# Vision model
# =============================================================================
# Load the model
print("Load model...")
model, preprocess = clip.load("ViT-B/32", device=device)

if device == 'cuda':
    model = model.float()

model.eval()

layer_names = [
    "transformer.resblocks.0",
    "transformer.resblocks.1",
    "transformer.resblocks.2",
    "transformer.resblocks.3",
    "transformer.resblocks.4",
    "transformer.resblocks.5",
    "transformer.resblocks.6",
    "transformer.resblocks.7",
    "transformer.resblocks.8",
    "transformer.resblocks.9",
    "transformer.resblocks.10",
    "transformer.resblocks.11",
]

print("Model loaded")


# =============================================================================
# Extract the TVSD training image features
# =============================================================================
print("Extract the TVSD training image features...")

# Wrap the vision model with torchextractor
visual = tx.Extractor(model.visual, layer_names)

# Load metadata
data_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-tvsd_monkey')
metadata_path = os.path.join(data_dir, f'tvsd_{args.monkey}_metadata.npz')
metadata = np.load(metadata_path)

n_train_images = len(metadata['train_img_files'])
fmaps_train = []

for start_idx in tqdm(range(0, n_train_images, args.feature_batch_size), leave=False):
	end_idx = min(start_idx + args.feature_batch_size, n_train_images)
	batch_images = []
	
	# Load batch of images
	for i in range(start_idx, end_idx):
		trial_info = map_trial_to_image(i, metadata, args.things_dir)
		img = Image.open(trial_info['full_path']).convert('RGB')
		img_tensor = preprocess(img)
		batch_images.append(img_tensor)
	
	# Process batch
	batch_tensor = torch.stack(batch_images).to(device)
	
	with torch.no_grad():
		# Extract features using torchextractor
		_, features = visual(batch_tensor)
		
		# Extract CLS token from each layer and concatenate
		batch_features = []
		for layer_name in layer_names:
			layer_features = features[layer_name][:, 0, :]  # CLS token only
			batch_features.append(layer_features)
		
		# Concatenate features from all layers
		ft = torch.cat(batch_features, dim=-1)
		fmaps_train.append(ft.detach().cpu().numpy())

# Concatenate all batches
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
# Train the encoding models
# =============================================================================
print("Train the encoding models...")
# Load the training neural responses
neural_train_path = os.path.join(data_dir, f'tvsd_{args.monkey}_split-train_normalized.h5')

with h5py.File(neural_train_path, 'r') as f:
	neural_data_shape = f['neural_data_normalized'].shape

n_trials, n_times, n_electrodes = neural_data_shape

# Define alpha values for cross-validation
alphas = [0.1, 1, 10, 100, 1000]

# Load neural data in chunks and reshape for training
n_chunks = int(np.ceil(n_trials / args.train_chunk_size))
neural_train_list = []

with h5py.File(neural_train_path, 'r') as f:
	for chunk_idx in tqdm(range(n_chunks), leave=False):
		start_idx = chunk_idx * args.train_chunk_size
		end_idx = min(start_idx + args.train_chunk_size, n_trials)
		
		chunk_data = f['neural_data_normalized'][start_idx:end_idx]
		# Reshape to (trials, features) where features = times * electrodes
		chunk_reshaped = chunk_data.reshape(chunk_data.shape[0], -1)
		neural_train_list.append(chunk_reshaped)

# Concatenate all chunks
neural_train = np.concatenate(neural_train_list, axis=0)

# Fit the Ridge regression
reg = RidgeCV(alphas=alphas, cv=args.cv_folds, scoring='r2')
reg.fit(fmaps_train, neural_train)

# Store the linear regression weights
reg_param = {
	'coef_': reg.coef_,
	'intercept_': reg.intercept_,
	'alpha_': reg.alpha_,
	'cv_values_': reg.cv_values_,
	'n_features_in_': reg.n_features_in_
}


# =============================================================================
# Save the trained encoding models weights
# =============================================================================
print("Save the trained encoding models weights...")
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
	'reg_param': reg_param,
	'monkey_id': args.monkey
	}

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
	'train_dataset-tvsd_monkey', 'model-clip_vit_b_32',
	'encoding_models_weights')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = f'weights_{args.monkey}.npy'

np.save(os.path.join(save_dir, file_name), weights)