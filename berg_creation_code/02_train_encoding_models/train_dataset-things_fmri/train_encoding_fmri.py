"""Train a Ridge regression model to predict THINGS fMRI neural data using CLIP 
vision transformer feature maps as predictors. The Ridge regression is trained 
using the training images neural data (Y) and feature maps (X).

The feature maps come from a CLIP vision transformer, and are downsampled to 
250 principal components using PCA.

Pipeline steps:
1. Model setup: Load CLIP ViT-B/32 or ViT-B/32, wrap with torchextractor for multi-layer access
2. Feature extraction: Extract CLS tokens from 12 transformer layers (0-11)
   - Decision: Use CLS tokens only (not all 50 patch tokens) for computational efficiency
   - Batch processing (512 images) to handle training + test images
3. Preprocessing: StandardScaler normalization + PCA reduction to 250 components
   - Decision: PCA reduces 9,216 features (12 layers × 768 dims) for Ridge stability
4. Neural data loading: Chunked loading to prevent memory overflow during training
5. Model training: 32 separate Ridge models (~6,604 voxels each)
   - Decision: Split models for memory efficiency with 211,339 total voxels
   - Cross-validation over alphas [0.0001, 0.001, 0.01, 0.1, 1, 10]
6. Prediction: Load all 32 models, predict separately, concatenate results
7. Output: Neural predictions (100 test images × 211,339 voxels) + saved models

Parameters
----------
subject : str
    Which subject's data to use (e.g., 'sub-01').
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
only_cls : str
    If we should only use CLS token or all patches ('True' or 'False').
model : str
    Selecting which model to use ('vit_b_32' or 'clip.vit_b_32').
regression : str
    Select type of regression ('ridge' or 'linear').

Example usage:
python berg_creation_code/02_train_encoding_models/train_dataset-things_fmri/train_encoding_fmri.py \
    --subject sub-01 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --things_dir '/Volumes/Extreme SSD/Datasets/THINGS/things_images' \
    --only_cls True \
    --regression ridge \
    --n_pca_components 250 \
    --model clip.vit_b_32
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
from torchvision import transforms as trn
import torchvision
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression
import joblib


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
parser.add_argument('--only_cls', required=True, choices=["True", "False"],
                    help='If we should only use CLS token or all patches')
parser.add_argument('--model', required=True, choices=["vit_b_32", "clip.vit_b_32"],
                   help="Selecting which model to use")
parser.add_argument('--regression', required=True, choices=["ridge", "linear"],
                   help="Select type of regression")
parser.add_argument('--train_chunk_size', type=int, default=2000,
                   help='Number of trials per training chunk')
parser.add_argument('--feature_batch_size', type=int, default=512,
                   help='Batch size for feature extraction')
parser.add_argument('--n_pca_components', type=int, default=250,
                   help='Number of PCA components')
parser.add_argument('--cv_folds', type=int, default=5,
                   help='Cross-validation folds for Ridge alpha')
args = parser.parse_args()

args.only_cls = args.only_cls == "True"

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
print("Device:", device)

cls_suffix = 'cls' if args.only_cls else 'all'
print(f"Using {'CLS tokens only' if args.only_cls else 'all patches'}")


# =============================================================================
# Vision model
# =============================================================================
print("Load model...")

if args.model == "clip.vit_b_32":
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
    
    # Wrap the vision model with torchextractor
    visual = tx.Extractor(model.visual, layer_names)

elif args.model == "vit_b_32":
    model = torchvision.models.vit_b_32(weights='DEFAULT')
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
    
    preprocess = trn.Compose([
        trn.Lambda(lambda img: trn.CenterCrop(min(img.size))(img)),
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print("Model loaded")


# =============================================================================
# Extract the THINGS fMRI training and test image features
# =============================================================================
print("Extract the THINGS fMRI training and test image features...")

# Load metadata
data_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-things_fmri')
metadata_path = os.path.join(data_dir, f'fmri_{args.subject}_metadata.npz')
metadata = np.load(metadata_path, allow_pickle=True)

# Extract training image features
print("Extracting training features...")
n_train_images = len(metadata['train_stimuli'])
fmaps_train = []

for start_idx in tqdm(range(0, n_train_images, args.feature_batch_size), leave=False):
    end_idx = min(start_idx + args.feature_batch_size, n_train_images)
    batch_images = []
    
    # Load batch of training images
    for i in range(start_idx, end_idx):
        concept = metadata['train_concepts'][i]
        stimulus = metadata['train_stimuli'][i]
        full_path = os.path.join(args.things_dir, concept, stimulus)
        img = Image.open(full_path).convert('RGB')
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
            layer_features = features[layer_name]
            
            # Extract CLS token (first token in sequence) for all images in batch
            if args.only_cls:
                if args.model == "clip.vit_b_32":
                    # Shape: (seq_len=50, batch_size, hidden_dim=768)
                    model_features = layer_features[0, :, :]  # Shape: (batch_size, hidden_dim)
                elif args.model == "vit_b_32":
                    # Shape: (batch_size, seq_len=50, hidden_dim=768)
                    model_features = layer_features[:, 0, :]  # Shape: (batch_size, hidden_dim)
            else:
                # Flatten all tokens for each image: (batch_size, seq_len * hidden_dim)
                if args.model == "clip.vit_b_32":
                    # Shape: (seq_len=50, batch_size, hidden_dim=768)
                    # Flatten all tokens: (batch_size, seq_len * hidden_dim)
                    model_features = layer_features.permute(1, 0, 2).flatten(1, 2)
                elif args.model == "vit_b_32":
                    # Shape: (batch_size, seq_len=50, hidden_dim=768)
                    # Flatten: (batch_size, seq_len * hidden_dim)
                    model_features = layer_features.flatten(1, 2)
            
            batch_features.append(model_features)
        
        # Concatenate features from all layers
        ft = torch.cat(batch_features, dim=-1)  # Shape: (batch_size, 12*768)
        fmaps_train.append(ft.detach().cpu().numpy())

# Concatenate all training batches
fmaps_train = np.concatenate(fmaps_train, axis=0)
print(f"Training features shape (fmaps): {fmaps_train.shape}")

# Extract test image features
print("Extracting test features...")
test_images = []
for i in range(len(metadata['test_avg_stimuli'])):
    concept = metadata['test_avg_concepts'][i]
    stimulus = metadata['test_avg_stimuli'][i]
    full_path = os.path.join(args.things_dir, concept, stimulus)
    
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
        
        # Extract CLS token (first token in sequence) for all images in batch
        if args.only_cls:
            if args.model == "clip.vit_b_32":
                # Shape: (seq_len=50, batch_size, hidden_dim=768)
                model_features = layer_features[0, :, :]  # Shape: (batch_size, hidden_dim)
            elif args.model == "vit_b_32":
                # Shape: (batch_size, seq_len=50, hidden_dim=768)
                model_features = layer_features[:, 0, :]  # Shape: (batch_size, hidden_dim)
        else:
            # Flatten all tokens for each image: (batch_size, seq_len * hidden_dim)
            if args.model == "clip.vit_b_32":
                # Shape: (seq_len=50, batch_size, hidden_dim=768)
                # Flatten all tokens: (batch_size, seq_len * hidden_dim)
                model_features = layer_features.permute(1, 0, 2).flatten(1, 2)
            elif args.model == "vit_b_32":
                # Shape: (batch_size, seq_len=50, hidden_dim=768)
                # Flatten: (batch_size, seq_len * hidden_dim)
                model_features = layer_features.flatten(1, 2)
        
        batch_features.append(model_features)
    
    fmaps_test = torch.cat(batch_features, dim=-1).detach().cpu().numpy()

print(f"Test features shape: {fmaps_test.shape}")

# Standardize the image features (fit on training, transform both)
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)
fmaps_test = scaler.transform(fmaps_test)

# Downsample the image features using PCA (fit on training, transform both)
pca = PCA(n_components=args.n_pca_components, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_test = pca.transform(fmaps_test)

# Re-standardize after PCA
scaler_post_pca = StandardScaler()
scaler_post_pca.fit(fmaps_train)
fmaps_train = scaler_post_pca.transform(fmaps_train)
fmaps_test = scaler_post_pca.transform(fmaps_test)

print(f"Features after PCA - Train: {fmaps_train.shape}, Test: {fmaps_test.shape}")
# Verify standardization
print(f"After re-standardization: mean={fmaps_train.mean():.6f}, std={fmaps_train.std():.6f}")

print(f"\n=== PCA Diagnostics ===")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
print(f"First 10 components explain: {pca.explained_variance_ratio_[:10].sum():.4f}")
print(f"Variance per component (first 20): {pca.explained_variance_ratio_[:20]}")

# Convert to float32
fmaps_train = fmaps_train.astype(np.float32)
fmaps_test = fmaps_test.astype(np.float32)


# =============================================================================
# Train separate Ridge models for voxel chunks
# =============================================================================
print("Train the encoding models...")

# Define chunk parameters
n_chunks = 32
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# Load neural data shape info
neural_train_path = os.path.join(data_dir, f'fmri_{args.subject}_split-train.h5')
with h5py.File(neural_train_path, 'r') as f:
    n_trials, n_voxels = f['neural_data'].shape

print(f"Neural data shape: {n_trials} trials, {n_voxels} voxels")

# Calculate voxels per chunk
voxels_per_chunk = n_voxels // n_chunks
remainder = n_voxels % n_chunks

print(f"Training {n_chunks} separate models (~{voxels_per_chunk} voxels per chunk)")

# Train each chunk separately
for chunk_idx in range(n_chunks):
    # Calculate start and end voxels for this chunk
    start_voxel = chunk_idx * voxels_per_chunk + min(chunk_idx, remainder)
    if chunk_idx < remainder:
        chunk_size = voxels_per_chunk + 1
    else:
        chunk_size = voxels_per_chunk
    end_voxel = start_voxel + chunk_size
    
    print(f"\nTraining chunk {chunk_idx + 1}/{n_chunks}: voxels {start_voxel}-{end_voxel-1} ({chunk_size} voxels)")
    
    # Load neural data for this chunk only
    neural_chunk = np.empty((n_trials, chunk_size), dtype=np.float32)
    
    with h5py.File(neural_train_path, 'r') as f:
        # Load in batches to avoid memory issues
        for batch_start in range(0, n_trials, args.train_chunk_size):
            batch_end = min(batch_start + args.train_chunk_size, n_trials)
            
            # Load batch and slice voxels
            batch_data = f['neural_data'][batch_start:batch_end, start_voxel:end_voxel]
            neural_chunk[batch_start:batch_end] = batch_data
    
    print(f"Chunk data shape: {neural_chunk.shape}")
    print(f"  Neural chunk stats: mean={neural_chunk.mean():.6f}, std={neural_chunk.std():.6f}")
    print(f"  Neural chunk range: [{neural_chunk.min():.6f}, {neural_chunk.max():.6f}]")
    print(f"  % of values exactly 0.0: {(neural_chunk == 0).sum() / neural_chunk.size * 100:.2f}%")
    
    # Train model based on regression type
    print(f"  Feature stats going into model: mean={fmaps_train.mean():.6f}, std={fmaps_train.std():.6f}")
    print(f"  Feature range: [{fmaps_train.min():.6f}, {fmaps_train.max():.6f}]")
    
    if args.regression == 'ridge':
        chunk_reg = RidgeCV(alphas=alphas, cv=args.cv_folds, scoring='r2')
        chunk_reg.fit(fmaps_train, neural_chunk)
        print(f"  Training R² score: {chunk_reg.score(fmaps_train, neural_chunk):.4f}")
        
        print(f"Chunk {chunk_idx + 1} completed. Best alpha: {chunk_reg.alpha_}")
        
        # Try to access CV values if available
        if hasattr(chunk_reg, 'cv_values_'):
            cv_values = chunk_reg.cv_values_
            cv_results = cv_values.mean(axis=0)
            cv_std = cv_values.std(axis=0)
            
            print(f"  Mean CV scores per alpha:")
            for i, alpha in enumerate(alphas):
                best_marker = " <-- BEST" if alpha == chunk_reg.alpha_ else ""
                print(f"    Alpha={alpha:7.4f}: {cv_results[i]:7.4f} ± {cv_std[i]:.4f}{best_marker}")
        else:
            print(f"  CV values not stored (sklearn version doesn't support it)")
        
        # Check coefficient statistics
        print(f"  Coefficient stats: mean={chunk_reg.coef_.mean():.6f}, std={chunk_reg.coef_.std():.6f}, max_abs={np.abs(chunk_reg.coef_).max():.6f}")

    else:  # linear
        chunk_reg = LinearRegression()
        chunk_reg.fit(fmaps_train, neural_chunk)
        print(f"Chunk {chunk_idx + 1} completed.")
    
    # Save individual model
    model_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
        'train_dataset-things_fmri', f'model-{args.model}', 'encoding_model_weights')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    model_filename = f'{args.regression}_{cls_suffix}_chunk_{chunk_idx}_{args.subject}.pkl'
    joblib.dump(chunk_reg, os.path.join(model_dir, model_filename))

print("\nAll chunk models trained and saved!")


# =============================================================================
# Test encoding models on averaged test images
# =============================================================================
print("\nPredicting test responses...")

# Load all models and predict
chunk_predictions = []
model_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri', f'model-{args.model}', 'encoding_model_weights')

for chunk_idx in range(n_chunks):
    model_filename = f'{args.regression}_{cls_suffix}_chunk_{chunk_idx}_{args.subject}.pkl'
    chunk_model = joblib.load(os.path.join(model_dir, model_filename))
    
    # Predict for this chunk
    chunk_pred = chunk_model.predict(fmaps_test)
    chunk_predictions.append(chunk_pred)

# Concatenate along voxel dimension
test_predictions = np.concatenate(chunk_predictions, axis=1)

print(f"Test predictions shape: {test_predictions.shape}")

print(f"\n=== Test Prediction Diagnostics ===")
print(f"Prediction stats: mean={test_predictions.mean():.6f}, std={test_predictions.std():.6f}")
print(f"Prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")

assert test_predictions.shape == (100, n_voxels), \
    f"Expected shape (100, {n_voxels}), got {test_predictions.shape}"

# Save test predictions
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-fmri', 'train_dataset-things_fmri', args.model)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

test_file_name = f'fmri_test_pred_{args.regression}_{cls_suffix}_{args.subject}.npy'
np.save(os.path.join(results_dir, test_file_name), test_predictions)

print(f"Test predictions saved: {test_predictions.shape}")


# =============================================================================
# Save preprocessing parameters
# =============================================================================
print("\nSave preprocessing parameters...")

preprocessing_weights = {
    'scaler_pre_pca': {
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
    'scaler_post_pca': { 
        'scale_': scaler_post_pca.scale_,
        'mean_': scaler_post_pca.mean_,
        'var_': scaler_post_pca.var_,
        'n_features_in_': scaler_post_pca.n_features_in_,
        'n_samples_seen_': scaler_post_pca.n_samples_seen_
    },
    'model_info': {
        'model_type': f'chunked_{args.regression}',
        'only_cls': args.only_cls,
        'n_chunks': n_chunks,
        'voxels_per_chunk': voxels_per_chunk,
        'n_voxels': n_voxels,
        'subject_id': args.subject
    }
}

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri', f'model-{args.model}',
    'encoding_models_weights')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

file_name = f'preprocessing_{args.regression}_{cls_suffix}_{args.subject}.npy'
np.save(os.path.join(save_dir, file_name), preprocessing_weights)

print("Model training completed successfully!")