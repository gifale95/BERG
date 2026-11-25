"""Extract the DINOv2 features for the THINGS EEG2 images, and downsample them
using PCA.

Parameters
----------
model : str
    Name of the used encoding model.
things_eeg_2_dir : str
    Directory of the THINGS EEG2 dataset.
    https://osf.io/3jk45/
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from enum import Enum


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='dinov2')
parser.add_argument('--things_eeg_2_dir', default='/localscratch/giffordale95/things_eeg_2', type=str)
parser.add_argument('--berg_dir', default='/localscratch/giffordale95/brain-encoding-response-generator', type=str)
args, _ = parser.parse_known_args()

print('>>> Extract image features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the images
# =============================================================================
# Load the training images
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'training_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".JPEG"):
            image_list.append(os.path.join(root,file))
image_list.sort()
images_train = []
for file in tqdm(image_list):
    img = Image.open(file)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    images_train.append(img)
images_train = np.array(images_train)

# Load the testing images
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'test_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".JPEG"):
            image_list.append(os.path.join(root,file))
image_list.sort()
images_test = []
for file in tqdm(image_list):
    img = Image.open(file)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    images_test.append(img)
images_test = np.array(images_test)


# =============================================================================
# Load the DINOv2-Large model
# =============================================================================
# Model configuration
class DINOv2Config(Enum):
    """Expected (n_blocks, n_features) of each DINOv2 variant."""
    dinov2_vits14 = (12, 384)
    dinov2_vitb14 = (12, 768)
    dinov2_vitl14 = (24, 1024)
    dinov2_vitg14 = (40, 1536)

# Load the DINOv2 model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.to(device)
model.eval()

# Model size information
n_blocks = len(model.blocks)
n_features = model.embed_dim
patch_size = model.patch_size
# Only the even layers are used: layers 2, 4, ..., 24 (counting from 1), i.e.,
# the blocks with Python indices 1, 3, ..., 23
blocks_to_take = list(range(1, n_blocks, 2))
assert (n_blocks, n_features) == DINOv2Config['dinov2_vitl14'].value

# DINOv2 uses the ImageNet-1k statistics for input normalization
imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
    1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(
    1, 3, 1, 1)


# =============================================================================
# Activation extraction functions # !!!
# =============================================================================
def extract_dinov2l_activations(images, model, blocks_to_take, patch_size,
    n_features, imagenet_mean, imagenet_std, device, batch_size, feature_dtype,
    final_norm):
    """Extract the unpooled token activations of the given transformer blocks.
 
    Every block of a plain ViT preserves both the token count and the embedding
    dimensionality, so the flattened activations of all blocks have identical
    length and can be stacked along a single block dimension.
 
    Parameters
    ----------
    images : np.ndarray
        Images of shape (batch, height, width, channels), dtype uint8, RGB.
    model : torch.nn.Module
        DINOv2 model in eval mode.
    blocks_to_take : list of int
        Indices of the transformer blocks to extract, in ascending order.
    patch_size : int
        Side length in pixels of the ViT patches.
    n_features : int
        Embedding dimensionality of the model.
    imagenet_mean : torch.Tensor
        Per-channel normalization mean, of shape (1, 3, 1, 1).
    imagenet_std : torch.Tensor
        Per-channel normalization standard deviation, of shape (1, 3, 1, 1).
    device : torch.device
        Device on which the model runs.
    batch_size : int
        Number of images per forward pass.
    feature_dtype : torch.dtype
        Dtype of the stored activations.
    final_norm : bool
        Whether to apply the ViT's final LayerNorm to the block outputs.
 
    Returns
    -------
    block_features : torch.Tensor
        Activations of shape (n_images, n_tokens * n_features, n_blocks),
        stored on the device.
 
    """
 
    assert images.ndim == 4 and images.shape[3] == 3
    n_images = len(images)
    n_tokens = 1 + (images.shape[1] // patch_size) * (images.shape[2] //
        patch_size)
 
    # The activations are stored on the device, so they never occupy the RAM
    block_features = torch.zeros((n_images, n_tokens * n_features,
        len(blocks_to_take)), dtype=feature_dtype, device=device)
 
    for i in tqdm(range(0, n_images, batch_size), leave=False):
        images_batch = torch.from_numpy(np.ascontiguousarray(
            images[i:i+batch_size])).to(device)
        # (batch, height, width, channels) uint8 -> (batch, channels, height,
        # width) float32, scaled to [0, 1] and standardized
        images_batch = images_batch.permute(0, 3, 1, 2).float().div_(255)
        images_batch = (images_batch - imagenet_mean) / imagenet_std
 
        with torch.inference_mode():
            block_outputs = model.get_intermediate_layers(images_batch,
                n=blocks_to_take, reshape=False, return_class_token=True,
                norm=final_norm)
 
        # Get_intermediate_layers returns the patch and class tokens separately,
        # so the class token is concatenated back in front to recover the full
        # [CLS, patches] sequence. Casting before the stack keeps the transient
        # device memory at the target precision.
        tokens = [torch.cat((class_token.unsqueeze(1), patch_tokens),
            dim=1).flatten(start_dim=1).to(feature_dtype)
            for patch_tokens, class_token in block_outputs]
        # Stacking makes the write into block_features contiguous, instead of
        # one strided write per block.
        # Shape: (batch, n_tokens * n_features, n_blocks)
        tokens = torch.stack(tokens, dim=-1)
        block_features[i:i+len(images_batch)] = tokens
        del images_batch, block_outputs, tokens
 
    return block_features


# =============================================================================
# Extract the layerwise DINOv2 activations
# =============================================================================
# Model parameters. The activations are stored on the GPU as float16: in
# float32 the training features would take 209 GB (16540 x 3158016 x 4 bytes),
# more than the 180 GB of a B200, while float16 takes 104 GB, with ample
# precision for the PCA (which is computed in float32)
feature_dtype = torch.float16
batch_size = 64
final_norm = 1

# Extract the DNN activations for the training images
features_train = extract_dinov2l_activations(images_train, model,
    blocks_to_take, patch_size, n_features, imagenet_mean,
    imagenet_std, device, batch_size, feature_dtype, bool(final_norm))
features_train = features_train.reshape(len(features_train), -1)
# Shape per block: (n_images, n_tokens * n_features)
# Shape of dnn_features: (n_images, n_tokens * n_features, n_blocks)

# Extract the DNN activations for the testing images
features_test = extract_dinov2l_activations(images_test, model,
    blocks_to_take, patch_size, n_features, imagenet_mean,
    imagenet_std, device, batch_size, feature_dtype, bool(final_norm))
features_test = features_test.reshape(len(features_test), -1)
# Shape per block: (n_images, n_tokens * n_features)
# Shape of dnn_features: (n_images, n_tokens * n_features, n_blocks)


# =============================================================================
# Downsample the features with PCA
# =============================================================================
# The PCA transform z-scores the features with the training mean and SD, and
# multiplies them by a single weight matrix W, of shape (n_feat, n_pcs):
# pcs = ((features - mean) / sd) @ W. Since n_images << n_features, W is fitted
# from the eigendecomposition of the (n_train, n_train) Gram matrix of the
# z-scored training features Z, Z @ Z.T = U @ S**2 @ U.T, as
# W = Z.T @ U @ S**-1 (i.e., sklearn's components_.T). Everything runs on the
# GPU, where the features are stored, over chunks of feature columns, so no
# float32 or z-scored copy of the full features is ever created.
del images_train, images_test, model

# PCA parameters
n_pcs_fixed = 250
var_threshold = 0.95  # Fraction of variance explained by the retained PCs
chunk_size = 2 ** 15  # Feature columns processed at once on the GPU
n_train, n_feat = features_train.shape
n_test = len(features_test)

# Downsample the training features to (1) 250 PCs, or to (2) the number of PCs
# explaining 95% of variance. First, the z-scoring parameters are computed and
# the Gram matrix is accumulated, over chunks of feature columns.
mean = torch.zeros(n_feat, device=device)
sd = torch.zeros(n_feat, device=device)
gram = torch.zeros((n_train, n_train), device=device)
for j in tqdm(range(0, n_feat, chunk_size)):
    cols = slice(j, j+chunk_size)
    chunk_train = features_train[:, cols].float()
    sd[cols], mean[cols] = torch.std_mean(chunk_train, 0, correction=0)
    # Constant features are left unscaled, as in sklearn's StandardScaler
    sd[cols][sd[cols] == 0] = 1
    chunk_train = (chunk_train - mean[cols]) / sd[cols]
    gram.addmm_(chunk_train, chunk_train.T)
del chunk_train

# Eigendecomposition of the Gram matrix (in float64 for numerical precision),
# with the eigenvalues (S**2) sorted in descending order
eigval, eigvec = torch.linalg.eigh(gram.double())
del gram
eigval, eigvec = eigval.flip(0), eigvec.flip(1)

# Number of PCs explaining 95% of variance (selected as in sklearn's PCA)
explained_variance_ratio = (eigval / eigval.sum()).cpu().numpy()
n_pcs_var = int(np.searchsorted(np.cumsum(explained_variance_ratio),
    var_threshold, side='right')) + 1
n_pcs = max(n_pcs_fixed, n_pcs_var)

# PC scores of the training images: U @ S, which equals Z @ W
feat_train = (eigvec[:, :n_pcs] * eigval[:n_pcs].sqrt()).float().cpu().numpy()
feat_train_250pcs = feat_train[:, :n_pcs_fixed]
feat_train_95perc = feat_train[:, :n_pcs_var]
# Shape: (n_train, n_pcs)

# Use the PCA weights learned on the training features to downsample the
# testing features to (1) 250 PCs, or to (2) the number of PCs explaining 95%
# of variance. W can be very large (~12.6 GB every 1,000 PCs), so it is built
# over chunks of feature rows (W[cols] = Z[:, cols].T @ U @ S**-1), which are
# written straight to disk as two weight matrices: for 250 PCs, and for 95% of
# variance. Each chunk of z-scored testing features is multiplied with the
# matching chunk of W, and summing these products gives
# ((features_test - mean) / sd) @ W.
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-eeg',
    'train_dataset-things_eeg_2', 'model-'+args.model,
    'encoding_models_weights')
os.makedirs(save_dir, exist_ok=True)
pca_weights_250pcs = np.lib.format.open_memmap(os.path.join(save_dir,
    'pca_weights_250pcs.npy'), mode='w+', dtype=np.float32,
    shape=(n_feat, n_pcs_fixed))
pca_weights_95perc = np.lib.format.open_memmap(os.path.join(save_dir,
    'pca_weights_95perc.npy'), mode='w+', dtype=np.float32,
    shape=(n_feat, n_pcs_var))
# Shape: (n_feat, n_pcs)
eigvec_scaled = (eigvec[:, :n_pcs] / eigval[:n_pcs].sqrt()).float()
del eigvec
feat_test = torch.zeros((n_test, n_pcs), device=device)
for j in tqdm(range(0, n_feat, chunk_size)):
    cols = slice(j, j+chunk_size)
    chunk_train = (features_train[:, cols].float() - mean[cols]) / sd[cols]
    chunk_test = (features_test[:, cols].float() - mean[cols]) / sd[cols]
    weights_chunk = chunk_train.T @ eigvec_scaled
    feat_test.addmm_(chunk_test, weights_chunk)
    weights_chunk = weights_chunk.cpu().numpy()
    pca_weights_250pcs[cols] = weights_chunk[:, :n_pcs_fixed]
    pca_weights_95perc[cols] = weights_chunk[:, :n_pcs_var]
pca_weights_250pcs.flush()
pca_weights_95perc.flush()
del pca_weights_250pcs, pca_weights_95perc, weights_chunk, chunk_train, \
    chunk_test
feat_test = feat_test.cpu().numpy()
feat_test_250pcs = feat_test[:, :n_pcs_fixed]
feat_test_95perc = feat_test[:, :n_pcs_var]
# Shape: (n_test, n_pcs)

# Z-scoring parameters and PCA information
scaler_param = {'mean_': mean.cpu().numpy(), 'scale_': sd.cpu().numpy()}
pca_param = {
    'n_components_250pcs': n_pcs_fixed,
    'n_components_95perc': n_pcs_var,
    'explained_variance_ratio_': explained_variance_ratio,
    }


# =============================================================================
# Save
# =============================================================================
# Save the z-scoring parameters and PCA information. The PCA weight matrices
# were already saved while being built ('pca_weights_250pcs.npy' and
# 'pca_weights_95perc.npy'). Any new features are downsampled as:
# pcs = ((features - mean_) / scale_) @ pca_weights
weights = {
    'scaler_param': scaler_param,
    'pca_param': pca_param,
    }
file_name = 'weights.npy'
np.save(os.path.join(save_dir, file_name), weights)

# Save the train/test features
features = {
    'feat_train_250pcs': feat_train_250pcs,
    'feat_train_95perc': feat_train_95perc,
    'feat_test_250pcs': feat_test_250pcs,
    'feat_test_95perc': feat_test_95perc
}
np.save(os.path.join(save_dir, 'image_features.npy'), features)
