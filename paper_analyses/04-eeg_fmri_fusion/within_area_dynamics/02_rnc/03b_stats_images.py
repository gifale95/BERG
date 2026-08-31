"""Compute the complexity of the controlling images, and test whether the
images driving the early t-fMRI time window of a ROI while suppressing
the late time window of the same ROI have lower complexity compared to
the suppressing the early time window while driving the late time window.

Parameters
----------
roi: str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest.
imageset : str
    The image set to use for the analysis. Possible values are: 'imagenet'
    (ILSVRC-2012 validation split) and 'coco' (MS COCO 2017 test split).
n_images: int
    Number of retained controlling or baseline images.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
import torch
from enum import Enum
import torchvision
from torchvision import transforms as trn
from scipy.stats import ttest_ind
from tqdm import tqdm
import io
from PIL import Image
from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet', type=str)
parser.add_argument('--n_images', default=25, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Break down the time windows
# =============================================================================
time_window_1_start, time_window_1_end = map(
    float, args.time_window_pair.split('__')[0].split('-'))
time_window_2_start, time_window_2_end = map(
    float, args.time_window_pair.split('__')[1].split('-'))


# =============================================================================
# Load the RNC controlling image IDs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', 'cv-0',
    args.time_window_pair, f'imageset-{args.imageset}',
    f'stats_roi-{args.roi}.npy')
data = np.load(data_dir, allow_pickle=True).item()

controlling_images = data['controlling_images']


# =============================================================================
# Load the images
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224))
])

# Access the ILSVRC-2012 validation split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val',
    transform=transform)

# Load the controlling images
images = []
for key in tqdm(['high_1_low_2', 'low_1_high_2']):
    val = controlling_images[key]
    for i, img_id in enumerate(val):
        img, _ = imageset.__getitem__(img_id)
        images.append(img)
images = np.array(images)


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
blocks_to_take = list(range(n_blocks))
assert (n_blocks, n_features) == DINOv2Config['dinov2_vitl14'].value

# DINOv2 uses the ImageNet-1k statistics for input normalization
imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
    1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(
    1, 3, 1, 1)


# =============================================================================
# Activation extraction functions
# =============================================================================
def extract_dinov2l_activations(images, model, blocks_to_take, patch_size,
    n_features, imagenet_mean, imagenet_std, device, batch_size, feature_dtype,
    final_norm):
    """Extract the unpooled token activations of all transformer blocks.
 
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
    feature_dtype : np.dtype
        Dtype of the stored activations.
    final_norm : bool
        Whether to apply the ViT's final LayerNorm to the block outputs.
 
    Returns
    -------
    block_features : np.ndarray
        Activations of shape (n_images, n_tokens, n_features, n_blocks).
 
    """
 
    assert images.ndim == 4 and images.shape[3] == 3
    n_images = len(images)
    n_tokens = 1 + (images.shape[1] // patch_size) * (images.shape[2] //
        patch_size)
 
    torch_dtype = getattr(torch, feature_dtype.name)
    block_features = np.zeros((n_images, n_tokens, n_features,
        len(blocks_to_take)), dtype=feature_dtype)
 
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
            dim=1).to(torch_dtype)
            for patch_tokens, class_token in block_outputs]
        # Stacking on the device makes the host transfer and the write into
        # block_features contiguous, instead of one strided write per block.
        # Shape: (batch, n_tokens, n_features, n_blocks)
        tokens = torch.stack(tokens, dim=-1)
        block_features[i:i+len(images_batch)] = tokens.cpu().numpy()
        del images_batch, block_outputs, tokens
 
    return block_features


# =============================================================================
# Extract the layerwise DNN activations
# =============================================================================
# Model parameters
feature_dtype = np.dtype('float32')
batch_size = 64
final_norm = True

# Extract the DNN activations
dnn_feats = extract_dinov2l_activations(images, model,
    blocks_to_take, patch_size, n_features, imagenet_mean,
    imagenet_std, device, batch_size, feature_dtype, bool(final_norm))
# Shape per block: (n_images, n_tokens, n_features, n_blocks)
dnn_features = {}
dnn_features['high_1_low_2'] = dnn_feats[:args.n_images]
dnn_features['low_1_high_2'] = dnn_feats[args.n_images:]


# =============================================================================
# Compute image complexity
# =============================================================================
# Compute the DNN layerwise univariate responses, by averaging the activations
# across all units of a given layer; this gives a single response per image
# per layer, which is used to estimate image complexity
for key, val in dnn_features.items():
    dnn_features[key] = np.sqrt(np.mean(val**2, (1, 2))) # Shape: (n_images, n_layers)

# For each image, compute the complexity as the layer number yielding highest
# response; this gives a single complexity value per image # !!! THIS HAS TO CHANGE!!!!!
img_complexity = {}
for key, val in dnn_features.items():
    img_complexity[key] = np.argmax(val, 1) # Shape: (n_images,)


# =============================================================================
# Test for significant differences in image complexity between the two sets of
# controlling images
# =============================================================================
p_val = ttest_ind(img_complexity['high_1_low_2'],
    img_complexity['low_1_high_2'], alternative='less').pvalue


# =============================================================================
# Save the results
# =============================================================================
stats = {
    'dnn_features': dnn_features,
    'img_complexity': img_complexity,
    'p_val': p_val
    }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', 'cv-0', args.time_window_pair,
    f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_images_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), stats)