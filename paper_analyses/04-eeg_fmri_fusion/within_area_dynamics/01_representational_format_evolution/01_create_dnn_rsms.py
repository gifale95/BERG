"""Create RSMs using layerwise DNN activations for the 200 THINGS EEG2 test
images.

Parameters
----------
dnn : str
    Name of the used DNN. Possible values are 'dinov2l'.
images : str
    If 'things_eeg_2', use the 200 THINGS EEG2 test images.
    If 'nsd_515_shared', use the 515 NSD shared images.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/

"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
from PIL import Image
import h5py
import torch
from enum import Enum

parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--images', default='things_eeg_2', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Create DNN RSMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Load the images
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# 200 THINGS EEG2 test images
if args.images == 'things_eeg_2':
    # Load the metadata
    metadata_eeg = berg.get_model_metadata(
        'eeg-things_eeg_2-vit_b_32',
        subject=1
        )
    # Get the test image file names
    test_img_files = metadata_eeg['encoding_models']['test_img_info']\
        ['test_img_files']
    # Loop across test image files
    images = []
    for file in tqdm(test_img_files):
        # Find correct subfolder
        img_path = None
        for root, _, files in os.walk(os.path.join(args.things_dir)):
            if file in files:
                img_path = os.path.join(root, file)
                break
        # Load and resize the image
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
        images.append(img)
    # Format the images to a numpy array
    images = np.array(images)

# 515 NSD shared images
elif args.images == 'nsd_515_shared':
    # Initialize BERG
    berg = BERG(berg_dir=args.berg_dir)
    # Get the test image number
    metadata = berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=1
    )
    test_img_num = metadata['encoding_models']['test_img_num']
    # Load the test images
    sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
        'nsd', 'nsd_stimuli.hdf5'), 'r')
    sdataset = sf.get('imgBrick')
    images = sdataset[test_img_num]
    # Resize the images
    images = np.stack([np.array(Image.fromarray(img).resize((224, 224),
        Image.Resampling.LANCZOS)) for img in images])


# =============================================================================
# Load the DINOv2-Large model
# =============================================================================
if args.dnn == 'dinov2l':

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
        Activations of shape (n_images, n_tokens * n_features, n_blocks).
 
    """
 
    assert images.ndim == 4 and images.shape[3] == 3
    n_images = len(images)
    n_tokens = 1 + (images.shape[1] // patch_size) * (images.shape[2] //
        patch_size)
 
    torch_dtype = getattr(torch, feature_dtype.name)
    block_features = np.zeros((n_images, n_tokens * n_features,
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
            dim=1).flatten(start_dim=1).to(torch_dtype)
            for patch_tokens, class_token in block_outputs]
        # Stacking on the device makes the host transfer and the write into
        # block_features contiguous, instead of one strided write per block.
        # Shape: (batch, n_tokens * n_features, n_blocks)
        tokens = torch.stack(tokens, dim=-1)
        block_features[i:i+len(images_batch)] = tokens.cpu().numpy()
        del images_batch, block_outputs, tokens
 
    return block_features


# =============================================================================
# Extract the layerwise DNN activations
# =============================================================================
if args.dnn == 'dinov2l':

    # Model parameters
    feature_dtype = np.dtype('float32')
    batch_size = 64
    final_norm = 1

    # Extract the DNN activations
    dnn_features = extract_dinov2l_activations(images, model,
        blocks_to_take, patch_size, n_features, imagenet_mean,
        imagenet_std, device, batch_size, feature_dtype, bool(final_norm))
    # Shape per block: (n_images, n_tokens * n_features)


# =============================================================================
# Create RSMs using the layerwise DNN activations
# =============================================================================
Z = np.ascontiguousarray(dnn_features.transpose(2, 0, 1), dtype=np.float32)  # (Layers, Images, Features)
Z -= Z.mean(-1, keepdims=True)
Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
dnn_rsms = (Z @ Z.transpose(0, 2, 1)).transpose(1, 2, 0)    


# =============================================================================
# Save the DNN RSMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution', 'dnn_rsms')
os.makedirs(save_dir, exist_ok=True)

file_name = f'dnn_rsms_dnn-{args.dnn}_images-{args.images}.npy'

np.save(os.path.join(save_dir, file_name), dnn_rsms)