"""Generate in silico fMRI resposnes for the THINGS EEG2 train and test images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses.
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
import h5py
from PIL import Image
from torchvision import transforms as trn
from tqdm import tqdm
from berg import BERG
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.fmri_subject
    )


# =============================================================================
# Load the THINGS EEG2 metadata
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the metadata
metadata = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
    )


# =============================================================================
# Generate and save the in silico fMRI responses
# =============================================================================
for split in ['test', 'train']:

    # Extract the THINGS EEG2 train and test image filenames
    image_filenames = metadata['encoding_models'][f'{split}_img_info']\
        [f'{split}_img_files']

    # Empty in silico fMRI response lists
    lh = []
    rh = []

    for file in tqdm(image_filenames, desc=f'In silico fMRI {split}'):
        
        # Find correct subfolder
        img_path = None
        for root, _, files in os.walk(os.path.join(args.things_dir)):
            if file in files:
                img_path = os.path.join(root, file)
                break
        
        # Load and transform image
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
        img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
        img = np.expand_dims(img, 0) # Add the batch dimension

        # Generate the in silico fMRI responses
        insilico_fmri = berg.encode(model, img, return_metadata=False)

        # Store the in silico fMRI responses averaged across image exemplars
        lh.append(np.squeeze(insilico_fmri[0]))
        rh.append(np.squeeze(insilico_fmri[1]))

        # Delete unused variables
        del insilico_fmri
        torch.cuda.empty_cache()
        gc.collect()

    # Convert the in silico fMRI responses to numpy arrays
    lh = np.array(lh).astype(np.float32)
    rh = np.array(rh).astype(np.float32)

    # Save the in silico fMRI responses
    file_name_lh = f'things_eeg_2_{split}_sub-{args.fmri_subject:02d}_lh'
    file_name_rh = f'things_eeg_2_{split}_sub-{args.fmri_subject:02d}_rh'
    with h5py.File(os.path.join(save_dir, file_name_lh), 'w') as f:
        f.create_dataset('insilico_fmri', data=lh, dtype=np.float32)
    with h5py.File(os.path.join(save_dir, file_name_rh), 'w') as f:
        f.create_dataset('insilico_fmri', data=rh, dtype=np.float32)