"""Generate the in silico fMRI responses for the 200 THINGS EEG2 test images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list
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
from PIL import Image
from tqdm import tqdm
from berg import BERG
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
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
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'behavioral_modeling',
    'insilico_fmri_responses', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the THINGS EEG2 test images
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the metadata
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b-32',
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
    
    # Load and transform the image
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
    images.append(img)

# Format the images to a numpy array
images = np.array(images)


# =============================================================================
# Generate the in silico fMRI responses
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across subjects
for sub in args.subjects:

    # Load the encoding model
    model = berg.get_encoding_model(
        args.encoding_model,
        subject=sub
        )

    # Load the metadata
    metadata = berg.get_model_metadata(
        args.encoding_model,
        subject=sub
        )

    # Generate the in silico fMRI responses
    fmri_lh, fmri_rh = berg.encode(model, images)


# =============================================================================
# Save the in silico fMRI responses
# =============================================================================
    data_lh = {
        'fmri': fmri_lh,
        'metadata': metadata
    }
    data_rh = {
        'fmri': fmri_rh,
        'metadata': metadata
    }
    file_name_lh = 'insilico_fmri_responses_sub-' + format(sub, '02') + \
        '_lh.npy'
    file_name_rh = 'insilico_fmri_responses_sub-' + format(sub, '02') + \
        '_rh.npy'
    np.save(os.path.join(save_dir, file_name_lh), data_lh)
    np.save(os.path.join(save_dir, file_name_rh), data_rh)

    # Delete unused variables
    del fmri_lh, fmri_rh, data_lh, data_rh, metadata
    torch.cuda.empty_cache()
    gc.collect()