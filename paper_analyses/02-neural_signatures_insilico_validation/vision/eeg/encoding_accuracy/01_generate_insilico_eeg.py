"""Generate the in silico EEG responses for the 200 THINGS EEG2 test images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
from PIL import Image
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico EEG <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'encoding_accuracy', 'insilico_eeg_responses',
    args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Get the THINGS EEG2 test images
# =============================================================================
# Get the image metadata from BERG 
berg = BERG(berg_dir=args.berg_dir)
metadata_things = berg.get_model_metadata(
    args.encoding_model,
    subject=1
    )
test_img_files = metadata_things['encoding_models']['test_img_info']\
    ['test_img_files']
test_img_concepts_THINGS = metadata_things['encoding_models']['test_img_info']\
    ['test_img_concepts_THINGS']

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
# Loop across EEG subjects
# =============================================================================
for s, sub in enumerate(tqdm(args.subjects)):


# =============================================================================
# Load BERG's encoding model
# =============================================================================
    model = berg.get_encoding_model(
        args.encoding_model,
        subject=sub
        )


# =============================================================================
# Generate the in silico EEG responses
# =============================================================================
    eeg, metadata = berg.encode(model, images, return_metadata=True)
    eeg = eeg.astype(np.float32)


# =============================================================================
# Save the in silico EEG responses
# =============================================================================
    data = {
        'eeg': eeg,
        'metadata': metadata
    }
    file_name = 'insilico_eeg_responses_sub-' + format(sub, '02') + '.npy'
    np.save(os.path.join(save_dir, file_name), data)

    # Delete unused variables
    del eeg, data, metadata
    torch.cuda.empty_cache()
    gc.collect()