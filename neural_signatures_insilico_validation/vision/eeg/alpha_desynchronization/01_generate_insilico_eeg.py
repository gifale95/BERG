"""Generate in silico EEG responses for the 200 test images from THINGS EEG2.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of the subject identifiers for the EEG encoding models. Since the
    used encoding models are trained on THINGS EEG2 data, valid subject
    identifiers are integers from 1 to 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Alpha desynchronization - Generate in silico EEG <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus images
# =============================================================================
# Image directories
img_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'alpha_desynchronization', 'stimuli')
categories = os.listdir(img_dir)
categories.sort()

# Loop across image categories
images = []
for cat in tqdm(categories):
    img_files = os.listdir(os.path.join(img_dir, cat))
    img_files.sort()

    # Load the images
    for ifile in img_files:
        img_path = os.path.join(img_dir, cat, ifile)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        images.append(img)

# Format the images
images = np.array(images)
images = np.swapaxes(images, 1, 3)  # BHWC to BCHW


# =============================================================================
# Generate the in silico EEG responses using BERG
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Empty result dictionaries
insilico_eeg = []
metadata = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(args.encoding_model, subject=sub)

    # Generate the in silico EEG responses, and average them across repeats
    eeg, metadata_sub = berg.encode(model, images, return_metadata=True)
    insilico_eeg.append(np.mean(eeg, 1))
    metadata.append(metadata_sub)
    del eeg, metadata_sub, model
    torch.cuda.empty_cache()
    gc.collect()

# Convert to numpy arrays
insilico_eeg = np.array(insilico_eeg)
 

# =============================================================================
# Save the results
# =============================================================================
results = {
    'insilico_eeg': insilico_eeg,
    'metadata': metadata
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'alpha_desynchronization', 'insilico_eeg_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = 'insilico_eeg_responses.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore