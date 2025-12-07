"""Generate in silico MEG responses for images of faces and objects.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of the subject identifiers for the MEG encoding models. Since the
    used encoding models are trained on THINGS MEG1 data, valid subject
    identifiers are integers from 1 to 4.
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
parser.add_argument('--encoding_model', type=str, default='meg-things_meg_1-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> MEG M170 - Generate in silico MEG <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus images
# =============================================================================
# Image directories
img_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'm170_faces', 'stimuli')
categories = ['Faces', 'Objects']
img_type = ['Sel', 'Test']

# Loop across image categories and types
images = {}
for cat in tqdm(categories):
    img_cat = []
    for itype in img_type:
        # Load the images
        img_list = os.listdir(os.path.join(img_dir, cat+'-'+itype))
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(img_dir, cat+'-'+itype, img_name)
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
            img = np.array(img)
            img_cat.append(img)
    img_cat = np.array(img_cat)
    img_cat = np.swapaxes(img_cat, 1, 3)  # BHWC to BCHW
    images[cat] = img_cat
    del img_cat


# =============================================================================
# Generate the in silico EEG responses using BERG
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Empty result dictionaries
insilico_meg = {}
metadata = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(args.encoding_model, subject=sub)

    # Loop across image categories
    for c, cat in enumerate(categories):

        # Create empty lists inside the result dicionaries
        if s == 0:
            insilico_meg[cat] = []

        # Generate the in silico MEG responses
        meg, metadata_sub = berg.encode(model, images[cat],
            return_metadata=True)
        insilico_meg[cat].append(meg)
        if c == 0:
            metadata.append(metadata_sub)
        del meg, metadata_sub
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

# Convert to numpy arrays
for cat in categories:
    insilico_meg[cat] = np.array(insilico_meg[cat])
 

# =============================================================================
# Save the results
# =============================================================================
results = {
    'insilico_meg': insilico_meg,
    'metadata': metadata
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'm170_faces', 'insilico_meg_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = 'insilico_meg_responses.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore