"""Use BERG to generate the in silico fMRI responses to food images.

Parameters
----------
encoding_model : str
    The name of the fMRI encoding model in BERG to use for generating the
    in silico fMRI responses in surface space.
subjects : list
    List of the subject identifiers for the fMRI encoding models. Since the
    used encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import numpy as np
import os
from PIL import Image
import torch
from berg import BERG
from tqdm import tqdm
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus images
# =============================================================================
# Image directories
img_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'food_selectivity', 'stimuli')
categories = ['food', 'body', 'face', 'house', 'word']

# Load and format the images
images = {}
for cat in tqdm(categories):
    img_cat = []
    img_list = os.listdir(os.path.join(img_dir, cat))
    img_list.sort()
    for img_name in img_list:
        img_path = os.path.join(img_dir, cat, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_cat.append(img)
    img_cat = np.array(img_cat)
    img_cat = np.swapaxes(img_cat, 1, 3)  # BHWC to BCHW
    images[cat] = img_cat
    del img_cat


# =============================================================================
# Generate the in silico fMRI responses using BERG
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Empty result dictionaries
lh_insilico_fmri = {}
rh_insilico_fmri = {}
metadata = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(args.encoding_model, subject=sub)

    # Loop across image categories
    for c, cat in enumerate(categories):

        # Create empty lists inside the result dicionaries
        if s == 0:
            lh_insilico_fmri[cat] = []
            rh_insilico_fmri[cat] = []

        # Generate the in silico fMRI responses, and average them across images
        fmri, metadata_sub = berg.encode(model, images[cat],
            return_metadata=True)
        lh_insilico_fmri[cat].append(np.mean(fmri[0], 0).astype(np.float32))
        rh_insilico_fmri[cat].append(np.mean(fmri[1], 0).astype(np.float32))
        if c == 0:
            metadata.append(metadata_sub)
        del fmri, metadata_sub
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

# Convert to numpy arrays
for cat in categories:
    lh_insilico_fmri[cat] = np.array(lh_insilico_fmri[cat])
    rh_insilico_fmri[cat] = np.array(rh_insilico_fmri[cat])
 

# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_insilico_fmri': lh_insilico_fmri,
    'rh_insilico_fmri': rh_insilico_fmri,
    'metadata': metadata
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'food_selectivity', 'insilico_fmri_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = 'insilico_fmri_responses.npy'

np.save(os.path.join(save_dir, file_name), results)