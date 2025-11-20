"""Use BERG to generate the in silico fMRI responses used to test FFA- and
PPA-specific effects.

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
    'vision', 'fmri', 'ffa_ppa', 'stimuli')

# Loop across image effects
images = {}
effects = os.listdir(img_dir)
effects.sort()
for effect in tqdm(effects):

    # Loop across image types
    images[effect] = {}
    img_type = os.listdir(os.path.join(img_dir, effect))
    img_type.sort()
    for itype in img_type:

        # Loop across images
        img_cat = []
        img_list = os.listdir(os.path.join(img_dir, effect, itype))
        img_list.sort()
        for img_name in img_list:

            # Load the images
            img_path = os.path.join(img_dir, effect, itype, img_name)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img_cat.append(img)

        # Store the images
        img_cat = np.array(img_cat)
        img_cat = np.swapaxes(img_cat, 1, 3)  # BHWC to BCHW
        images[effect][itype] = img_cat
        del img_cat


# =============================================================================
# Generate the in silico fMRI responses using BERG
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Empty result dictionaries
lh_insilico_fmri_ffa1 = {}
lh_insilico_fmri_ffa2 = {}
lh_insilico_fmri_ppa = {}
rh_insilico_fmri_ffa1 = {}
rh_insilico_fmri_ffa2 = {}
rh_insilico_fmri_ppa = {}
metadata = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the encoding models for FFA and PPA
    model_ffa1 = berg.get_encoding_model(
        args.encoding_model,
        subject=sub,
        selection={'roi': 'FFA-1'})
    model_ffa2 = berg.get_encoding_model(
        args.encoding_model,
        subject=sub,
        selection={'roi': 'FFA-2'})
    model_ppa = berg.get_encoding_model(
        args.encoding_model,
        subject=sub,
        selection={'roi': 'PPA'})

    # Loop across image effects
    for e, effect in enumerate(images.keys()):

        # Create nested result dicionaries
        if s == 0:
            lh_insilico_fmri_ffa1[effect] = {}
            lh_insilico_fmri_ffa2[effect] = {}
            lh_insilico_fmri_ppa[effect] = {}
            rh_insilico_fmri_ffa1[effect] = {}
            rh_insilico_fmri_ffa2[effect] = {}
            rh_insilico_fmri_ppa[effect] = {}

        # Loop across image types
        for i, itype in enumerate(images[effect].keys()):

            # Create empty result lists
            if s == 0:
                lh_insilico_fmri_ffa1[effect][itype] = []
                lh_insilico_fmri_ffa2[effect][itype] = []
                lh_insilico_fmri_ppa[effect][itype] = []
                rh_insilico_fmri_ffa1[effect][itype] = []
                rh_insilico_fmri_ffa2[effect][itype] = []
                rh_insilico_fmri_ppa[effect][itype] = []

            # Generate the in silico fMRI responses
            fmri_ffa1, metadata_sub = berg.encode(model_ffa1,
                images[effect][itype], return_metadata=True)
            fmri_ffa2, metadata_sub = berg.encode(model_ffa2,
                images[effect][itype], return_metadata=True)
            fmri_ppa, metadata_sub = berg.encode(model_ppa,
                images[effect][itype], return_metadata=True)

            # Average the in silico fMRI responses across images, and store
            # them
            lh_insilico_fmri_ffa1[effect][itype].append(
                np.mean(fmri_ffa1[0], 0))
            rh_insilico_fmri_ffa1[effect][itype].append(
                np.mean(fmri_ffa1[1], 0))
            lh_insilico_fmri_ffa2[effect][itype].append(
                np.mean(fmri_ffa2[0], 0))
            rh_insilico_fmri_ffa2[effect][itype].append(
                np.mean(fmri_ffa2[1], 0))
            lh_insilico_fmri_ppa[effect][itype].append(
                np.mean(fmri_ppa[0], 0))
            rh_insilico_fmri_ppa[effect][itype].append(
                np.mean(fmri_ppa[1], 0))

            # Store the metadata
            if e == 0 and i == 0:
                metadata.append(metadata_sub)

            # Delete unused variables
            del fmri_ffa1, fmri_ffa2, fmri_ppa, metadata_sub
            torch.cuda.empty_cache()
            gc.collect()

    # Delete models
    del model_ffa1, model_ffa2, model_ppa
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_insilico_fmri_ffa1': lh_insilico_fmri_ffa1,
    'rh_insilico_fmri_ffa1': rh_insilico_fmri_ffa1,
    'lh_insilico_fmri_ffa2': lh_insilico_fmri_ffa2,
    'rh_insilico_fmri_ffa2': rh_insilico_fmri_ffa2,
    'lh_insilico_fmri_ppa': lh_insilico_fmri_ppa,
    'rh_insilico_fmri_ppa': rh_insilico_fmri_ppa,
    'metadata': metadata
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa', 'insilico_fmri_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = 'insilico_fmri_responses.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore