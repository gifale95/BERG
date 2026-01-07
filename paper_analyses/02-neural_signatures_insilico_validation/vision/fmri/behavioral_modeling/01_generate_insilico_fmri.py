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
# Load the THINGS EEG2 image metadata
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the metadata
metadata = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b-32',
    subject=1
    )

# Get the test image category number based on the original THINGS database
test_img_concepts_THINGS = metadata['encoding_models']['test_img_info']\
    ['test_img_concepts_THINGS']


# =============================================================================
# Load BERG's encoding model
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


# =============================================================================
# Generate the in silico fMRI responses
# =============================================================================
    fmri_lh = []
    fmri_rh = []

    # Loop across test object concepts
    for cat in tqdm(test_img_concepts_THINGS):

        # Get the image exemplar file names for each concept
        image_list = os.listdir(os.path.join(args.things_dir,
            'image-database_things', cat[6:]))
        image_list.sort()

        # Loop across image exemplars
        images = []
        for ifile in image_list:

            # Load the images
            img_path = os.path.join(args.things_dir, 'image-database_things',
                cat[6:], ifile)
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
            img = np.array(img)
            images.append(img)
        
        # Format the images
        images = np.array(images)
        images = np.swapaxes(images, 1, 3)  # BHWC to BCHW

        # Generate the in silico fMRI responses
        fmri_cat = berg.encode(model, images, return_metadata=False)

        # Store the in silico fMRI responses averaged across image exemplars
        fmri_lh.append(np.mean(fmri_cat[0], 0))
        fmri_rh.append(np.mean(fmri_cat[1], 0))

        # Delete unused variables
        del fmri_cat, images

    # Convert the in silico fMRI resposnes to numpy arrays
    fmri_lh = np.array(fmri_lh).astype(np.float32)
    fmri_rh = np.array(fmri_rh).astype(np.float32)


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