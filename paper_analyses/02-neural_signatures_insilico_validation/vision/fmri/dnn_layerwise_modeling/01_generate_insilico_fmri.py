"""Generate the in silico fMRI responses for the 515 images that all NSD
subjects saw for three times.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses.
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
berg_dir : str
    Directory of the BERG.
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
import h5py
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'insilico_fmri_responses')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the 515 test images
# =============================================================================
# The test images consist of the 515 images that all NSD subjects saw for three
# times, and which were used to test BERG's encoding models

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the test image number
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=1
)
test_img_num = metadata['encoding_models']['test_img_num']

# Load the test images
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')
images = sdataset[test_img_num]
images = np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)


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


# =============================================================================
# Generate and save the in silico fMRI responses
# =============================================================================
    # Generate the in silico fMRI responses
    fmri = berg.encode(model, images, return_metadata=False)

    # Convert the in silico fMRI resposnes to numpy arrays
    fmri_lh = np.array(fmri[0]).astype(np.float32)
    fmri_rh = np.array(fmri[1]).astype(np.float32)


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
    del fmri, fmri_lh, fmri_rh
    torch.cuda.empty_cache()
    gc.collect()