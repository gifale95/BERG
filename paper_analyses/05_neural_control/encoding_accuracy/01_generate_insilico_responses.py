"""Generate the in silico monkey electrophysiology responses for the 100 TVSD
test images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subjects : list
    The subject identifiers for the monkey encoding models. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
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
parser.add_argument('--encoding_model', type=str, default='eeg-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_control', 'encoding_accuracy',
    'insilico_responses', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Get the TVSD test images # !!!
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
# Loop across subjects
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
# Generate the in silico responses
# =============================================================================
    insilico_resp, metadata = berg.encode(model, images, return_metadata=True)
    insilico_resp = insilico_resp.astype(np.float32)


# =============================================================================
# Save the in silico responses
# =============================================================================
    data = {
        'insilico_resp': insilico_resp,
        'metadata': metadata
    }
    file_name = 'insilico_responses_sub-' + sub + '.npy'
    np.save(os.path.join(save_dir, file_name), data)

    # Delete unused variables
    del insilico_resp, data, metadata
    torch.cuda.empty_cache()
    gc.collect()