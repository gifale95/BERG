"""Use BERG to generate the in silico fMRI responses used to test the
tripartite organization effect (Konkle & Caramazza, 2013).

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

print('>>> Tripartite organization <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus images
# =============================================================================
img_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'stimuli')

# Animals
dir_animals = 'Tripartite-Animals'
animals = os.listdir(os.path.join(img_dir, dir_animals))
animals.sort()
animal_img = []
for img_name in animals:
    img_path = os.path.join(img_dir, dir_animals, img_name)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    animal_img.append(img)
animal_img = np.array(animal_img)
animal_img = np.swapaxes(animal_img, 1, 3)  # BHWC to BCHW

# Big objects
dir_big_object = 'Tripartite-BigObjects'
big_objects = os.listdir(os.path.join(img_dir, dir_big_object))
big_objects.sort()
big_object_img = []
for img_name in big_objects:
    img_path = os.path.join(img_dir, dir_big_object, img_name)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    big_object_img.append(img)
big_object_img = np.array(big_object_img)
big_object_img = np.swapaxes(big_object_img, 1, 3)  # BHWC to BCHW

# Small objects
dir_small_objects = 'Tripartite-SmallObjects'
small_objects = os.listdir(os.path.join(img_dir, dir_small_objects))
small_objects.sort()
small_object_img = []
for img_name in small_objects:
    img_path = os.path.join(img_dir, dir_small_objects, img_name)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    small_object_img.append(img)
small_object_img = np.array(small_object_img)
small_object_img = np.swapaxes(small_object_img, 1, 3) # BHWC to BCHW


# =============================================================================
# Generate the in silico fMRI responses using BERG
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across subjects
lh_animals = []
rh_animals = []
lh_big_objects = []
rh_big_objects = []
lh_small_objects = []
rh_small_objects = []
metadata = []
for sub in tqdm(args.subjects):
      
    # Load the encoding model
    model = berg.get_encoding_model(args.encoding_model, subject=sub)

    # Generate the in silico fMRI responses for the animal images, and average
    # them across images from the same condition
    fmri_animals, metadata_sub = berg.encode(model, animal_img,
        return_metadata=True)
    lh_animals.append(np.mean(fmri_animals[0], 0).astype(np.float32))
    rh_animals.append(np.mean(fmri_animals[1], 0).astype(np.float32))
    metadata.append(metadata_sub)
    del fmri_animals, metadata_sub
    torch.cuda.empty_cache()
    gc.collect()
    
    # Generate the in silico fMRI responses for the big object images, and
    # average them across images from the same condition
    fmri_big_objects = berg.encode(model, big_object_img)
    lh_big_objects.append(np.mean(fmri_big_objects[0], 0).astype(np.float32))
    rh_big_objects.append(np.mean(fmri_big_objects[1], 0).astype(np.float32)) # type: ignore
    del fmri_big_objects
    torch.cuda.empty_cache()
    gc.collect()
    
    # Generate the in silico fMRI responses for the small object images, and
    # average them across images from the same condition
    fmri_small_objects = berg.encode(model, small_object_img)
    lh_small_objects.append(np.mean(fmri_small_objects[0], 0).astype(np.float32))
    rh_small_objects.append(np.mean(fmri_small_objects[1], 0).astype(np.float32)) # type: ignore
    del fmri_small_objects
    torch.cuda.empty_cache()
    gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

# Convert to numpy arrays
lh_animals = np.array(lh_animals)
rh_animals = np.array(rh_animals)
lh_big_objects = np.array(lh_big_objects)
rh_big_objects = np.array(rh_big_objects)
lh_small_objects = np.array(lh_small_objects)
rh_small_objects = np.array(rh_small_objects)
 

# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_animals': lh_animals,
    'rh_animals': rh_animals,
    'lh_big_objects': lh_big_objects,
    'rh_big_objects': rh_big_objects,
    'lh_small_objects': lh_small_objects,
    'rh_small_objects': rh_small_objects,
    'metadata': metadata
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'insilico_fmri_responses')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

file_name = 'insilico_fmri_responses.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore