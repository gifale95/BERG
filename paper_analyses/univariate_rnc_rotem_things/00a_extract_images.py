"""Load, preprocess, and save the THINGS images.

Parameters
----------
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
from PIL import Image
import pandas as pd
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/ccn_datasets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Load, preprocess, and save the THINGS images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and preprocess the THINGS images
# =============================================================================
# Get the image paths
data_dir = os.path.join(args.things_dir, '01_image-level', 'image-paths.csv')
image_paths = pd.read_csv(data_dir, header=None).values.tolist()

# Load and preprocess the images
images = []
for img_path in tqdm(image_paths):
    img = Image.open(os.path.join(args.things_dir, img_path[0]))
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
    img = img.astype(np.uint8)
    images.append(img)

# Format the images to a numpy array
images = np.array(images)


# =============================================================================
# Save the preprocessed THINGS images
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'images')
os.makedirs(save_dir, exist_ok=True)

file_name = f'things.h5'

with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('images', data=images, dtype=np.uint8)