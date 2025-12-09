"""Perform RSA between in silico fMRI responses and behavioral embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses.
subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
import pandas as pd
from sklearn.svm import SVC
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


Remove hemisphere selection after finding way of vectorizing correlations


# =============================================================================
# Load the THINGS EEG2 image metadata
# =============================================================================
# The THINGS EEG2 image metadata can be downloaded from: https://osf.io/y63gw/files/qkgtf

# Load the metadata
metadata_dir = os.path.join(args.berg_dir, args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'behavioral_modeling', 'image_metadata.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()

# Get the test image category number based on the original THINGS database
test_img_concepts_THINGS = metadata['test_img_concepts_THINGS']


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Select the vertices from the chosen hemisphere
n_vertices = 163842
if args.hemisphere == 'lh':
    lh_vertices = np.ones(n_vertices, dtype=int)
    rh_vertices = np.zeros(n_vertices, dtype=int)
    rh_vertices[0] = 1 # At least one vertex must be selected
elif args.hemisphere == 'rh':
    lh_vertices = np.zeros(n_vertices, dtype=int)
    lh_vertices[0] = 1 # At least one vertex must be selected
    rh_vertices = np.ones(n_vertices, dtype=int)

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={
        'lh_vertices': lh_vertices,
        'rh_vertices': rh_vertices
        }
    )


# =============================================================================
# Generate the in silico fMRI responses # !!!
# =============================================================================
fmri = []

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
    fmri_cat, metadata = berg.encode(model, images, return_metadata=True)

    # Store the in silico fMRI responses averaged across image exemplars
    fmri.append(np.mean(fmri_cat, 0))

    # Delete unused variables
    del fmri_cat, images

# Convert the fMRI responses to numpy arrays
fmri = np.array(fmri)


# =============================================================================
# Create the behavioral RDM
# =============================================================================
# Load the behavioral embeddings (the behavioral emebddings can be downloaded
# from: https://osf.io/f5rn6/overview)
embedding_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'behavioral_modeling', 'spose_embedding_66d_sorted.txt')
beh_embeddings_all = np.array(pd.read_csv(embedding_dir, delim_whitespace=True,
    header=None))

# Retain the embeddings from the 200 test image concepts
idx_test = np.zeros(len(test_img_concepts_THINGS), dtype=int)
for i, img in enumerate(test_img_concepts_THINGS):
    idx_test[i] = int(img[:5]) - 1
beh_embeddings = beh_embeddings_all[idx_test]

# Create the RDM
beh_rdm = np.zeros((len(beh_embeddings), len(beh_embeddings)), dtype=np.float32)
for i1 in range(len(beh_embeddings)):
    for i2 in range(i1):
        beh_rdm[i1,i2] = 1 - pearsonr(beh_embeddings[i1], beh_embeddings[i2])[0] # type: ignore
        beh_rdm[i2,i1] = beh_rdm[i1,i2]


# =============================================================================
# Perform searchlight RSA # !!!
# =============================================================================
See how Adrien implements the searchlight


# Loop across fMRI vertices

# Create the fMRI RDMs

# Take the lower triangle of the EEG and behavior RDMs
idx = np.tril_indices(len(eeg_rdm), -1)
eeg_rdm_tril = eeg_rdm[idx]
beh_rdm_tril = beh_rdm[idx]

# Perform RSA
rsa = np.zeros(len(times), dtype=np.float32)
for t in range(len(times)): 
    rsa[t] = pearsonr(beh_rdm_tril, eeg_rdm_tril[:,t])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'fmri_rdm': fmri_rdm,
    'beh_rdm': beh_rdm,
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'behavioral_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_' + args.hemisphere + \
    '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore







X = np.random.randn(500, 1000).astype(np.float32)
y = np.random.randn(500).astype(np.float32)

def ultra_fast_corr(X, y):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    return num / den

def corr_matrix(X):
    Xc = X - X.mean(axis=0)
    Xc /= np.sqrt((Xc**2).sum(axis=0))
    return Xc.T @ Xc

corr = corr_matrix(X)

corr_2 = np.ones((X.shape[1], X.shape[1]), dtype=np.float32)
for i in tqdm(range(X.shape[1])):
    for j in range(i):
        corr_2[i,j] = pearsonr(X[:,i], X[:,j])[0]
        corr_2[j,i] = corr_2[i,j]