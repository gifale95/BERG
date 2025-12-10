"""Perform RSA between in silico EEG responses and behavioral embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subject : int
    The subject identifier for the EEG encoding models. Since the used
    encoidng models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
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
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
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


# =============================================================================
# Define the vectorized correlation function to compute the RDMs
# =============================================================================
def corr_matrix(X):
    Xc = X - X.mean(axis=0)
    Xc /= np.sqrt((Xc**2).sum(axis=0))
    return (Xc.T @ Xc).astype(np.float32)


# =============================================================================
# Load the THINGS EEG2 image metadata
# =============================================================================
# The THINGS EEG2 image metadata can be downloaded from: https://osf.io/y63gw/files/qkgtf

# Load the metadata
metadata_dir = os.path.join(args.berg_dir, args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'behavioral_modeling', 'image_metadata.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()

# Get the test image category number based on the original THINGS database
test_img_concepts_THINGS = metadata['test_img_concepts_THINGS']


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the model metadata
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
    )
times = metadata['eeg']['times']

# EEG channel selection
ch_names = metadata['eeg']['ch_names']
kept_ch_names = []
for c in ch_names:
    for ch_select in args.channels:
        if ch_select in c:
            kept_ch_names.append(c)
            break

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={'channels': kept_ch_names}
    )


# =============================================================================
# Generate the in silico EEG responses
# =============================================================================
eeg = []

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

    # Generate the in silico EEG responses
    eeg_cat, metadata = berg.encode(model, images, return_metadata=True)

    # Store the in silico EEG responses averaged across image exemplars
    eeg.append(np.mean(eeg_cat, 0))

    # Delete unused variables
    del eeg_cat, images

# Convert the EEG responses to numpy arrays
eeg = np.array(eeg)
times = metadata['eeg']['times']


# =============================================================================
# Create the EEG RDM (pairwise decoding)
# =============================================================================
# The code assumes EEG responses in the format:
# (Image conditions × Repeats × Channels × Time points)

# Results array of shape:
# (Image conditions × Image conditions × EEG time points)
eeg_rdm = np.zeros((len(eeg), len(eeg), len(times)), dtype=np.float32)

# Loop over EEG time points and images
for t in tqdm(range(len(times))):
    for i1 in range(len(eeg)):
        for i2 in range(i1):

            # Select the image condition data
            eeg_cond_1 = eeg[i1,:,:,t] # type: ignore
            eeg_cond_2 = eeg[i2,:,:,t] # type: ignore

            # SVM target vectors
            y_train = np.zeros(((len(eeg_cond_1)-1)*2))
            y_train[int(len(y_train)/2):] = 1
            y_test = np.asarray((0, 1))
            scores = np.zeros(len(eeg_cond_1))

            # Loop across repeats (leave-one-repeat-out cross-decoding)
            for r in range(len(eeg_cond_1)):

                # Define the train/test partitions
                X_train = np.append(np.delete(eeg_cond_1, r, 0),
                    np.delete(eeg_cond_2, r, 0), 0)
                X_test = np.append(np.expand_dims(eeg_cond_1[r], 0),
                    np.expand_dims(eeg_cond_2[r], 0), 0)

                # Train the classifier
                dec_svm = SVC(kernel='linear')
                dec_svm.fit(X_train, y_train)

                # Test the classifier
                y_pred = dec_svm.predict(X_test)
                scores[r] = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            eeg_rdm[i1,i2,t] = np.mean(scores)
            eeg_rdm[i2,i1,t] = eeg_rdm[i1,i2,t]


# =============================================================================
# Create the behavioral RDM
# =============================================================================
# Load the behavioral embeddings (the behavioral emebddings can be downloaded
# from: https://osf.io/f5rn6/overview)
embedding_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'behavioral_modeling', 'spose_embedding_66d_sorted.txt')
beh_embeddings_all = np.array(pd.read_csv(embedding_dir, delim_whitespace=True,
    header=None)).astype(np.float32)

# Retain the embeddings from the 200 test image concepts
idx_test = np.zeros(len(test_img_concepts_THINGS), dtype=int)
for i, img in enumerate(test_img_concepts_THINGS):
    idx_test[i] = int(img[:5]) - 1
beh_embeddings = beh_embeddings_all[idx_test]

# Create the RDM
beh_rdm = 1 - corr_matrix(beh_embeddings.T)


# =============================================================================
# Perform RSA
# =============================================================================
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
    'eeg_rdm': eeg_rdm,
    'beh_rdm': beh_rdm,
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'behavioral_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_channels-' + \
    '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore