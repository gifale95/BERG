"""Perform RSA between in silico MEG responses and LLM embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico MEG
    responses.
subject : int
    The subject identifier for the MEG encoding models.
sensors : string
    String containing the MEG sensor type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal).
tmax : float
    Maximum epoch time point for the MEG analyses.
berg_dir : str
    Directory of the BERG.
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/
coco_dir : str
    Directory of the COCO dataset.
    https://cocodataset.org/

"""

import argparse
import os
import random
import numpy as np
import h5py
from tqdm import tqdm
from berg import BERG
import pandas as pd
from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='meg-things_meg_1-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--sensors', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--tmax', default=0.6, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--coco_dir', default='/scratch/giffordale95/datasets/image_sets/coco', type=str)
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
    """
    Computes the correlation matrix of the input data.
    Parameters
    ----------
    X : (N, M) float array
        Input data matrix with N features and M samples.

    Returns
    -------
    corr : (M, M) float array
        Correlation matrix of the input data.
    """

    Xc = X - X.mean(axis=0)
    Xc /= np.sqrt((Xc**2).sum(axis=0))

    return (Xc.T @ Xc).astype(np.float32)


# =============================================================================
# Load the 515 NSD test images
# =============================================================================
# The test images consist of the 515 images that all NSD subjects saw for three
# times, and which were used to test BERG's encoding models

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the test image number
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
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
# MEG time point and sensor selection
# =============================================================================
# Load the MEG model metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_meg = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
)

# Time point selection
times = metadata_meg['meg']['times']
timepoints = np.zeros(len(times), dtype=int)
timepoints[times <= args.tmax] = 1
times = times[times <= args.tmax]

# Sensor selection
region = []
if 'O' in args.sensors:
    region += ['Occipital']
if 'P' in args.sensors:
    region += ['Parietal']
if 'T' in args.sensors:
    region += ['Temporal']
if 'C' in args.sensors:
    region += ['Central']
if 'F' in args.sensors:
    region += ['Frontal']


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    train_splits='all',
    selection={'region': region,
               'timepoints': timepoints}
    )


# =============================================================================
# Generate the in silico MEG responses
# =============================================================================
meg, metadata = berg.encode(model, images, return_metadata=True)


# =============================================================================
# Create the MEG RDM (pairwise decoding)
# =============================================================================
# The code assumes MEG responses in the format:
# (Image conditions × Repeats × Sensors × Time points)

# Results array of shape:
# (Image conditions × Image conditions × MEG time points)
meg_rdm = np.zeros((len(meg), len(meg), len(times)), dtype=np.float32)

# Loop over MEG time points and images
for t in tqdm(range(len(times))):
    for i1 in range(len(meg)):
        for i2 in range(i1):

            # Select the image condition data
            meg_cond_1 = meg[i1,:,:,t]
            meg_cond_2 = meg[i2,:,:,t]

            # SVM target vectors
            y_train = np.zeros(((len(meg_cond_1)-1)*2))
            y_train[int(len(y_train)/2):] = 1
            y_test = np.asarray((0, 1))
            scores = np.zeros(len(meg_cond_1))

            # Loop across repeats (leave-one-repeat-out cross-decoding)
            for r in range(len(meg_cond_1)):
                # Define the train/test partitions
                X_train = np.append(np.delete(meg_cond_1, r, 0),
                    np.delete(meg_cond_2, r, 0), 0)
                X_test = np.append(np.expand_dims(meg_cond_1[r], 0),
                    np.expand_dims(meg_cond_2[r], 0), 0)

                # Train the classifier
                dec_svm = SVC(kernel='linear')
                dec_svm.fit(X_train, y_train)

                # Test the classifier
                y_pred = dec_svm.predict(X_test)
                scores[r] = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            meg_rdm[i1,i2,t] = np.mean(scores)
            meg_rdm[i2,i1,t] = meg_rdm[i1,i2,t]


# =============================================================================
# Create the LLM RDM
# =============================================================================
# Load the LLM
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Load the NSD image COCO IDs
info_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments', 'nsd',
    'nsd_stim_info_merged.csv') 
nsd_stim_info = np.array(pd.read_csv(info_dir, sep=',', header=0))
cocoId = nsd_stim_info[:,1]
cocoSplit = nsd_stim_info[:,2]

# Loop across test images
llm_embeddings = []
cocoSplit_img = ''
for img in tqdm(test_img_num):

    # Initialize the COCO api
    if cocoSplit[img] != cocoSplit_img:
        cocoSplit_img = cocoSplit[img]
        annFile = os.path.join(args.coco_dir, 'annotations', 'annotations',
            'captions_'+cocoSplit[img]+'.json')
        coco = COCO(annFile)

    # Get the 5 captions instances for each images
    annIds = coco.getAnnIds(imgIds=[cocoId[img]])
    annotations = coco.loadAnns(annIds)
    captions = []
    for ann in annotations:
        captions.append(ann['caption'])

    # Get the embeddings of the captions, and average them across caption
    # instances
    llm_embeddings.append(np.mean(embedding_model.encode(captions), 0))

# Format the embeddings to numpy array
llm_embeddings = np.array(llm_embeddings).astype(np.float32)

# Create the RDM
llm_rdm = 1 - corr_matrix(llm_embeddings.T)


# =============================================================================
# Perform RSA
# =============================================================================
# Take the lower triangle of the MEG and behavior RDMs
idx = np.tril_indices(len(meg_rdm), -1)
meg_rdm_tril = meg_rdm[idx]
llm_rdm_tril = llm_rdm[idx]

# Perform RSA
rsa = np.zeros(len(times), dtype=np.float32)
for t in range(len(times)): 
    rsa[t] = pearsonr(llm_rdm_tril, meg_rdm_tril[:,t])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'meg_rdm': meg_rdm,
    'llm_rdm': llm_rdm,
    'rsa': rsa,
    'metadata': metadata,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'llm_modeling', 'rsa', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_sensors-' + \
    '-'.join(args.sensors) + '.npy'

np.save(os.path.join(save_dir, file_name), results)