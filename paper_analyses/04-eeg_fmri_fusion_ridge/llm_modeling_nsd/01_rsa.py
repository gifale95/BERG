"""Perform searchlight RSA between t-fMRI responses and LLM embeddings.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
criterion : str
    Criterion to define the searchlight neighborhood: 'radius' for all vertices
    within a geodesic radius, 'nearest' for k-nearest neighbors.
radius_mm : float
    Geodesic radius in millimeters (default = 10 mm), if criterion is 'radius'.
k : int
    Number of nearest geodesic neighbors (default = 10), if criterion is
    'nearest'.
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
import numpy as np
from tqdm import tqdm
from berg import BERG
import h5py
from sklearn.linear_model import Ridge
import pandas as pd
from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import h5py
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_reps', default='average', type=str)
parser.add_argument('--criterion', default='radius', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--coco_dir', default='/scratch/giffordale95/datasets/image_sets/coco', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate t-fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define the vectorized correlation function
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
    subject=args.fmri_subject
)
test_img_num = metadata['encoding_models']['test_img_num']

# Load the test images
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')
images = sdataset[test_img_num]
images = np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)


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
# Generate the in silico EEG image responses
# =============================================================================
# Loop across EEG subjects
for s, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=esub
    )

    # Generate and store the in silico EEG responses
    if s == 0:
        eeg = berg.encode(model, images)
    else:
        eeg = np.append(eeg, berg.encode(model, images), 2)

    # Delete unused variables
    torch.cuda.empty_cache()
    gc.collect()
    del model


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Fusion model and save directories
model_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge',
    'encoding_fusion_weights', f'eeg_reps-{args.eeg_reps}')
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge',
    'llm_modeling', 'tfmri_responses', f'eeg_reps-{args.eeg_reps}')
os.makedirs(save_dir, exist_ok=True)

# Only select vertices falling within the NSD visual streams
n_vertex = 163842
idx_v = np.zeros(n_vertex, dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]

# Empty t-fMRI response array of shape:
# (515 Images, 163842 Vertices, 4 EEG repeats, 30 Time points)
n_rep = eeg.shape[1]
time_range = np.arange(20, 50) # !!! CHANGE
tfmri = np.zeros((len(eeg), n_vertex, n_rep, len(time_range)), dtype=np.float32)

# Loop across EEG time points
for t, t_idx in enumerate(time_range):

    if args.eeg_reps == 'average':

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                    f'hemi-{args.hemisphere}_eeg_time-{t_idx:03d}.npy')
        reg_param = np.load(os.path.join(model_dir, file_name),
            allow_pickle=True).item()

        # Instantiate the fusion regression model
        reg = Ridge()
        reg.coef_ = reg_param['coef_']
        reg.intercept_ = reg_param['intercept_']
        reg.n_features_in_ = reg_param['n_features_in_']

        # Generate the t-fMRI responses
        for r in range(eeg.shape[1]):
            tfmri[:,idx_v,r,t] = reg.predict(eeg[:,r,:,t_idx])

    elif args.eeg_reps == 'single':

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                    f'hemi-{args.hemisphere}_eeg_time-{t_idx:03d}.npy')
        reg_param = np.load(os.path.join(model_dir, file_name),
            allow_pickle=True)

        # Loop across EEG repeats
        for r in range(len(reg_param)):

            # Instantiate the fusion regression model
            reg = Ridge()
            reg.coef_ = reg_param[r]['coef_']
            reg.intercept_ = reg_param[r]['intercept_']
            reg.n_features_in_ = reg_param[r]['n_features_in_']

            # Generate the t-fMRI responses
            tfmri[:,idx_v,r,t] = reg.predict(eeg[:,r,:,t_idx])


# =============================================================================
# Perform searchlight RSA
# =============================================================================
# Empty RSA results array of shape:
# (163842 Vertices, 4 EEG repeats, 140 Time points)
rsa = np.empty((tfmri.shape[1], tfmri.shape[2], tfmri.shape[3]), dtype=np.float32)
rsa[:] = np.nan

# Take the lower triangle of the LLM RDM
idx_tril = np.tril_indices(len(llm_rdm), -1)
llm_rdm_tril = llm_rdm[idx_tril]

# Access the precomputed geodesic distances
data_dir = os.path.join(args.berg_dir, 'geodesic_vertex_distances',
    'geodesic_vertex_distances_'+args.hemisphere+'.h5')
geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances']

# Loop across fMRI vertices and EEG time points
for v in tqdm(idx_v):

    # Select the neighborhood based on the chosen criterion
    if args.criterion == 'nearest':
        # Get k-smallest distances (including the target vertex)
        neighborhood = np.argsort(geodesic_distances[v])[:args.k]
    elif args.criterion == 'radius':
        # Select all vertices whose distance is within the radius
        mask = geodesic_distances[v] <= args.radius_mm
        neighborhood = np.where(mask)[0]

    # Loop across EEG time points and repeats
    for t in range(tfmri.shape[3]):
        for r in range(tfmri.shape[2]):

            # Create the fMRI RDM
            fmri_rdm = 1 - corr_matrix(tfmri[:,neighborhood,r,t].T)

            # Perform RSA
            rsa[v,r,t] = pearsonr(llm_rdm_tril, fmri_rdm[idx_tril])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge', 'llm_modeling_nsd',
    'rsa', f'eeg_reps-{args.eeg_reps}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'rsa_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}.npy'

np.save(os.path.join(save_dir, file_name), results)