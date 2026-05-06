"""Extract LLM embeddings for the NSD stimuli of each subject, and reduce them
to N principal components using PCA, where N corresponds to the number of
principal components that explain 95% of the variance.

Parameters
----------
subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
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
import pandas as pd
from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--coco_dir', default='/scratch/giffordale95/datasets/image_sets/coco', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract LLM embeddings <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the fMRI train/test image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-nsd_fsaverage')
meta_file_name = f'metadata_subject-{args.subject}.npy'
metadata_fmri = np.load(os.path.join(data_dir, meta_file_name),
    allow_pickle=True).item()

train_img_num = metadata_fmri['train_img_num']
train_img_num.sort()

test_img_num = metadata_fmri['test_img_num']
test_img_num.sort()


# =============================================================================
# Extract the LLM embeddings
# =============================================================================
# Load the LLM
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Load the NSD image COCO IDs
info_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments', 'nsd',
    'nsd_stim_info_merged.csv') 
nsd_stim_info = np.array(pd.read_csv(info_dir, sep=',', header=0))
cocoId = nsd_stim_info[:,1]
cocoSplit = nsd_stim_info[:,2]

# Train stimuli
llm_embeddings_train = []
cocoSplit_img = ''
for img in tqdm(train_img_num):
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
    llm_embeddings_train.append(np.mean(embedding_model.encode(captions), 0))
# Format the embeddings to numpy array
llm_embeddings_train = np.array(llm_embeddings_train).astype(np.float32)

# Test stimuli
llm_embeddings_test = []
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
    llm_embeddings_test.append(np.mean(embedding_model.encode(captions), 0))
# Format the embeddings to numpy array
llm_embeddings_test = np.array(llm_embeddings_test).astype(np.float32)


# =============================================================================
# Downsample the LLM embeddings using PCA
# =============================================================================
# Z-score the image features
scaler = StandardScaler()
scaler.fit(llm_embeddings_train)
llm_embeddings_train = scaler.transform(llm_embeddings_train)
llm_embeddings_test = scaler.transform(llm_embeddings_test)

# Downsample the features with PCA
pca = PCA(random_state=20200220)
pca.fit(llm_embeddings_train)
llm_embeddings_train = pca.transform(llm_embeddings_train)
llm_embeddings_test = pca.transform(llm_embeddings_test)

# Only retain the N principal components that explain 95% of the variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
llm_embeddings_train = llm_embeddings_train[:,:n_components_95]
llm_embeddings_test = llm_embeddings_test[:,:n_components_95]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'llm_embeddings_train': llm_embeddings_train,
    'llm_embeddings_test': llm_embeddings_test
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'stimulus_features')
os.makedirs(save_dir, exist_ok=True)

file_name = f'llm_embeddings_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), results)