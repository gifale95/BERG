"""Extract vision DNN features for the NSD stimuli of each subject, and reduce
them to 250 principal components using PCA.

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
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import h5py
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract vision DNN features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Define the image preprocessing
# =============================================================================
transform = trn.Compose([
    trn.Lambda(lambda img: trn.CenterCrop(min(img.size))(img)),
    trn.Resize((224,224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the deep neural network model
# =============================================================================
# AlexNet
# Load the model
model = torchvision.models.alexnet(weights='DEFAULT')

# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
model_layers = [
    'features.2',
    'features.5',
    'features.7',
    'features.9',
    'features.12',
    'classifier.2',
    'classifier.5',
    'classifier.6'
    ]

# Create the feature extractor
feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
feature_extractor.to(device)
feature_extractor.eval()


# =============================================================================
# Access the NSD-core images
# =============================================================================
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')


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
# Extract the vision DNN features
# =============================================================================
# Train stimuli
vision_dnn_features_train = []
with torch.no_grad():
    for img in tqdm(train_img_num):
        # Preprocess the images
        img = Image.fromarray(sdataset[img]).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Format the features
        vision_dnn_features_train.append(
            np.concatenate([v.ravel() for v in ft.values()]))
        del ft
# Format the features to numpy array
vision_dnn_features_train = np.array(
    vision_dnn_features_train).astype(np.float32)

# Test stimuli
vision_dnn_features_test = []
with torch.no_grad():
    for img in tqdm(test_img_num):
        # Preprocess the images
        img = Image.fromarray(sdataset[img]).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Format the features
        vision_dnn_features_test.append(
            np.concatenate([v.ravel() for v in ft.values()]))
        del ft
# Format the features to numpy array
vision_dnn_features_test = np.array(
    vision_dnn_features_test).astype(np.float32)


# =============================================================================
# Downsample the vision DNN features to 250 dimensions using PCA
# =============================================================================
# Z-score the image features
scaler = StandardScaler()
scaler.fit(vision_dnn_features_train)
vision_dnn_features_train = scaler.transform(vision_dnn_features_train)
vision_dnn_features_test = scaler.transform(vision_dnn_features_test)

# Downsample the features with PCA
n_components = 250
pca = PCA(n_components=n_components, random_state=20200220)
pca.fit(vision_dnn_features_train)
vision_dnn_features_train = pca.transform(vision_dnn_features_train)
vision_dnn_features_test = pca.transform(vision_dnn_features_test)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'vision_dnn_features_train': vision_dnn_features_train,
    'vision_dnn_features_test': vision_dnn_features_test
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'stimulus_features')
os.makedirs(save_dir, exist_ok=True)

file_name = f'vision_dnn_features_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), results)