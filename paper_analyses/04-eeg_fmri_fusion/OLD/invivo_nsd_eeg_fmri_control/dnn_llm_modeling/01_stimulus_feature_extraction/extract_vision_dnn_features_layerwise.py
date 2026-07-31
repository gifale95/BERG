"""Extract layerwise vision DNN (AlexNet) features for the NSD stimuli of each 
ubject, and reduce them to N principal components using PCA, where N
corresponds to the number of principal components that explain 95% of the
variance.

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
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import h5py
from PIL import Image
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
vision_dnn_features_train = {}
for layer in model_layers:
    vision_dnn_features_train[layer] = []
with torch.no_grad():
    for img in tqdm(train_img_num):
        # Preprocess the images
        img = Image.fromarray(sdataset[img]).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Format the features
        for layer, v in ft.items():
            vision_dnn_features_train[layer].append(v.ravel().detach().numpy())
        del ft
# Format the features to numpy array
for layer in model_layers:
    vision_dnn_features_train[layer] = np.array(
        vision_dnn_features_train[layer]).astype(np.float32)

# Test stimuli
vision_dnn_features_test = {}
for layer in model_layers:
    vision_dnn_features_test[layer] = []
with torch.no_grad():
    for img in tqdm(test_img_num):
        # Preprocess the images
        img = Image.fromarray(sdataset[img]).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Format the features
        for layer, v in ft.items():
            vision_dnn_features_test[layer].append(v.ravel().detach().numpy())
        del ft
# Format the features to numpy array
for layer in model_layers:
    vision_dnn_features_test[layer] = np.array(
        vision_dnn_features_test[layer]).astype(np.float32)


# =============================================================================
# Downsample the vision DNN features using PCA
# =============================================================================
# Z-score the image features
for layer in model_layers:
    scaler = StandardScaler()
    scaler.fit(vision_dnn_features_train[layer])
    vision_dnn_features_train[layer] = scaler.transform(
        vision_dnn_features_train[layer])
    vision_dnn_features_test[layer] = scaler.transform(
        vision_dnn_features_test[layer])

    # Downsample the features with PCA
    pca = PCA(random_state=20200220)
    pca.fit(vision_dnn_features_train[layer])
    vision_dnn_features_train[layer] = pca.transform(
        vision_dnn_features_train[layer])
    vision_dnn_features_test[layer] = pca.transform(
        vision_dnn_features_test[layer])

    # Only retain the N principal components that explain 95% of the variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
    vision_dnn_features_train[layer] = \
        vision_dnn_features_train[layer][:,:n_components_95]
    vision_dnn_features_test[layer] = \
        vision_dnn_features_test[layer][:,:n_components_95]


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

file_name = f'vision_dnn_features_layerwise_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), results)