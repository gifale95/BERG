"""Extract the THINGS EEG2 stimulus features for each AlexNet layer, and
downsample them to 250 principal components using PCA.

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
from berg import BERG
from PIL import Image
import torch
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/ccn_datasets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract stimulus features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Get the THINGS EEG2 image metadata
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

metadata_things = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
    )

train_img_files = metadata_things['encoding_models']['train_img_info']\
    ['train_img_files']
test_img_files = metadata_things['encoding_models']['test_img_info']\
    ['test_img_files']


# =============================================================================
# Load the AlexNet model
# =============================================================================
# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names('alexnet')

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

# Define the image preprocessing
transform = trn.Compose([
    trn.Lambda(lambda img: trn.CenterCrop(min(img.size))(img)),
    trn.Resize((224,224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Extract the image features
# =============================================================================
# Train images
with torch.no_grad():
    for i, file in enumerate(tqdm(train_img_files, leave=False)):
        # Find correct subfolder
        img_path = None
        for root, _, files in os.walk(os.path.join(args.things_dir)):
            if file in files:
                img_path = os.path.join(root, file)
                break
        # Load and preprocess the images
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Store the features
        if i == 0:
            ft_train = {}
            for key, val in ft.items():
                ft_train[key] = []
        for key, val in ft.items():
            ft_train[key].append(val.cpu().detach().numpy().flatten())
        del ft
    for key, val in ft_train.items():
        ft_train[key] = np.array(val)

# Test images
with torch.no_grad():
    for i, file in enumerate(tqdm(test_img_files, leave=False)):
        # Find correct subfolder
        img_path = None
        for root, _, files in os.walk(os.path.join(args.things_dir)):
            if file in files:
                img_path = os.path.join(root, file)
                break
        # Load and preprocess the images
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Store the features
        if i == 0:
            ft_test = {}
            for key, val in ft.items():
                ft_test[key] = []
        for key, val in ft.items():
            ft_test[key].append(val.cpu().detach().numpy().flatten())
        del ft
    for key, val in ft_test.items():
        ft_test[key] = np.array(val)

# Z-score the features
for layer in ft_train.keys():
    scaler = StandardScaler()
    scaler.fit(ft_train[layer])
    ft_train[layer] = scaler.transform(ft_train[layer])
    ft_test[layer] = scaler.transform(ft_test[layer])

# Downsample the features with PCA
n_components = 250
if n_components > len(train_img_files):
    n_components = len(train_img_files)
for layer in ft_train.keys():
    pca = PCA(n_components=n_components, random_state=20200220)
    pca.fit(ft_train[layer])
    ft_train[layer] = pca.transform(ft_train[layer])
    ft_test[layer] = pca.transform(ft_test[layer])


# =============================================================================
# Save the stimulus features
# =============================================================================
data = {
    'ft_train': ft_train,
    'ft_test': ft_test
    }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stimulus_features')
os.makedirs(save_dir, exist_ok=True)

file_name = 'alexnet_layerwise_stimulus_features.npy'

np.save(os.path.join(save_dir, file_name), data)