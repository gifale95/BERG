"""Compute the stimulus RDMs using features from each DNN layer.

Parameters
----------
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
import h5py
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> DNN RDMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
# Load the 515 test images
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


# =============================================================================
# Load the DNN
# =============================================================================
# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
# AlexNet
if args.model == 'alexnet':
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

# ResNet-50
elif args.model == 'resnet50':
    # Load the model
    model = torchvision.models.resnet50(weights='DEFAULT')
    # Select the used layers for feature extraction
    #nodes, _ = get_graph_node_names(model)
    model_layers = [
        'layer1.2.relu_2',
        'layer2.3.relu_2',
        'layer3.5.relu_2',
        'layer4.2.relu_2',
        'fc'
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
with torch.no_grad():
    for i, img in enumerate(tqdm(test_img_num, leave=False)):
        # Preprocess the images
        img = Image.fromarray(sdataset[img]).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Store the features
        if i == 0:
            ft_dict = {}
            for key, val in ft.items():
                ft_dict[key] = []
        for key, val in ft.items():
            ft_dict[key].append(val.cpu().detach().numpy().flatten())
        del ft
    for key, val in ft_dict.items():
        ft_dict[key] = np.array(val)

# Z-score the features
for key, val in ft_dict.items():
    scaler = StandardScaler()
    ft_dict[key] = scaler.fit_transform(val)


# Downsample the features with PCA
for key, val in ft_dict.items():
    pca = PCA(n_components=250, random_state=20200220)
    ft_dict[key] = pca.fit_transform(val)


# =============================================================================
# Create the DNN RDMs
# =============================================================================
dnn_rdms = {}

# Create the RDM
for key, val in ft_dict.items():
    dnn_rdms[key] = 1 - corr_matrix(val.T)


# =============================================================================
# Save the DNN RDMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'dnn_rdms')
os.makedirs(save_dir, exist_ok=True)

file_name = 'dnn_rdms_' + args.model + '.npy'

np.save(os.path.join(save_dir, file_name), dnn_rdms)