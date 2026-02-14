"""Compute the stimulus RDMs using features from each DNN layer.

Parameters
----------
source_dataset : str
    If 'things_eeg_2', the source dataset is THINGS EEG2. If 'things_meg_1',
    the source dataset  is THINGS MEG1. (The source dataset is the dataset that
    is mapped onto fMRI responses.)
dnn_model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
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
parser.add_argument('--source_dataset', default='things_eeg_2', type=str)
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
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
# Get the 200 THINGS EEG2 test images file names
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the M/EEG metadata
if args.source_dataset == 'things_eeg_2':
    metadata = berg.get_model_metadata(
        'eeg-things_eeg_2-vit_b_32',
        subject=1
    )
elif args.source_dataset == 'things_meg_1':
    metadata = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=1
    )

# Get the test image file names
if args.source_dataset == 'things_eeg_2':
    test_img_files = metadata['encoding_models']['test_img_info']\
        ['test_img_files']
elif args.source_dataset == 'things_meg_1':
    img_ids = metadata['encoding_model']['test_img_ids']
    img_stimuli = metadata['encoding_model']['test_stimuli']
    unique_img_ids = np.unique(img_ids)
    test_img_files = []
    for id in unique_img_ids:
        test_img_files.append(img_stimuli[np.where(img_ids == id)[0][0]])


# =============================================================================
# Load the DNN
# =============================================================================
# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
# AlexNet
if args.dnn_model == 'alexnet':
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
elif args.dnn_model == 'resnet50':
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
    for i, file in enumerate(tqdm(test_img_files, leave=False)):
        # Find correct subfolder
        img_path = None
        for root, _, files in os.walk(os.path.join(args.things_dir)):
            if file in files:
                img_path = os.path.join(root, file)
                break
        # Preprocess the images
        img = Image.open(img_path).convert('RGB')
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
n_components = 250
if n_components > len(test_img_files):
    n_components = len(test_img_files)
for key, val in ft_dict.items():
    pca = PCA(n_components=n_components, random_state=20200220)
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
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'dnn_rdms',
    f'source_dataset-{args.source_dataset}')
os.makedirs(save_dir, exist_ok=True)

file_name = 'dnn_rdms_' + args.dnn_model + '.npy'

np.save(os.path.join(save_dir, file_name), dnn_rdms)