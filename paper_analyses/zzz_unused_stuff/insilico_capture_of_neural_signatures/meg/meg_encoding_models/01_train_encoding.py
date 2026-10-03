"""Fit a linear regression to predict MEG data from the THINGS MEG1 dataset
using DNN feature maps as predictors. The linear regression is trained using
the training images MEG data (Y) and feature maps (X). A separate model is
trained for each MEG sensor and time point. Furthermore, the training data is
is randomly split into four partitions, and a separate encoding model is
trained on each partition: in this way, for each input image we can have four
different instances of synthetic MEG response.

The feature maps come from a CLIP vision transformer, and are downsampled to
250 principal components using PCA.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
subject : int
    Number of the used THINGS MEG1 subject.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import torch
import numpy as np
import os
import random
import h5py
from tqdm import tqdm
import copy
from PIL import Image
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220 + args.subject
random.seed(seed)
np.random.seed(seed)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the MEG metadata
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_meg_1')
metadata_dir = os.path.join(data_dir, 'meg_P'+str(args.subject)+
    '_metadata.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()


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
# Vision model
# =============================================================================
# Load the model
model = torchvision.models.vit_b_32(weights='DEFAULT')
model.to(device)
model.eval()

# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
model_layers = ['encoder.layers.encoder_layer_0.add_1',
                'encoder.layers.encoder_layer_1.add_1',
                'encoder.layers.encoder_layer_2.add_1',
                'encoder.layers.encoder_layer_3.add_1',
                'encoder.layers.encoder_layer_4.add_1',
                'encoder.layers.encoder_layer_5.add_1',
                'encoder.layers.encoder_layer_6.add_1',
                'encoder.layers.encoder_layer_7.add_1',
                'encoder.layers.encoder_layer_8.add_1',
                'encoder.layers.encoder_layer_9.add_1',
                'encoder.layers.encoder_layer_10.add_1',
                'encoder.layers.encoder_layer_11.add_1']
feature_extractor = create_feature_extractor(model, return_nodes=model_layers)


# =============================================================================
# Extract the THINGS MEG1 training image features
# =============================================================================
# Get the train image IDs
train_img_ids = metadata['encoding_model']['full']['train_img_ids'].astype(int)
unique_train_img_ids = np.unique(train_img_ids)

# Get the directories of the training images
train_img_files = metadata['encoding_model']['full']['train_img_files']

# Extract the image features
fmaps_train = []
with torch.no_grad():
    for i, img_id in enumerate(tqdm(unique_train_img_ids, leave=False)):
        # Get the directory of the training image
        img_file = train_img_files[np.where(train_img_ids == img_id)[0][0]]
        # Load the images
        img_dir = os.path.join(args.things_dir, 'image-database_things',
            img_file)
        img = Image.open(img_dir).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        fmaps_train.append(np.squeeze(ft.detach().cpu().numpy()))
        del ft
fmaps_train = np.asarray(fmaps_train)

# Standardize the image features
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)

# Downsample the image features using PCA
pca = PCA(n_components=250, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_train = fmaps_train.astype(np.float32)


# =============================================================================
# Extract the THINGS EEG2 testing image features
# =============================================================================
# Get the test image IDs
test_img_ids = metadata['encoding_model']['test_things_img_ids'].astype(int)
unique_test_img_ids = np.unique(test_img_ids)

# Get the directories of the testing images
test_img_files = metadata['encoding_model']['test_img_files']

# Extract the image features
fmaps_test = []
with torch.no_grad():
    for i, img_id in enumerate(tqdm(unique_test_img_ids, leave=False)):
        # Get the directory of the test image
        img_file = test_img_files[np.where(test_img_ids == img_id)[0][0]]
        # Load the images
        img_dir = os.path.join(args.things_dir, 'image-database_things',
            img_file)
        img = Image.open(img_dir).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        fmaps_test.append(np.squeeze(ft.detach().cpu().numpy()))
        del ft
fmaps_test = np.asarray(fmaps_test)

# Standardize the image features
fmaps_test = scaler.transform(fmaps_test)

# Downsample the image features using PCA
fmaps_test = pca.transform(fmaps_test)


# =============================================================================
# Load and format the train MEG responses
# =============================================================================
meg_dir = os.path.join(data_dir, 'meg_P'+str(args.subject)+
    '_split-train.h5')
meg_train = h5py.File(meg_dir, 'r')['neural_data'][:]
n_sensors = metadata['sensors']['n_sensors']
n_times = len(metadata['meg']['times'])

# Reshape the MEG to (Samples x Features)
meg_train = np.reshape(meg_train, (len(meg_train), -1))

# Sort the MEG responses based on training image IDs
idx_sort = np.argsort(train_img_ids)
meg_train = meg_train[idx_sort]


# =============================================================================
# Train the encoding models
# =============================================================================
# Fit an encoding model for each time-point and sensor, using either all the
# training data, or each of the 4 training data splits separately
reg_param = {}
meg_test_pred = {}

# Train the encoding models using all MEG training data
# Fit the linear regression
reg = LinearRegression()
reg.fit(fmaps_train, meg_train)
# Store the linear regression weights
reg_dict = {
    'coef_': reg.coef_,
    'intercept_': reg.intercept_,
    'n_features_in_': reg.n_features_in_
    }
reg_param['split-all'] = copy.deepcopy(reg_dict)
# Use the learned weights to generate in silico MEG responses for the test
# images
meg_pred = reg.predict(fmaps_test)
meg_pred = np.reshape(meg_pred, (-1, n_sensors, n_times))
meg_test_pred['split-all'] = meg_pred
del reg_dict, meg_pred, reg

# Train the encoding models using single MEG training data splits
n_splits = 4
meg_pred = np.zeros((len(fmaps_test), n_splits, n_sensors, n_times),
    dtype=np.float32)
# Shuffle the training data indices
train_indices = np.arange(len(meg_train))
np.random.shuffle(train_indices)
split_size = len(meg_train) // n_splits
# Loop over the 4 training MEG splits
for s in range(n_splits):
    # Get the training split indices
    if s == n_splits - 1:
        split_indices = train_indices[s*split_size:]
    else:
        split_indices = train_indices[s*split_size:(s+1)*split_size]
    # Fit the linear regression
    reg = LinearRegression()
    reg.fit(fmaps_train[split_indices], meg_train[split_indices])
    # Store the linear regression weights
    reg_dict = {
        'coef_': reg.coef_,
        'intercept_': reg.intercept_,
        'n_features_in_': reg.n_features_in_
        }
    reg_param['split-'+str(s+1)] = copy.deepcopy(reg_dict)
    # Use the learned weights to generate in silico MEG responses for the test
    # images
    meg_pred_split = reg.predict(fmaps_test)
    meg_pred[:,s] = np.reshape(meg_pred_split, (-1, n_sensors, n_times))
    del reg_dict, reg, meg_pred_split
meg_test_pred['split-single'] = meg_pred
del meg_pred

# Save the in silico MEG responses for the test images
save_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-meg', 'train_dataset-things_meg_1', 'model-vit_b_32')
file_name = 'meg_test_pred_P' + str(args.subject) + '.npy'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), meg_test_pred)


# =============================================================================
# Save the trained encoding models weights
# =============================================================================
weights = {
    'scaler_param': {
        'scale_': scaler.scale_,
        'mean_': scaler.mean_,
        'var_': scaler.var_,
        'n_features_in_': scaler.n_features_in_,
        'n_samples_seen_': scaler.n_samples_seen_
        },
    'pca_param': {
        'components_': pca.components_,
        'explained_variance_': pca.explained_variance_,
        'explained_variance_ratio_': pca.explained_variance_ratio_,
        'singular_values_': pca.singular_values_,
        'mean_': pca.mean_,
        'n_components_': pca.n_components_,
        'n_samples_': pca.n_samples_,
        'noise_variance_': pca.noise_variance_,
        'n_features_in_': pca.n_features_in_
        },
    'reg_param': reg_param
    }

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', 'model-vit_b_32', 'encoding_models_weights')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

file_name = 'weights_P' + str(args.subject) + '.npy'

np.save(os.path.join(save_dir, file_name), weights)