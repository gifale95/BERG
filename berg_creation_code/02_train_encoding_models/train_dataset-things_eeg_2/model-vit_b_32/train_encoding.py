"""Fit a linear regression to predict EEG data using DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X). A separate model is trained for each EEG channel and
time point, and also for each of the four EEG train data repeats: in this way,
for each input image we can have four different instances of synthetic EEG
response.

The feature maps come from a CLIP vision transformer, and are downsampled to 250
principal components using PCA.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
subject : int
	Number of the used THINGS EEG2 subject.
model : str
	Name of the used encoding model.
things_eeg_2_dir : str
	Directory of the THINGS EEG2 dataset.
	https://osf.io/3jk45/
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG

"""

import argparse
import torch
import numpy as np
import os
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
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--things_eeg_2_dir', default='../things_eeg_2', type=str)
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args = parser.parse_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220 + args.subject

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
# Extract the THINGS EEG2 training image features
# =============================================================================
# Image directories
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'training_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".JPEG"):
			image_list.append(os.path.join(root,file))
image_list.sort()

fmaps_train = []
with torch.no_grad():
	for i, img_dir in enumerate(tqdm(image_list, leave=False)):
		# Load the images
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
# Image directories
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'test_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".JPEG"):
			image_list.append(os.path.join(root,file))
image_list.sort()

fmaps_test = []
for i, img_dir in enumerate(tqdm(image_list, leave=False)):
	# Load the images
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
# Train the encoding models
# =============================================================================
# Load the training EEG responses
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
	'train_dataset-things_eeg_2')
eeg_dir = os.path.join(data_dir, 'eeg_sub-'+format(args.subject,'02')+
	'_split-train.h5')
eeg_train = h5py.File(eeg_dir, 'r')['eeg'][:]
n_repeats = eeg_train.shape[1]
n_channels = eeg_train.shape[2]
n_times = eeg_train.shape[3]

# Fit an encoding model at each EEG repeat, time-point and channel
reg_param = {}
eeg_test_pred = np.zeros((len(fmaps_test), n_repeats, n_channels, n_times),
	dtype=np.float32)

for r in range(eeg_train.shape[1]): # Loop over the 4 training EEG repeats
	# Reshape the EEG to (Samples x Features)
	eeg = eeg_train[:,r]
	eeg = np.reshape(eeg, (len(eeg), -1))
	# Fit the linear regression
	reg = LinearRegression()
	reg.fit(fmaps_train, eeg)

	# Store the linear regression weights
	reg_dict = {
		'coef_': reg.coef_,
		'intercept_': reg.intercept_,
		'n_features_in_': reg.n_features_in_
		}
	reg_param['rep-'+str(r+1)] = copy.deepcopy(reg_dict)

	# Use the learned weights to generate in silico EEG responses for the test
	# images
	eeg_test_pred_rep = reg.predict(fmaps_test)
	eeg_test_pred[:,r] = np.reshape(eeg_test_pred_rep, (-1, n_channels,
		n_times))
	del reg_dict

# Save the in silico EEG responses for the test images
save_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
	'modality-eeg', 'train_dataset-things_eeg_2', 'model-'+args.model)
file_name = 'eeg_test_pred_subject-' + str(args.subject) + '.npy'
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), eeg_test_pred)


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

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-'+args.model,
	'encoding_models_weights')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'weights_subject-' + format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), weights)
