"""Fit a linear regression to predict fMRI responses using DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X). A separate model is trained for each fMRI vertex.

The feature maps come from a CLIP vision transformer, and are downsampled to 250
principal components using PCA.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
subject : int
	Number of the used NSD subject.
model : str
	Name of the used encoding model.
nsd_dir : str
	Directory of the Natural Scenes Dataset (NSD).
	https://naturalscenesdataset.org/
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import os
import h5py
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the image presentation order and the training/testing splits
# =============================================================================
data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
	'train_dataset-nsd_fsaverage')

metadata = np.load(os.path.join(data_dir, 'metadata_subject-'+
	str(args.subject)+'.npy'), allow_pickle=True).item()

img_presentation_order = metadata['img_presentation_order']
train_img_num = metadata['train_img_num']
test_img_num = metadata['test_img_num']


# =============================================================================
# Access the NSD images
# =============================================================================
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
	'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')


# =============================================================================
# Vision model
# =============================================================================
# Load the model
model = torchvision.models.vit_b_32(weights='DEFAULT')

# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
model_layers = [
	'encoder.layers.encoder_layer_0.add_1',
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
	'encoder.layers.encoder_layer_11.add_1'
	]

feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
feature_extractor.to(device)
feature_extractor.eval()


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
# Extract the image features
# =============================================================================
# Training images
fmaps_train = []
with torch.no_grad():
	for i in tqdm(train_img_num, leave=False):
		# Preprocess the images
		img = sdataset[i]
		img = Image.fromarray(img).convert('RGB')
		img = transform(img).unsqueeze(0)
		img = img.to(device)
		# Extract the features
		ft = feature_extractor(img)
		# Flatten the features
		ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
		fmaps_train.append(np.squeeze(ft.detach().cpu().numpy()))
		del ft
fmaps_train = np.asarray(fmaps_train)

# Test images
fmaps_test = []
with torch.no_grad():
	for i in tqdm(test_img_num, leave=False):
		# Preprocess the images
		img = sdataset[i]
		img = Image.fromarray(img).convert('RGB')
		img = transform(img).unsqueeze(0)
		img = img.to(device)
		# Extract the features
		ft = feature_extractor(img)
		# Flatten the features
		ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
		fmaps_test.append(np.squeeze(ft.detach().cpu().numpy()))
		del ft
fmaps_test = np.asarray(fmaps_test)


# =============================================================================
# Downsample the features using PCA
# =============================================================================
# Standardize the features
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)
fmaps_test = scaler.transform(fmaps_test)

# Apply PCA
pca = PCA(n_components=250, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_train = fmaps_train.astype(np.float32)
fmaps_test = pca.transform(fmaps_test)
fmaps_test = fmaps_test.astype(np.float32)


# =============================================================================
# Train the encoding models
# =============================================================================
# Load the training fMRI responses (and average them across repeats)
lh_betas_train = []
rh_betas_train = []
lh_betas_dir = os.path.join(data_dir, 'lh_betas_subject-'+str(args.subject)+
	'.h5')
rh_betas_dir = os.path.join(data_dir, 'rh_betas_subject-'+str(args.subject)+
	'.h5')
lh_betas_train_all = h5py.File(lh_betas_dir, 'r')['betas']
rh_betas_train_all = h5py.File(rh_betas_dir, 'r')['betas']
for i in train_img_num:
	idx = np.where(img_presentation_order == i)[0]
	lh_betas_train.append(np.mean(lh_betas_train_all[idx], 0))
	rh_betas_train.append(np.mean(rh_betas_train_all[idx], 0))
lh_betas_train = np.asarray(lh_betas_train)
rh_betas_train = np.asarray(rh_betas_train)
# Set NaN values (missing fMRI data) to zero
lh_betas_train = np.nan_to_num(lh_betas_train)
rh_betas_train = np.nan_to_num(rh_betas_train)

# Train encoding models using the NSD-core subject-unique images: fit the
# regression models at each fMRI vertex
lh_reg = LinearRegression().fit(fmaps_train, lh_betas_train)
rh_reg = LinearRegression().fit(fmaps_train, rh_betas_train)

# Use the learned weights to generate in silico fMRI responses for the test
# images
lh_betas_test_pred = lh_reg.predict(fmaps_test)
rh_betas_test_pred = rh_reg.predict(fmaps_test)

# Save the in silico fMRI responses for the test images
save_dir = os.path.join(args.nest_dir, 'results', 'test_encoding_models',
	'modality-fmri', 'train_dataset-nsd_fsaverage', 'model-'+args.model)
lh_file_name = 'lh_betas_test_pred_subject-' + str(args.subject) + '.npy'
rh_file_name = 'rh_betas_test_pred_subject-' + str(args.subject) + '.npy'
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, lh_file_name), lh_betas_test_pred)
np.save(os.path.join(save_dir, rh_file_name), rh_betas_test_pred)


# =============================================================================
# Save the trained encoding models
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
	'lh_reg_param': {
		'coef_': lh_reg.coef_,
		'intercept_': lh_reg.intercept_,
		'n_features_in_': lh_reg.n_features_in_
		},
	'rh_reg_param': {
		'coef_': rh_reg.coef_,
		'intercept_': rh_reg.intercept_,
		'n_features_in_': rh_reg.n_features_in_
		}
	}

save_dir = os.path.join(args.nest_dir, 'encoding_models', 'modality-fmri',
	'train_dataset-nsd_fsaverage', 'model-'+args.model,
	'encoding_models_weights')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'weights_subject-' + format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), weights)
