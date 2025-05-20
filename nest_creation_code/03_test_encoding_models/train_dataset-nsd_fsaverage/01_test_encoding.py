"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : int
	Number of the used NSD subject.
model : str
	Name of the used encoding model.
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the fMRI metadata
# =============================================================================
data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
	'train_dataset-nsd_fsaverage')

metadata_nsd = np.load(os.path.join(data_dir, 'metadata_subject-'+
	str(args.subject)+'.npy'), allow_pickle=True).item()


# =============================================================================
# Load the in vivo tesing fMRI responses (and average them across repeats)
# =============================================================================
lh_betas_test = []
rh_betas_test = []
lh_betas_dir = os.path.join(data_dir, 'lh_betas_subject-'+str(args.subject)+
	'.h5')
rh_betas_dir = os.path.join(data_dir, 'rh_betas_subject-'+str(args.subject)+
	'.h5')
lh_betas_test_all = h5py.File(lh_betas_dir, 'r')['betas']
rh_betas_test_all = h5py.File(rh_betas_dir, 'r')['betas']
for i in metadata_nsd['test_img_num']:
	idx = np.where(metadata_nsd['img_presentation_order'] == i)[0]
	lh_betas_test.append(np.mean(lh_betas_test_all[idx], 0))
	rh_betas_test.append(np.mean(rh_betas_test_all[idx], 0))
lh_betas_test = np.asarray(lh_betas_test)
rh_betas_test = np.asarray(rh_betas_test)
# Set NaN values (missing fMRI data) to zero
lh_betas_test = np.nan_to_num(lh_betas_test)
rh_betas_test = np.nan_to_num(rh_betas_test)

# Convert the ncsnr into noise ceiling
norm_term = 1 / 3
lh_noise_ceiling = (metadata_nsd['lh_ncsnr'] ** 2) / \
	((metadata_nsd['lh_ncsnr'] ** 2) + norm_term)
rh_noise_ceiling = (metadata_nsd['rh_ncsnr'] ** 2) / \
	((metadata_nsd['rh_ncsnr'] ** 2) + norm_term)


# =============================================================================
# Load the in silico fMRI responses for the test images
# =============================================================================
# These are whole brain in silico fMRI responses for each of the 515 test images
data_dir = os.path.join(args.nest_dir, 'results', 'test_encoding_models',
	'modality-fmri', 'train_dataset-nsd_fsaverage', 'model-'+args.model)
lh_betas_dir = os.path.join(data_dir, 'lh_betas_test_pred_subject-'+
	str(args.subject)+'.npy')
rh_betas_dir = os.path.join(data_dir, 'rh_betas_test_pred_subject-'+
	str(args.subject)+'.npy')

lh_betas_test_pred = np.load(lh_betas_dir)
rh_betas_test_pred = np.load(rh_betas_dir)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
# Correlate the in vivo and in silico fMRI responses
lh_correlation = np.zeros(lh_betas_test.shape[1])
rh_correlation = np.zeros(rh_betas_test.shape[1])
for v in range(lh_betas_test.shape[1]):
	lh_correlation[v] = pearsonr(lh_betas_test[:,v], lh_betas_test_pred[:,v])[0]
	rh_correlation[v] = pearsonr(rh_betas_test[:,v], rh_betas_test_pred[:,v])[0]

# Set negative correlation scores to zero
lh_correlation[lh_correlation<0] = 0
rh_correlation[rh_correlation<0] = 0

# Turn the correlations into r2 scores
lh_r2 = lh_correlation ** 2
rh_r2 = rh_correlation ** 2

# Add a very small number to noise ceiling values of 0, otherwise the
# noise-ceiling-normalized encoding accuracy cannot be calculated (division
# by 0 is not possible)
lh_noise_ceiling[lh_noise_ceiling==0] = 1e-14
rh_noise_ceiling[rh_noise_ceiling==0] = 1e-14

# Compute the noise-ceiling-normalized encoding accuracy
lh_explained_variance = np.divide(lh_r2, lh_noise_ceiling) * 100
rh_explained_variance = np.divide(rh_r2, rh_noise_ceiling) * 100

# Set the noise-normalized encoding accuracy to 100 for vertices where the
# the correlation is higher than the noise ceiling, to prevent encoding
# accuracy values higher than 100%
lh_explained_variance[lh_explained_variance>100] = 100
rh_explained_variance[rh_explained_variance>100] = 100


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
metadata = {}

# fMRI-related metadata
fmri = {
	'lh_ncsnr': metadata_nsd['lh_ncsnr'],
	'rh_ncsnr': metadata_nsd['rh_ncsnr'],
	'lh_fsaverage_rois': metadata_nsd['lh_fsaverage_rois'],
	'rh_fsaverage_rois': metadata_nsd['rh_fsaverage_rois']
	}
metadata['fmri'] = fmri

# Encoding-models-related metadata
encoding_models = {
	'train_img_num': metadata_nsd['train_img_num'],
	'val_img_num': metadata_nsd['val_img_num'],
	'test_img_num': metadata_nsd['test_img_num'],
	'lh_correlation': lh_correlation,
	'rh_correlation': rh_correlation,
	'lh_r2': lh_r2,
	'rh_r2': rh_r2,
	'lh_noise_ceiling': lh_noise_ceiling,
	'rh_noise_ceiling': rh_noise_ceiling,
	'lh_explained_variance': lh_explained_variance,
	'rh_explained_variance': rh_explained_variance
	}
metadata['encoding_models'] = encoding_models

# Save the metadata
save_dir = os.path.join(args.nest_dir, 'encoding_models', 'modality-fmri',
	'train_dataset-nsd_fsaverage', 'model-'+args.model, 'metadata')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
file_name = 'metadata_subject-' + format(args.subject, '02') + '.npy'
np.save(os.path.join(save_dir, file_name), metadata)
