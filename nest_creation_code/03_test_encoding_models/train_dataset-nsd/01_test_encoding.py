"""Test the trained encoding models' predictions for the test stimuli, and save
the encoding accuracy as part of the trained encoding models' metadata.

Parameters
----------
subject : int
	Number of the used NSD subject.
rois : list of str
	List of all used ROIs.
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
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'OFA', 'FFA-1', 'FFA-2', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'PPA',
	'RSC', 'OPA', 'EBA', 'FBA-2', 'early', 'midventral', 'midlateral',
	'midparietal', 'parietal', 'lateral', 'ventral'])
parser.add_argument('--model', type=str, default='fwrf')
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--nest_dir', default='../datasetsneural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Loop over ROIs
# =============================================================================
for r in tqdm(args.rois):


# =============================================================================
# Load the in vivo tesing fMRI responses (and average them across repeats)
# =============================================================================
	data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
		'train_dataset-nsd', 'nsd_betas_sub-'+format(args.subject,'02')+'_roi-'+
		r+'.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()
	betas_bio = np.zeros((len(data_dict['test_img_num']),
		data_dict['betas'].shape[1]), dtype=np.float32)
	for i, img in enumerate(data_dict['test_img_num']):
		idx = np.where(data_dict['img_presentation_order'] == img)[0]
		betas_bio[i] = np.mean(data_dict['betas'][idx], 0)
	ncsnr = data_dict['ncsnr']

	# Convert the ncsnr to noise ceiling
	norm_term = (len(betas_bio) / 3) / len(betas_bio)
	noise_ceiling = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)


# =============================================================================
# Load the in silico fMRI responses for the test images
# =============================================================================
	data_dir = os.path.join(args.nest_dir, 'results', 'test_encoding_models',
		'modality-fmri', 'train_dataset-nsd', 'model-'+args.model,
		'betas_test_pred_subject-'+str(args.subject)+'_roi-'+r+'.npy')
	betas_pred = np.load(data_dir)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
	# Correlate the biological and predicted data
	correlation = np.zeros(betas_pred.shape[1])
	for v in range(len(correlation)):
		correlation[v] = pearsonr(betas_bio[:,v], betas_pred[:,v])[0]
	del betas_bio

	# Set negative correlation scores to 0
	correlation[correlation<0] = 0

	# Turn the correlations into r2 scores
	r2 = correlation ** 2

	# Add a very small number to noise ceiling values of 0, otherwise
	# the noise-ceiling-normalized encoding accuracy cannot be calculated
	# (division by 0 is not possible)
	noise_ceiling[noise_ceiling==0] = 1e-14

	# Compute the noise-ceiling-normalized encoding accuracy
	explained_variance = np.divide(r2, noise_ceiling) * 100

	# Set the noise-ceiling-normalized encoding accuracy to 1 for those
	# voxels in which the correlation is higher than the noise ceiling, to
	# prevent encoding accuracy values higher than 100%
	explained_variance[explained_variance>100] = 100
	del betas_pred


# =============================================================================
# Save the encoding accuracy as part of the encoding models metadata
# =============================================================================
	metadata = {}

	# fMRI-related metadata
	fmri = {
		'ncsnr': data_dict['ncsnr'],
		'roi_mask_volume': data_dict['roi_mask_volume'],
		'fmri_affine': data_dict['fmri_affine'],
		}
	metadata['fmri'] = fmri

	# Encoding-models-related metadata
	encoding_models = {
		'r2': r2,
		'noise_ceiling': noise_ceiling,
		'explained_variance': explained_variance,
		'train_img_num': data_dict['train_img_num'],
		'val_img_num': data_dict['val_img_num'],
		'test_img_num': data_dict['test_img_num']
		}
	metadata['encoding_models'] = encoding_models
	del data_dict

	# Save the metadata
	save_dir = os.path.join(args.nest_dir, 'encoding_models', 'modality-fmri',
		'train_dataset-nsd', 'model-'+args.model, 'metadata')
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	file_name = 'metadata_sub-' + format(args.subject, '02') + '_roi-' + r + \
		'.npy'
	np.save(os.path.join(save_dir, file_name), metadata)
