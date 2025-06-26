"""Test the trained encoding models' predictions in-distribution (ID) on the
515 test images from NSD-core, and out-of-distribution (OOD) on the 286 images
from NSD-synthetic. Then save the encoding accuracy as part of the trained
encoding models' metadata.

Parameters
----------
subject : int
	Number of the used NSD subject.
model_id : str
	Unique identifier of the model to load.
nsd_dir : str
	Directory of the Natural Scenes Dataset.
	https://naturalscenesdataset.org/
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from scipy.stats import zscore
import h5py
from scipy.stats import pearsonr
from tqdm import tqdm
from berg import BERG


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--model_id', type=str, default='fmri-nsd_fsaverage-<your_model_name>')
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args = parser.parse_args()


# =============================================================================
# Load the fMRI metadata
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

for s, sub in enumerate(tqdm(args.subjects)):
	data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
		'train_dataset-nsd_fsaverage')

	metadata_nsd = np.load(os.path.join(data_dir, 'metadata_subject-'+
		str(sub)+'.npy'), allow_pickle=True).item()


# =============================================================================
# Load the in vivo NSD-core test fMRI responses (and average them across repeats)
# =============================================================================
	lh_betas_test_nsdcore = []
	rh_betas_test_nsdcore = []
	lh_betas_dir = os.path.join(data_dir, 'lh_betas_subject-'+str(sub)+'.h5')
	rh_betas_dir = os.path.join(data_dir, 'rh_betas_subject-'+str(sub)+'.h5')
	lh_betas_test_all = h5py.File(lh_betas_dir, 'r')['betas']
	rh_betas_test_all = h5py.File(rh_betas_dir, 'r')['betas']
	for i in metadata_nsd['test_img_num']:
		idx = np.where(metadata_nsd['img_presentation_order'] == i)[0]
		lh_betas_test_nsdcore.append(np.mean(lh_betas_test_all[idx], 0))
		rh_betas_test_nsdcore.append(np.mean(rh_betas_test_all[idx], 0))
	lh_betas_test_nsdcore = np.asarray(lh_betas_test_nsdcore)
	rh_betas_test_nsdcore = np.asarray(rh_betas_test_nsdcore)
	# Set NaN values (missing fMRI data) to zero
	lh_betas_test_nsdcore = np.nan_to_num(lh_betas_test_nsdcore)
	rh_betas_test_nsdcore = np.nan_to_num(rh_betas_test_nsdcore)

	# Convert the ncsnr into noise ceiling
	norm_term = 1 / 3
	lh_noise_ceiling_nsdcore = (metadata_nsd['lh_ncsnr'] ** 2) / \
		((metadata_nsd['lh_ncsnr'] ** 2) + norm_term)
	rh_noise_ceiling_nsdcore = (metadata_nsd['rh_ncsnr'] ** 2) / \
		((metadata_nsd['rh_ncsnr'] ** 2) + norm_term)


# =============================================================================
# Load the in vivo NSD-synthetic fMRI responses (and average them across repeats)
# =============================================================================
	# Load the experimental design info
	expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
		'nsdsynthetic', 'nsdsynthetic_expdesign.mat'))
	# Subtract 1 since the indices start with 1 (and not 0)
	masterordering = np.squeeze(expdesign['masterordering'] - 1)
	# Get the NSD synthetic image condition repeats

	# Load the fMRI betas
	betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
		format(sub, '02'), 'fsaverage',
		'nsdsyntheticbetas_fithrf_GLMdenoise_RR')
	lh_file_name = 'lh.betas_nsdsynthetic.mgh'
	rh_file_name = 'rh.betas_nsdsynthetic.mgh'
	lh_betas_all = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		lh_file_name)).get_fdata())).astype(np.float32)
	rh_betas_all = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		rh_file_name)).get_fdata())).astype(np.float32)
	# z-score the betas of each vertex within the scan session
	lh_betas_all = zscore(lh_betas_all, nan_policy='omit')
	rh_betas_all = zscore(rh_betas_all, nan_policy='omit')

	# Average the fMRI betas across repeats
	nsdsynthetic_img_num = np.unique(masterordering)
	lh_betas_test_nsdsynthetic = np.zeros((len(nsdsynthetic_img_num),
		lh_betas_all.shape[1]))
	rh_betas_test_nsdsynthetic = np.zeros((len(nsdsynthetic_img_num),
		rh_betas_all.shape[1]))
	for i, img in enumerate(nsdsynthetic_img_num):
		idx = np.where(masterordering == img)[0]
		lh_betas_test_nsdsynthetic[i] = np.nanmean(lh_betas_all[idx], 0)
		rh_betas_test_nsdsynthetic[i] = np.nanmean(rh_betas_all[idx], 0)

	# Compute the ncsnr
	# When computing the ncsnr on image conditions with different amounts of trials
	# (i.e., different sample sizes), I need to correct for this:
	# https://stats.stackexchange.com/questions/488911/combined-variance-estimate-for-samples-of-varying-sizes
	lh_num_var = np.zeros((lh_betas_all.shape[1]))
	rh_num_var = np.zeros((rh_betas_all.shape[1]))
	den_var = np.zeros((lh_betas_all.shape[1]))
	for i, img in enumerate(nsdsynthetic_img_num):
		idx = np.where(masterordering == img)[0]
		lh_num_var += np.var(lh_betas_all[idx], axis=0, ddof=1) * (len(idx) - 1)
		rh_num_var += np.var(rh_betas_all[idx], axis=0, ddof=1) * (len(idx) - 1)
		den_var += len(idx) - 1
	lh_sigma_noise = np.sqrt(lh_num_var/den_var)
	rh_sigma_noise = np.sqrt(rh_num_var/den_var)
	lh_var_data = np.var(lh_betas_all, axis=0, ddof=1)
	rh_var_data = np.var(rh_betas_all, axis=0, ddof=1)
	lh_sigma_signal = lh_var_data - (lh_sigma_noise ** 2)
	rh_sigma_signal = rh_var_data - (rh_sigma_noise ** 2)
	lh_sigma_signal[lh_sigma_signal<0] = 0
	rh_sigma_signal[rh_sigma_signal<0] = 0
	lh_sigma_signal = np.sqrt(lh_sigma_signal)
	rh_sigma_signal = np.sqrt(rh_sigma_signal)
	lh_ncsnr = lh_sigma_signal / lh_sigma_noise
	rh_ncsnr = rh_sigma_signal / rh_sigma_noise

	# Convert the ncsnr to noise ceiling
	img_reps_2 = 236
	img_reps_4 = 32
	img_reps_8 = 8
	img_reps_10 = 8
	norm_term = (img_reps_2/2 + img_reps_4/4 + img_reps_8/8 + img_reps_10/10) / \
		(img_reps_2 + img_reps_4 + img_reps_8 + img_reps_10)
	lh_noise_ceiling_nsdsynthetic = (lh_ncsnr ** 2) / \
		((lh_ncsnr ** 2) + norm_term)
	rh_noise_ceiling_nsdsynthetic = (rh_ncsnr ** 2) / \
		((rh_ncsnr ** 2) + norm_term)


# =============================================================================
# # Load the encoding model
# =============================================================================
	model = berg.get_encoding_model(args.model_id, subject=sub, device='auto')


# =============================================================================
# Generate the in silico fMRI responses for the 515 NSD-core test images
# =============================================================================
	# Access the NSD-core test images
	sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsd', 'nsd_stimuli.hdf5'), 'r')
	sdataset = sf.get('imgBrick')
	test_img_num = metadata_nsd['test_img_num']
	images = sdataset[test_img_num]
	images = np.transpose(images, (0, 3, 1, 2))

	# Generate the in silico fMRI responses
	lh_betas_test_pred_nsdcore, rh_betas_test_pred_nsdcore = berg.encode(model,
		images)


# =============================================================================
# Generate the in silico fMRI responses for the 286 NSD-synthetic images
# =============================================================================
	# Access the NSD-synthetic images (stimuli)
	stimuli = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsdsynthetic', 'nsdsynthetic_stimuli.hdf5'), 'r').get('imgBrick')[:]
	# Access the NSD-synthetic images (colorstimuli)
	colorstimuli = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli',
		'stimuli', 'nsdsynthetic','nsdsynthetic_colorstimuli_subj0'+str(sub)+
		'.hdf5'), 'r').get('imgBrick')[:]
	images = np.append(stimuli, colorstimuli, 0)
	images = np.transpose(images, (0, 3, 1, 2))

	# Generate the in silico fMRI responses
	lh_betas_test_pred_nsdsynthetic, rh_betas_test_pred_nsdsynthetic = \
		berg.encode(model, images)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
	# Correlate the in vivo and in silico fMRI responses
	lh_correlation_nsdcore = np.zeros(lh_betas_test_nsdcore.shape[1])
	rh_correlation_nsdcore = np.zeros(rh_betas_test_nsdcore.shape[1])
	lh_correlation_nsdsynthetic = np.zeros(lh_betas_test_nsdcore.shape[1])
	rh_correlation_nsdsynthetic = np.zeros(rh_betas_test_nsdcore.shape[1])
	for v in range(lh_betas_test_nsdcore.shape[1]):
		lh_correlation_nsdcore[v] = pearsonr(lh_betas_test_nsdcore[:,v],
			lh_betas_test_pred_nsdcore[:,v])[0]
		rh_correlation_nsdcore[v] = pearsonr(rh_betas_test_nsdcore[:,v],
			rh_betas_test_pred_nsdcore[:,v])[0]
		lh_correlation_nsdsynthetic[v] = pearsonr(
			lh_betas_test_nsdsynthetic[:,v],
			lh_betas_test_pred_nsdsynthetic[:,v])[0]
		rh_correlation_nsdsynthetic[v] = pearsonr(
			rh_betas_test_nsdsynthetic[:,v],
			rh_betas_test_pred_nsdsynthetic[:,v])[0]

	# Set negative correlation scores to zero
	lh_correlation_nsdcore[lh_correlation_nsdcore<0] = 0
	rh_correlation_nsdcore[rh_correlation_nsdcore<0] = 0
	lh_correlation_nsdsynthetic[lh_correlation_nsdsynthetic<0] = 0
	rh_correlation_nsdsynthetic[rh_correlation_nsdsynthetic<0] = 0

	# Turn the correlations into r2 scores
	lh_r2_nsdcore = lh_correlation_nsdcore ** 2
	rh_r2_nsdcore = rh_correlation_nsdcore ** 2
	lh_r2_nsdsynthetic = lh_correlation_nsdsynthetic ** 2
	rh_r2_nsdsynthetic = rh_correlation_nsdsynthetic ** 2

	# Add a very small number to noise ceiling values of 0, otherwise the
	# noise-ceiling-normalized encoding accuracy cannot be calculated (division
	# by 0 is not possible)
	lh_noise_ceiling_nsdcore[lh_noise_ceiling_nsdcore==0] = 1e-14
	rh_noise_ceiling_nsdcore[rh_noise_ceiling_nsdcore==0] = 1e-14
	lh_noise_ceiling_nsdsynthetic[lh_noise_ceiling_nsdsynthetic==0] = 1e-14
	rh_noise_ceiling_nsdsynthetic[rh_noise_ceiling_nsdsynthetic==0] = 1e-14

	# Compute the noise-ceiling-normalized encoding accuracy
	lh_explained_variance_nsdcore = np.divide(lh_r2_nsdcore,
		lh_noise_ceiling_nsdcore) * 100
	rh_explained_variance_nsdcore = np.divide(rh_r2_nsdcore,
		rh_noise_ceiling_nsdcore) * 100
	lh_explained_variance_nsdsynthetic = np.divide(lh_r2_nsdsynthetic,
		lh_noise_ceiling_nsdsynthetic) * 100
	rh_explained_variance_nsdsynthetic = np.divide(rh_r2_nsdsynthetic,
		rh_noise_ceiling_nsdsynthetic) * 100

	# Set the noise-normalized encoding accuracy to 100 for vertices where the
	# the correlation is higher than the noise ceiling, to prevent encoding
	# accuracy values higher than 100%
	lh_explained_variance_nsdcore[lh_explained_variance_nsdcore>100] = 100
	rh_explained_variance_nsdcore[rh_explained_variance_nsdcore>100] = 100
	lh_explained_variance_nsdsynthetic\
		[lh_explained_variance_nsdsynthetic>100] = 100
	rh_explained_variance_nsdsynthetic\
		[rh_explained_variance_nsdsynthetic>100] = 100


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
		'lh_correlation_nsdcore': lh_correlation_nsdcore,
		'rh_correlation_nsdcore': rh_correlation_nsdcore,
		'lh_r2_nsdcore': lh_r2_nsdcore,
		'rh_r2_nsdcore': rh_r2_nsdcore,
		'lh_noise_ceiling_nsdcore': lh_noise_ceiling_nsdcore,
		'rh_noise_ceiling_nsdcore': rh_noise_ceiling_nsdcore,
		'lh_explained_variance_nsdcore': lh_explained_variance_nsdcore,
		'rh_explained_variance_nsdcore': rh_explained_variance_nsdcore,
		'lh_correlation_nsdsynthetic': lh_correlation_nsdsynthetic,
		'rh_correlation_nsdsynthetic': rh_correlation_nsdsynthetic,
		'lh_r2_nsdsynthetic': lh_r2_nsdsynthetic,
		'rh_r2_nsdsynthetic': rh_r2_nsdsynthetic,
		'lh_noise_ceiling_nsdsynthetic': lh_noise_ceiling_nsdsynthetic,
		'rh_noise_ceiling_nsdsynthetic': rh_noise_ceiling_nsdsynthetic,
		'lh_explained_variance_nsdsynthetic': lh_explained_variance_nsdsynthetic,
		'rh_explained_variance_nsdsynthetic': rh_explained_variance_nsdsynthetic
		}
	metadata['encoding_models'] = encoding_models

	# Save the metadata
	model_name = args.model_id.split('-')[-1]
	save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
		'train_dataset-nsd_fsaverage', 'model-'+model_name, 'metadata')
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	file_name = 'metadata_subject-' + format(sub, '02') + '.npy'
	np.save(os.path.join(save_dir, file_name), metadata)
