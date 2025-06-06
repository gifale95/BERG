"""Prepare and save the Natural Scenes Dataset (NSD)
(https://doi.org/10.1038/s41593-021-00962-x) betas in fsaverage space. The fMRI
betas of each vertex are z-scored at each scan session across trials.

The code additionally saves the ROI masks, the vertices' noise ceiling
signal-to-noise ratio (ncsnr) scores, the stimulus presentation order, and the
training/validation/testing stimulus splits.

Parameters
----------
subject : int
	Number of the used NSD subject.
nsd_dir : str
	Directory of the Natural Scenes Dataset (NSD).
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
from tqdm import tqdm
import pandas as pd
from scipy.stats import zscore
from nsdcode.nsd_mapdata import NSDmapdata # https://github.com/cvnlab/nsdcode
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args = parser.parse_args()

print('>>> Prepare NSD fsaverage betas <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get order and ID of the presented images
# =============================================================================
# Load the experimental design info
nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsd', 'nsd_expdesign.mat'))
# Subtract 1 since the indices start with 1 (and not 0)
masterordering = nsd_expdesign['masterordering'] - 1
subjectim = nsd_expdesign['subjectim'] - 1

# Completed sessions per subject
if args.subject in (1, 2, 5, 7):
	sessions = 40
elif args.subject in (3, 6):
	sessions = 32
elif args.subject in (4, 8):
	sessions = 30

# Image presentation matrix of the selected subject
image_per_session = 750
tot_images = sessions * image_per_session
img_presentation_order = subjectim[args.subject-1,masterordering[0]][:tot_images]


# =============================================================================
# Get the training/validation/testing data splits
# =============================================================================
# The training split consists of fMRI responses for up to 9,000 non-shared
# images (i.e., the images uniquely seen by each subject during the NSD
# experiment).
train_img_num = subjectim[args.subject-1,1000:]
# Only retain image conditions that the subject actually saw at least once
# during the NSD experiment
train_img_num = train_img_num[np.isin(train_img_num, img_presentation_order)]
train_img_num.sort()

# The validation split consist of fMRI responses for up to 485 (out of 1,000)
# shared images (i.e., the 485 images that not all subjects saw for up to three
# times during the NSD experiment).
# The test split consists of fMRI responses for 515 (out of 1,000) shared images
# (i.e., the 515 images that each subject saw for exactly three times during the
# NSD experiment).
min_sess = 30
min_images = min_sess * 750
min_img_presentation = img_presentation_order[:min_images]
test_part = subjectim[args.subject-1,:1000]
# Only retain image conditions that the subject actually saw at least once
# during the NSD experiment
test_part = test_part[np.isin(test_part, img_presentation_order)]
test_part.sort()
val_img_num = []
test_img_num = []
for i in range(len(test_part)):
	if len(np.where(min_img_presentation == test_part[i])[0]) == 3:
		test_img_num.append(test_part[i])
	else:
		val_img_num.append(test_part[i])
val_img_num = np.asarray(val_img_num)
test_img_num = np.asarray(test_img_num)


# =============================================================================
# Load the noise ceiling signal-to-noise ratio (ncsnr)
# =============================================================================
lh_ncsnr = np.squeeze(nib.load(os.path.join(args.nsd_dir, 'nsddata_betas',
	'ppdata', 'subj'+format(args.subject, '02'), 'fsaverage',
	'betas_fithrf_GLMdenoise_RR', 'lh.ncsnr.mgh')).get_fdata())
lh_ncsnr = lh_ncsnr.astype(np.float32)
rh_ncsnr = np.squeeze(nib.load(os.path.join(args.nsd_dir, 'nsddata_betas',
	'ppdata', 'subj'+format(args.subject, '02'), 'fsaverage',
	'betas_fithrf_GLMdenoise_RR', 'rh.ncsnr.mgh')).get_fdata())
rh_ncsnr = rh_ncsnr.astype(np.float32)


# =============================================================================
# Prepare the ROI mask indices
# =============================================================================
# Save the mapping between ROI names and ROI mask values
roi_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
	format(args.subject, '02'), 'label')
roi_map_files = ['prf-visualrois.mgz.ctab', 'floc-bodies.mgz.ctab',
	'floc-faces.mgz.ctab', 'floc-places.mgz.ctab', 'floc-words.mgz.ctab',
	'streams.mgz.ctab']
roi_name_maps = []
for r in roi_map_files:
	roi_map = pd.read_csv(os.path.join(roi_dir, r), delimiter=' ',
		header=None, index_col=0)
	roi_map = roi_map.to_dict()[1]
	roi_name_maps.append(roi_map)

# Map the ROI mask indices from subject native space to fsaverage space
lh_roi_files = ['lh.prf-visualrois.mgz', 'lh.floc-bodies.mgz',
	'lh.floc-faces.mgz', 'lh.floc-places.mgz', 'lh.floc-words.mgz',
	'lh.streams.mgz']
rh_roi_files = ['rh.prf-visualrois.mgz', 'rh.floc-bodies.mgz',
	'rh.floc-faces.mgz', 'rh.floc-places.mgz', 'rh.floc-words.mgz',
	'rh.streams.mgz']
# Initiate NSDmapdata
nsd = NSDmapdata(args.nsd_dir)
lh_fsaverage_rois = {}
rh_fsaverage_rois = {}
for r1 in range(len(lh_roi_files)):
	# Map the ROI masks from subject native to fsaverage space
	lh_fsaverage_roi = np.squeeze(nsd.fit(args.subject, 'lh.white',
		'fsaverage', os.path.join(roi_dir, lh_roi_files[r1])))
	rh_fsaverage_roi = np.squeeze(nsd.fit(args.subject, 'rh.white',
		'fsaverage', os.path.join(roi_dir, rh_roi_files[r1])))
	# Store the ROI masks
	for r2 in roi_name_maps[r1].items():
		if r2[0] != 0:
			lh_fsaverage_rois[r2[1]] = np.where(lh_fsaverage_roi == r2[0])[0]
			rh_fsaverage_rois[r2[1]] = np.where(rh_fsaverage_roi == r2[0])[0]


# =============================================================================
# Save the metadata
# =============================================================================
metadata = {
	'img_presentation_order': img_presentation_order,
	'train_img_num': train_img_num,
	'test_img_num': test_img_num,
	'val_img_num': val_img_num,
	'lh_ncsnr': lh_ncsnr,
	'rh_ncsnr': rh_ncsnr,
	'lh_fsaverage_rois': lh_fsaverage_rois,
	'rh_fsaverage_rois': rh_fsaverage_rois
	}

save_dir = os.path.join(args.berg_dir, 'model_training_datasets',
	'train_dataset-nsd_fsaverage')
file_name = 'metadata_subject-'+str(args.subject)+'.npy'
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

np.save(os.path.join(save_dir, file_name), metadata)


# =============================================================================
# Prepare and save the fMRI betas
# =============================================================================
betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
	format(args.subject, '02'), 'fsaverage', 'betas_fithrf_GLMdenoise_RR')

for s in tqdm(range(sessions)):

	# Load the fMRI betas
	lh_file_name = 'lh.betas_session' + format(s+1, '02') + '.mgh'
	rh_file_name = 'rh.betas_session' + format(s+1, '02') + '.mgh'
	lh_betas_sess = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		lh_file_name)).get_fdata())).astype(np.float32)
	rh_betas_sess = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		rh_file_name)).get_fdata())).astype(np.float32)

	# Z-score the betas of each vertex within each scan session
	lh_betas_sess = zscore(lh_betas_sess, nan_policy='omit')
	rh_betas_sess = zscore(rh_betas_sess, nan_policy='omit')

	# Store the betas
	if s == 0:
		lh_betas = lh_betas_sess
		rh_betas = rh_betas_sess
	else:
		lh_betas = np.append(lh_betas, lh_betas_sess, 0)
		rh_betas = np.append(rh_betas, rh_betas_sess, 0)
	del lh_betas_sess, rh_betas_sess

# Save the fMRI betas
lh_filename = 'lh_betas_subject-' + str(args.subject) + '.h5'
with h5py.File(os.path.join(save_dir, lh_filename), 'w') as f:
	f.create_dataset('betas', data=lh_betas, dtype=np.float32)
rh_filename = 'rh_betas_subject-' + str(args.subject) + '.h5'
with h5py.File(os.path.join(save_dir, rh_filename), 'w') as f:
	f.create_dataset('betas', data=rh_betas, dtype=np.float32)
