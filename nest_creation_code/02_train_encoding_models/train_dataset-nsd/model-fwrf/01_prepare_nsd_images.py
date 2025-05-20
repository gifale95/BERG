"""Resize and sort the stimulus images of all NSD subjects for later fwRF model
training.

Parameters
----------
used_nsd_subjects : list of int
	List containing ID numbers of the used NSD subjects.
resize_px : int
	Pixel resolution of the resized images.
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
from scipy.io import loadmat
import h5py
from tqdm import tqdm
from PIL import Image
from src_new.file_utility import save_stuff

parser = argparse.ArgumentParser()
parser.add_argument('--used_nsd_subjects', type=list,
	default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--resize_px', type=int, default=227)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Prepare NSD images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the NSD experimental design info and access the stimuli images
# =============================================================================
# Load the NSD experimental design info
nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsd', 'nsd_expdesign.mat'))
# Zero-indexed ordering of indices (matlab-like to python-like)
subjectim = nsd_expdesign['subjectim'] - 1

# Access the '.hdf5' NSD images file
image_data_set = h5py.File(os.path.join(args.nsd_dir,'nsddata_stimuli',
	'stimuli', 'nsd', 'nsd_stimuli.hdf5'), 'r')


# =============================================================================
# Reformat and resize the stimuli images
# =============================================================================
# Resize the NSD stimuli images to 227×227 pixels, and save the images of each
# NSD subject in separate '.hdf5' files.

save_dir = os.path.join(args.nest_dir, 'model_training_datasets',
	'train_dataset-nsd')

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

img_chan = 3
for s in args.used_nsd_subjects:
	# Get the indices of 10,000 subject-specific NSD images
	img_idxs = subjectim[s-1]
	# Subject images array of shape:
	# (Images × Image channels × Resized image height × Resized image width)
	image_data = np.zeros((len(img_idxs),img_chan,args.resize_px,
		args.resize_px), dtype=np.int16)
	# Load and resize the images
	for i, img_idx in enumerate(tqdm(img_idxs, desc='Subj '+format(s))):
		img = np.copy(image_data_set['imgBrick'][img_idx])
		img = np.asarray(Image.fromarray(img).resize((
			args.resize_px,args.resize_px), resample=Image.BILINEAR))
		image_data[i] = img.transpose(2,0,1)
	# Save the images of each subject in separate '.hdf5' files
	output_dir = os.path.join(save_dir,'nsd_images_sub-'+
		format(s,'02')+'_px-'+format(args.resize_px))
	save_stuff(output_dir, {'stimuli': image_data})
image_data_set.close()
