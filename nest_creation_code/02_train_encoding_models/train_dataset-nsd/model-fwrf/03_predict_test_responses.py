"""Use the trained fwRF ecoding models to synthesize the fMRI responses for the
NSD test images, and compute the encoding accuracy.
The 'lateral' and 'ventral' ROIs were too big to be predicted by a single fwrf
model (not enough GPU RAM), and therefore were split into two partitions.

Parameters
----------
all_subs : list of int
	List with all subject numbers.
all_rois : list of str
	List with all modeled ROIs.
nsd_dir : str
	Directory of the Natural Scenes Dataset (NSD).
	https://naturalscenesdataset.org/
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
import os
import random
import numpy as np
import torch
from torchvision import transforms as trn
from tqdm import tqdm
from PIL import Image
import h5py

from src_new.load_nsd import image_feature_fn
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF

parser = argparse.ArgumentParser()
parser.add_argument('--all_subs', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--all_rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'OFA', 'FFA-1', 'FFA-2', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'PPA',
	'RSC', 'OPA', 'EBA', 'FBA-2', 'early', 'midventral', 'midlateral',
	'midparietal', 'parietal', 'lateral', 'ventral'])
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Predict fMRI responses for the test images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Computing resources
# =============================================================================
# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
	torch.backends.cudnn.enabled=True
	print ('#device:', torch.cuda.device_count())
	print ('device#:', torch.cuda.current_device())
	print ('device name:', torch.cuda.get_device_name(
		torch.cuda.current_device()))

batch_size = 100


# =============================================================================
# Import the NSD images
# =============================================================================
img_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
	'nsd', 'nsd_stimuli.hdf5')
img_dataset = h5py.File(img_dir, 'r').get('imgBrick')


# =============================================================================
# Get the test images IDs
# =============================================================================
data_dir = os.path.join(args.nest_dir, 'model_training_datasets',
	'train_dataset-nsd', 'nsd_betas_sub-'+format(args.all_subs[0],'02')+'_roi-'+
	args.all_rois[0]+'.npy')

data = np.load(data_dir, allow_pickle=True).item()
test_img_num = data['test_img_num']
del data


# =============================================================================
# Loop over subjects and ROIs
# =============================================================================
for sub in tqdm(args.all_subs, leave=False):
	for r in args.all_rois:

		# Random seeds
		seed = (sub * 100) + (np.sum([ord(c) for c in r]))
		seed = int(seed)
		torch.manual_seed(seed)
		random.seed(seed)
		np.random.seed(seed)

		# Generator object for DataLoader random batching
		g_cpu = torch.Generator()
		g_cpu.manual_seed(seed)


# =============================================================================
# Load the trained encoding model weights
# =============================================================================
		# Subjects and ROI directory
		nsd_subjects = [sub]
		if r in ['lateral', 'ventral']:
			subj_roi_dir_1 = 'sub-' + format(sub, '02') + '_roi-' + r + \
				'_split-1'
			subj_roi_dir_2 = 'sub-' + format(sub, '02') + '_roi-' + r + \
				'_split-2'
		else:
			subj_roi_dir = 'sub-' + format(sub, '02') + '_roi-' + r

		# Total model directory
		if r in ['lateral', 'ventral']:
			model_dir_1 = os.path.join(args.nest_dir, 'encoding_models',
				'modality-fmri', 'train_dataset-nsd', 'model-fwrf',
				'encoding_models_weights', 'weights_'+subj_roi_dir_1+'.pt')
			model_dir_2 = os.path.join(args.nest_dir, 'encoding_models',
				'modality-fmri', 'train_dataset-nsd', 'model-fwrf',
				'encoding_models_weights', 'weights_'+subj_roi_dir_2+'.pt')
		else:
			model_dir = os.path.join(args.nest_dir, 'encoding_models',
				'modality-fmri', 'train_dataset-nsd', 'model-fwrf',
				'encoding_models_weights', 'weights_'+subj_roi_dir+'.pt')

		# Load the model
		if r in ['lateral', 'ventral']:
			trained_model_1 = torch.load(model_dir_1,
				map_location=torch.device('cpu'))
			trained_model_2 = torch.load(model_dir_2,
				map_location=torch.device('cpu'))
			stim_mean = trained_model_1['stim_mean']
		else:
			trained_model = torch.load(model_dir,
				map_location=torch.device('cpu'))
			stim_mean = trained_model['stim_mean']


# =============================================================================
# Model instantiation
# =============================================================================
		# Voxel number
		if r in ['lateral', 'ventral']:
			nnv_1 = {}
			nnv_2 = {}
			nnv_1[sub] = len(trained_model_1['best_params']['fwrfs'][sub]['b'])
			nnv_2[sub] = len(trained_model_2['best_params']['fwrfs'][sub]['b'])
		else:
			nnv = {}
			nnv[sub] = len(trained_model['best_params']['fwrfs'][sub]['b'])

		# Load 20 images:
		# (Images × Image channels × Resized image height × Resized image width)
		img_chan = 3
		resize_px = 227
		stim_data = {}
		stim_data[nsd_subjects[0]] = np.zeros((20, img_chan, resize_px, resize_px),
			dtype=np.float32)
		for i in range(20):
			img = img_dataset[i]
			img = Image.fromarray(np.uint8(img))
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((resize_px,resize_px))
			])
			img = transform(img)
			img = np.asarray(img)
			img = img.transpose(2,0,1)
			img = image_feature_fn(img)
			stim_data[i] = img

		# Model functions
		_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)

		def _model_fn(_ext, _con, _x):
			'''model consists of an extractor (_ext) and a connection model (_con)'''
			_y, _fm, _h = _ext(_x)
			return _con(_fm)

		def _pred_fn(_ext, _con, xb):
			return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))

		# Shared encoder model
		if r in ['lateral', 'ventral']:
			shared_model_1 = Encoder(mu=stim_mean, trunk_width=64,
				use_prefilter=1).to(device)
			shared_model_2 = Encoder(mu=stim_mean, trunk_width=64,
				use_prefilter=1).to(device)
			rec, fmaps, h = shared_model_1(torch.from_numpy(
				stim_data[nsd_subjects[0]]).to(device))
		else:
			shared_model = Encoder(mu=stim_mean, trunk_width=64,
				use_prefilter=1).to(device)
			rec, fmaps, h = shared_model(torch.from_numpy(
				stim_data[nsd_subjects[0]]).to(device))

		# Subject specific FWRF models
		if r in ['lateral', 'ventral']:
			subject_fwrfs_1 = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv_1[s], pre_nl=_log_act_fn, \
				post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}
			subject_fwrfs_2 = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv_2[s], pre_nl=_log_act_fn, \
				post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}
		else:
			subject_fwrfs = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv[s], pre_nl=_log_act_fn, \
				post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}


# =============================================================================
# Load the pretrained weights into the model
# =============================================================================
		if r in ['lateral', 'ventral']:
			shared_model_1.load_state_dict(trained_model_1['best_params']['enc'])
			shared_model_2.load_state_dict(trained_model_2['best_params']['enc'])
			for s,sd in subject_fwrfs_1.items():
				sd.load_state_dict(trained_model_1['best_params']['fwrfs'][s])
			for s,sd in subject_fwrfs_2.items():
				sd.load_state_dict(trained_model_2['best_params']['fwrfs'][s])
		else:
			shared_model.load_state_dict(trained_model['best_params']['enc'])
			for s,sd in subject_fwrfs.items():
				sd.load_state_dict(trained_model['best_params']['fwrfs'][s])

		if r in ['lateral', 'ventral']:
			shared_model_1.eval()
			shared_model_2.eval()
			for s,sd in subject_fwrfs_1.items():
				sd.eval()
			for s,sd in subject_fwrfs_2.items():
				sd.eval()
		else:
			shared_model.eval()
			for s,sd in subject_fwrfs.items():
				sd.eval()


# =============================================================================
# Predict the responses for the test images
# =============================================================================
		if r in ['lateral', 'ventral']:
			betas_pred_1 = np.zeros((len(test_img_num), nnv_1[sub]),
				dtype=np.float32)
			betas_pred_2 = np.zeros((len(test_img_num), nnv_2[sub]),
				dtype=np.float32)
		else:
			betas_pred = np.zeros((len(test_img_num), nnv[sub]),
				dtype=np.float32)

		if r in ['lateral', 'ventral']:
			for i in range(len(test_img_num)):
				img = img_dataset[test_img_num[i]]
				img = Image.fromarray(np.uint8(img))
				min_size = min(img.size)
				transform = trn.Compose([
					trn.CenterCrop(min_size),
					trn.Resize((resize_px,resize_px))
				])
				img = transform(img)
				img = np.asarray(img)
				img = img.transpose(2,0,1)
				img = image_feature_fn(img)
				img = np.expand_dims(img, 0)
				sd_1 = subject_fwrfs_1[sub]
				sd_2 = subject_fwrfs_2[sub]
				with torch.no_grad():
					betas_pred_1[i] = subject_pred_pass(_pred_fn,
						shared_model_1, sd_1, img, batch_size)
					betas_pred_2[i] = subject_pred_pass(_pred_fn,
						shared_model_2, sd_2, img, batch_size)
		else:
			with torch.no_grad():
				for s, sd in subject_fwrfs.items():
					for i in range(len(test_img_num)):
						img = img_dataset[test_img_num[i]]
						img = Image.fromarray(np.uint8(img))
						min_size = min(img.size)
						transform = trn.Compose([
							trn.CenterCrop(min_size),
							trn.Resize((resize_px,resize_px))
						])
						img = transform(img)
						img = np.asarray(img)
						img = img.transpose(2,0,1)
						img = image_feature_fn(img)
						img = np.expand_dims(img, 0)
						betas_pred[i] = subject_pred_pass(_pred_fn,
							shared_model, sd, img, batch_size)

		if r in ['lateral', 'ventral']:
			betas_pred = np.append(betas_pred_1, betas_pred_2, 1)


# =============================================================================
# Save the predicted fMRI responses
# =============================================================================
		save_dir = os.path.join(args.nest_dir, 'results', 'test_encoding_models',
			'modality-fmri', 'train_dataset-nsd', 'model-fwrf')

		if os.path.isdir(save_dir) == False:
			os.makedirs(save_dir)

		file_name = 'betas_test_pred_subject-' + str(sub) + '_roi-' + r + '.npy'

		np.save(os.path.join(save_dir, file_name), betas_pred)
		del betas_pred

