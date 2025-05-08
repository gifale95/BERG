"""Train fMRI encoding models on NSD using the feature-weighted receptive field
(fwrf) encoding model (St-Yves & Naselaris, 2018).

This code is an adapted version of the code from the paper:
Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle,
	L.T., Nau, M., Caron, B., Pestilli, F., Charest, I. and Hutchinson,
	J.B., 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience
	and artificial intelligence. Nature neuroscience, 25(1), pp.116-126.

The original code, written by Ghislain St-Yves, can be found here:
https://github.com/styvesg/nsd_gnet8x

The 'lateral' and 'ventral' ROIs were too big to be modeled by a
single fwrf model (not enough GPU RAM), and therefore were split
into two partitions.

Parameters
----------
subjects : str
	Whether to train the model using the data of 'all' or 'single' subjects.
used_nsd_sub : int
	If subjects=='single', the number of the used NSD subject.
roi : str
	Used ROI.
split_roi : int
	Whether to split [1] or not [0] the ROI in two parts.
split : int
	ROI split [1 or 2], if split_roi==1.
trained : int
	If [1] the entire model is trained, if [0] only the fwrf projection heads
	(and not the backbone) are trained.
random_prefilter : int
	If 1 import weights of pre-filter from a pre-trained AlexNet, if 0 use
	randomly intialized weights.
train_prefilter : int
	If 0 the prefilter weights are frozen during training phase 1. If 1 the
	prefilter weights are trained during training phase 1.
	from a pre-trained AlexNet.
use_prefilter : int
	If 0 the prefilter features are not used to encode the brain data, if 1 the
	prefilter features are used to encode the brain data.
train_phases : int
	Whether to run 1, 2 or 3 training phases.
epochs_phase_1 : int
	Number of epochs for the first training phase.
epochs_phase_2 : int
	Number of epochs for the second training phase.
epochs_phase_3 : int
	Number of epochs for the third training phase.
lr : float
	Learning rate.
weight_decay : float
	Weight decay coefficient.
batch_size : int
	Batch size for weight update.
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
import os
import sys
import imp
import random
import numpy as np
import torch
import h5py
from src_new.load_nsd import image_feature_fn, ordering_split
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=str, default='single')
parser.add_argument('--used_nsd_sub', type=int, default=1) # [1, 2, 3, 4, 5, 6, 7, 8]
parser.add_argument('--roi', type=str, default='V1') # ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA-1', 'FFA-2', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'PPA', 'RSC', 'OPA', 'EBA', 'FBA-2', 'early', 'midventral', 'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral']
parser.add_argument('--split_roi', default=0, type=int)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--trained', default=1, type=int)
parser.add_argument('--random_prefilter', type=int, default=1)
parser.add_argument('--train_prefilter', type=int, default=1)
parser.add_argument('--use_prefilter', type=int, default=1)
parser.add_argument('--train_phases', type=int, default=1)
parser.add_argument('--epochs_phase_1', type=int, default=75)
parser.add_argument('--epochs_phase_2', type=int, default=10)
parser.add_argument('--epochs_phase_3', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> Train encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

print ('\ntorch:', torch.__version__)
print ('cuda: ', torch.version.cuda)
print ('cudnn:', torch.backends.cudnn.version())
print ('dtype:', torch.get_default_dtype())


# =============================================================================
# Set random seeds to make results reproducible
# =============================================================================
# Random seeds
if args.subjects == 'single':
	seed = (args.used_nsd_sub * 100) + (np.sum([ord(c) for c in args.roi]))
elif args.subjects == 'all':
	seed = np.sum([ord(c) for c in args.roi])
seed = int(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generator object for DataLoader random batching
g_cpu = torch.Generator()
g_cpu.manual_seed(seed)


# =============================================================================
# Computing resources
# =============================================================================
# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda:1") #cuda

if device == 'cuda':
	torch.backends.cudnn.enabled=True
	print ('#device:', torch.cuda.device_count())
	print ('device#:', torch.cuda.current_device())
	print ('device name:', torch.cuda.get_device_name(
		torch.cuda.current_device()))


# =============================================================================
# Output directories
# =============================================================================
# Subjects and ROI directory
if args.subjects == 'single':
	nsd_subjects = [args.used_nsd_sub]
	subj_roi_dir = 'sub-' + format(args.used_nsd_sub, '02') + '_roi-' + args.roi
elif args.subjects == 'all':
	nsd_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
	subj_roi_dir = 'sub-all_roi-' + args.roi
if args.split_roi == 1:
	subj_roi_dir = subj_roi_dir + '_split-' + str(args.split)

# Hyperparameters directory
# model_hyperpar = 'lr-{:.0e}'.format(args.lr) + '__bs-' + \
#         format(args.batch_size,'03')
# hyperparam_dir = os.path.join('random_prefilter-'+str(args.random_prefilter),
#         'train_prefilter-'+str(args.train_prefilter), 'use_prefilter-'+
#         str(args.use_prefilter), model_hyperpar)

# Create models output directory
model_output_dir = os.path.join(args.nest_dir, 'encoding_models',
	'modality-fmri', 'training_dataset-nsd', 'model-fwrf',
	'encoding_models_weights')
if not os.path.exists(model_output_dir):
	os.makedirs(model_output_dir)

# TensorBoard output directory
tensorboard_parent = os.path.join(args.nest_dir, 'results',
	'train_encoding_models', 'modality-fmri', 'training_dataset-nsd',
	'model-fwrf', 'tensorboard', subj_roi_dir)


# =============================================================================
# Load the stimulus images
# =============================================================================
print ('\nStimulus Images:')
stim_data = {}
for s in nsd_subjects:
	image_data_set = h5py.File(os.path.join(args.nest_dir,
		'model_training_datasets', 'train_dataset-nsd',
		'nsd_images_sub-0%d_px-227.h5py'%s), 'r')
	stim_data[s] = image_feature_fn(np.copy(image_data_set['stimuli']))
	image_data_set.close()
	print ('--------  subject %d  -------' % s)
	print ('block size:', stim_data[s].shape, ', dtype:', stim_data[s].dtype,
		', value range:', np.min(stim_data[s][0]), np.max(stim_data[s][0]))

trn_stim_mean = sum([np.mean(stim_data[s], axis=(0,2,3), keepdims=True) for s in nsd_subjects]) / len(nsd_subjects)


# =============================================================================
# Load the fMRI betas
# =============================================================================
print ('\nfMRI Betas:')
voxel_data = {}
betas_info ={}

for s in nsd_subjects:
	print ('--------  subject %d  -------' % s)
	betas_dir = os.path.join(args.nest_dir, 'model_training_datasets',
		'train_dataset-nsd', 'nsd_betas_sub-'+format(s,'02')+'_roi-'+args.roi+
		'.npy')
	betas_dict = np.load(betas_dir, allow_pickle=True).item()
	voxel_data[s] = betas_dict['betas']
	if args.split_roi == 1:
		idx = int(np.ceil(voxel_data[s].shape[1] / 2))
		if args.split == 1:
			voxel_data[s] = voxel_data[s][:,:idx]
		elif args.split == 2:
			voxel_data[s] = voxel_data[s][:,idx:]
	del betas_dict['betas']
	betas_info[s] = betas_dict
	print ('Betas shape:', voxel_data[s].shape, ', dtype:',
		voxel_data[s].dtype, ', value range:', np.min(voxel_data[s][0]),
		np.max(voxel_data[s][0]))
	del betas_dict


# =============================================================================
# Split the dataset into training, holdout and validation partitions
# =============================================================================
# Use the shared images that all subjects viewed for 3 times as validation
# images, and the remaining shared images as holdout images.
trn_stim_ordering, trn_voxel_data, hld_stim_ordering, hld_voxel_data, \
	val_stim_ordering, val_voxel_data = {}, {}, {}, {}, {}, {}
nnv = {}

# Split the dataset into training, holdout and validation partitions
for s in nsd_subjects:
	nnv[s] = voxel_data[s].shape[1]
	ordering = betas_info[s]['masterordering'].flatten()
	val_idx = betas_info[s]['test_img_idx']
	trn_stim_ordering[s], trn_voxel_data[s], \
		hld_stim_ordering[s], hld_voxel_data[s], \
		val_stim_ordering[s], val_voxel_data[s] = \
		ordering_split(voxel_data[s], ordering, val_idx, combine_trial=False)
del voxel_data


# =============================================================================
# Model instantiation
# =============================================================================
_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)

# Shared encoder model
shared_model = Encoder(mu=trn_stim_mean, trunk_width=64,
	use_prefilter=args.use_prefilter).to(device)
rec, fmaps, h = shared_model(torch.from_numpy(
	stim_data[nsd_subjects[0]][:20]).to(device))

# Subject specific FWRF models
subject_fwrfs = {s: Torch_LayerwiseFWRF(fmaps, nv=nnv[s], pre_nl=_log_act_fn, \
	post_nl=_log_act_fn, dtype=np.float32).to(device) for s in nsd_subjects}

# Print parameters of FWRF models
for s,sp in subject_fwrfs.items():
	print ("\n--------- subject %d ----------"%s)
	for p in sp.parameters():
		print ("block size %-16s" % (list(p.size())))

# Print parameters of shared encoder model
param_count = 0
for w in shared_model.enc.parameters():
	param_count += np.prod(tuple(w.size()))
print ('')
print (param_count, "shared params")
total_nv = 0
for s,sp in subject_fwrfs.items():
	for p in sp.parameters():
		param_count += np.prod(tuple(p.size()))
	total_nv += nnv[s]
print (param_count // total_nv, "approx params per voxels")


# =============================================================================
# Load prefilter weights from a trained AlexNet
# =============================================================================
if args.random_prefilter == 0:
	try:
		from torch.hub import load_state_dict_from_url
	except ImportError:
		from torch.utils.model_zoo import load_url as load_state_dict_from_url

	# Load the AlexNet weights
	state_dict = load_state_dict_from_url(
		'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
		progress=True)

	# Rename dictionary keys to match new breakdown
	pre_state_dict = {}
	pre_state_dict['conv1.0.weight'] = state_dict.pop('features.0.weight')
	pre_state_dict['conv1.0.bias'] = state_dict.pop('features.0.bias')
	pre_state_dict['conv2.0.weight'] = state_dict.pop('features.3.weight')
	pre_state_dict['conv2.0.bias'] = state_dict.pop('features.3.bias')

	# Add the AlexNet weights to the prefilter network
	shared_model.pre.load_state_dict(pre_state_dict)

# If "args.trained == 0" do NOT train the shared model layers
if args.trained == 0:
	for param in shared_model.parameters():
		param.requires_grad = False


# =============================================================================
# Loss functions, etc.
# =============================================================================
fpX = np.float32

def _model_fn(_ext, _con, _x):
	'''model consists of an extractor (_ext) and a connection model (_con)'''
	_y, _fm, _h = _ext(_x)
	return _con(_fm)

def _smoothness_loss_fn(_rf, n):
	delta_x = torch.sum(torch.pow(torch.abs(_rf[:,1:] - _rf[:,:-1]), n))
	delta_y = torch.sum(torch.pow(torch.abs(_rf[:,:,1:] - _rf[:,:,:-1]), n))
	return delta_x + delta_y

def vox_loss_fn(r, v, nu=0.5, delta=1.):
	#err = torch.sum(huber(r, v, delta), dim=0)
	err = torch.sum((r - v)**2, dim=0)
	# Squared correlation coefficient with 'leak'
	cr = r - torch.mean(r, dim=0, keepdim=True)
	cv = v - torch.mean(v, dim=0, keepdim=True)
	wgt = torch.clamp(torch.pow(torch.mean(cr*cv, dim=0), 2) / \
		((torch.mean(cr**2, dim=0)) * (torch.mean(cv**2, dim=0)) + 1e-6), \
		min=nu, max=1).detach()
	weighted_err = wgt * err # error per voxel
	loss = torch.sum(weighted_err) / torch.mean(wgt)
	return err, loss

def _loss_fn(_ext, _con, _x, _v):
	_r = _model_fn(_ext, _con, _x)
	#_err = T.sum((_r - _v)**2, dim=0)
	#_loss = T.sum(_err)
	_err, _loss = vox_loss_fn(_r, _v, nu=0.1, delta=.5)
	_loss += fpX(1e-1) * torch.sum(torch.abs(_con.w))
	return _err, _loss

def _training_fn(_ext, _con, _opts, xb, yb):
	for _opt in _opts:
		_opt.zero_grad()
		_err, _loss = _loss_fn(_ext, _con, torch.from_numpy(xb).to(device),
			torch.from_numpy(yb).to(device))
		_loss.backward()
		_opt.step()
	return _err

def _holdout_fn(_ext, _con, xb, yb):
	# print (xb.shape, yb.shape)
	_err,_ = _loss_fn(_ext, _con, torch.from_numpy(xb).to(device),
		torch.from_numpy(yb).to(device))
	return _err

def _pred_fn(_ext, _con, xb):
	return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))

def print_grads(_ext, _con, _params, _opt, xb, yb):
	_opt.zero_grad()
	_err, _loss = _loss_fn(_ext, _con, torch.from_numpy(xb).to(device),
		torch.from_numpy(yb).to(device))  
	_loss.backward()
	for p in _params:
		prg = get_value(p.grad)
		print ("%-16s : value=%f, grad=%f" % (list(p.size()),
			np.mean(np.abs(get_value(p))), np.mean(np.abs(prg))))
	print ('--------------------------------------')
	sys.stdout.flush()


# =============================================================================
# Training phase 1
# =============================================================================
import src_new.torch_joint_training_unpacked_sequences as aaa
imp.reload(aaa)

# Shared model optimizer
if args.train_prefilter == 0:
	optimizer_net = torch.optim.Adam([
		{'params': shared_model.enc.parameters()},
		], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
		weight_decay=args.weight_decay)
elif args.train_prefilter == 1:
	optimizer_net = torch.optim.Adam([
		{'params': shared_model.pre.parameters()},
		{'params': shared_model.enc.parameters()},
		], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
		weight_decay=args.weight_decay)
# FWRF model optimizers
subject_optimizer = {s: torch.optim.Adam([
	{'params': sp.parameters()}
	], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
	weight_decay=args.weight_decay) for s,sp in subject_fwrfs.items()}
# All optimizers
if args.trained == 0:
	# If "args.trained == 0" do NOT train the shared model layers
	subject_opts = {s: [subject_optimizer[s]] for s in subject_optimizer.keys()}
elif args.trained == 1:
	# If "args.trained == 1" train all model layers
	subject_opts = {s: [optimizer_net, subject_optimizer[s]] \
		for s in subject_optimizer.keys()}

# TensorBoard
tensorboard_child = os.path.join(tensorboard_parent)
if not os.path.exists(tensorboard_child):
	os.makedirs(tensorboard_child)
writer = SummaryWriter(tensorboard_child)

# Model training
best_params, final_params, hold_cc_hist, hold_hist, trn_hist, best_epoch, \
	best_joint_cc_score = learn_params_(
	writer,
	_training_fn,
	_holdout_fn,
	_pred_fn,
	shared_model,
	subject_fwrfs,
	subject_opts,
	stim_data,
	trn_voxel_data,
	trn_stim_ordering,
	hld_voxel_data,
	hld_stim_ordering,
	num_epochs=args.epochs_phase_1,
	batch_size=args.batch_size,
	holdout_frac=0.1,
	masks=None,
	randomize=True)

# Model testing
val_voxel = {s: val_voxel_data[s] for s in val_voxel_data.keys()}
shared_model.load_state_dict(best_params['enc'])
shared_model.eval()
for s,sd in subject_fwrfs.items():
	sd.load_state_dict(best_params['fwrfs'][s])
	sd.eval()
subject_val_cc = validation_(_pred_fn, shared_model, subject_fwrfs, stim_data,
	val_voxel, val_stim_ordering, args.batch_size)
joined_val_cc = np.concatenate(list(subject_val_cc.values()), axis=0)
print ("\nBest joint score = %.3f"%best_joint_cc_score)
print ("Best joint val cc = %.3f"% np.median(joined_val_cc))
for s,v in subject_val_cc.items():
	print ("Subject %s: val cc = %.3f"%(s, np.median(v)))

out_dir = os.path.join(model_output_dir, 'weights_'+subj_roi_dir+'.pt')

# Save model parameters
torch.save({
	'args.': args,
	'best_params': best_params,
	'final_params': final_params,
	'trn_hist': trn_hist,
	'hold_hist': hold_hist,
	'hold_cc_hist': hold_cc_hist,
	'best_epoch': best_epoch,
	'best_joint_cc_score': best_joint_cc_score,
	'subject_val_cc': subject_val_cc,
	'stim_mean': trn_stim_mean,
	'betas_info': betas_info
	}, out_dir)


# =============================================================================
# Training phase 2
# =============================================================================
if args.train_phases > 1:

	import src_new.torch_joint_training_unpacked_sequences as aaa
	imp.reload(aaa)

	# Set the model weights to the best model
	shared_model.load_state_dict(best_params['enc'])
	for s,sd in subject_fwrfs.items():
		sd.load_state_dict(best_params['fwrfs'][s])

	# Shared model optimizer
	optimizer_net = torch.optim.Adam([
		{'params': shared_model.pre.parameters()},
		{'params': shared_model.enc.parameters()},
		], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
		weight_decay=args.weight_decay)
	# All optimizers
	subject_opts = {s: [optimizer_net] for s in subject_optimizer.keys()}

	# TensorBoard
	tensorboard_child = os.path.join(tensorboard_parent,
		'tensorboard_train_phase_02')
	if not os.path.exists(tensorboard_child):
		os.makedirs(tensorboard_child)
	writer = SummaryWriter(tensorboard_child)

	# Model training
	best_params, final_params, hold_cc_hist, hold_hist, trn_hist, best_epoch, \
		best_joint_cc_score = learn_params_(
		writer,
		_training_fn,
		_holdout_fn,
		_pred_fn,
		shared_model,
		subject_fwrfs,
		subject_opts,
		stim_data,
		trn_voxel_data,
		trn_stim_ordering,
		hld_voxel_data,
		hld_stim_ordering,
		num_epochs=args.epochs_phase_2,
		batch_size=args.batch_size,
		holdout_frac=0.1,
		masks=None,
		randomize=True)

	# Model testing
	val_voxel = {s: val_voxel_data[s] for s in val_voxel_data.keys()}
	shared_model.load_state_dict(best_params['enc'])
	shared_model.eval()
	for s,sd in subject_fwrfs.items():
		sd.load_state_dict(best_params['fwrfs'][s])
		sd.eval()
	subject_val_cc = validation_(_pred_fn, shared_model, subject_fwrfs, stim_data,
		val_voxel, val_stim_ordering, args.batch_size)
	joined_val_cc = np.concatenate(list(subject_val_cc.values()), axis=0)
	print ("\nBest joint score = %.3f"%best_joint_cc_score)
	print ("Best joint val cc = %.3f"% np.median(joined_val_cc))
	for s,v in subject_val_cc.items():
		print ("Subject %s: val cc = %.3f"%(s, np.median(v)))

	out_dir = os.path.join(model_output_dir, 'weights_'+subj_roi_dir+
		'train_phase-02.pt')

	# Save model parameters
	torch.save({
		'args.': args,
		'best_params': best_params,
		'final_params': final_params,
		'trn_hist': trn_hist,
		'hold_hist': hold_hist,
		'hold_cc_hist': hold_cc_hist,
		'best_epoch': best_epoch,
		'best_joint_cc_score': best_joint_cc_score,
		'subject_val_cc': subject_val_cc,
		'stim_mean': trn_stim_mean,
		'betas_info': betas_info
		}, out_dir)


# =============================================================================
# Training phase 3
# =============================================================================
if args.train_phases > 2:

	import src_new.torch_joint_training_unpacked_sequences as aaa
	imp.reload(aaa)

	# Set the model weights to the best model
	shared_model.load_state_dict(best_params['enc'])
	for s,sd in subject_fwrfs.items():
		sd.load_state_dict(best_params['fwrfs'][s])

	# FWRF model optimizers
	subject_optimizer = {s: torch.optim.Adam([
		{'params': sp.parameters()}
		], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
		weight_decay=args.weight_decay) for s,sp in subject_fwrfs.items()}
	# All optimizers
	subject_opts = {s: [subject_optimizer[s]] for s in nsd_subjects}

	# TensorBoard
	tensorboard_child = os.path.join(tensorboard_parent,
		'tensorboard_train_phase_03')
	if not os.path.exists(tensorboard_child):
		os.makedirs(tensorboard_child)
	writer = SummaryWriter(tensorboard_child)

	# Model training
	best_params, final_params, hold_cc_hist, hold_hist, trn_hist, best_epoch, \
		best_joint_cc_score = learn_params_(
		writer,
		_training_fn,
		_holdout_fn,
		_pred_fn,
		shared_model,
		subject_fwrfs,
		subject_opts,
		stim_data,
		trn_voxel_data,
		trn_stim_ordering,
		hld_voxel_data,
		hld_stim_ordering,
		num_epochs=args.epochs_phase_3,
		batch_size=args.batch_size,
		holdout_frac=0.1,
		masks=None,
		randomize=True)

	# Model testing
	val_voxel = {s: val_voxel_data[s] for s in val_voxel_data.keys()}
	shared_model.load_state_dict(best_params['enc'])
	shared_model.eval()
	for s,sd in subject_fwrfs.items():
		sd.load_state_dict(best_params['fwrfs'][s])
		sd.eval()
	subject_val_cc = validation_(_pred_fn, shared_model, subject_fwrfs, stim_data,
		val_voxel, val_stim_ordering, args.batch_size)
	joined_val_cc = np.concatenate(list(subject_val_cc.values()), axis=0)
	print ("\nBest joint score = %.3f"%best_joint_cc_score)
	print ("Best joint val cc = %.3f"% np.median(joined_val_cc))
	for s,v in subject_val_cc.items():
		print ("Subject %s: val cc = %.3f"%(s, np.median(v)))

	out_dir = os.path.join(model_output_dir, 'weights_'+subj_roi_dir+
		'train_phase-03.pt')

	# Save model parameters
	torch.save({
		'args.': args,
		'best_params': best_params,
		'final_params': final_params,
		'trn_hist': trn_hist,
		'hold_hist': hold_hist,
		'hold_cc_hist': hold_cc_hist,
		'best_epoch': best_epoch,
		'best_joint_cc_score': best_joint_cc_score,
		'subject_val_cc': subject_val_cc,
		'stim_mean': trn_stim_mean,
		'betas_info': betas_info
		}, out_dir)
