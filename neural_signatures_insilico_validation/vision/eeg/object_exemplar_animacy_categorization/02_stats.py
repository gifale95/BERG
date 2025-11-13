"""Compute the stats on the in silico EEG responses decoding results.

Parameters
----------
subjects : list
	List of used subjects.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'], occipital/parital ['OP'], or all ['all']
	channels.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
nest_dir : str
	Neural encoding simulation toolkit directory.
imagenet_dir : str
	Directory of the ImageNet image set.
	https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='all', type=str) # ['O', 'P', 'T', 'C', 'F', 'OP', 'all']
parser.add_argument('--n_iter', default=100000, type=int)
#parser.add_argument('--nest_dir', default='/home/ale/aaa_stuff/PhD/projects/neural_encoding_simulation_toolkit', type=str)
#parser.add_argument('--imagenet_dir', default='/home/ale/Downloads/imagenet_val', type=str)
#parser.add_argument('--nest_dir', default='/home/ale/scratch/projects/neural_encoding_simulation_toolkit', type=str)
parser.add_argument('--nest_dir', default='/scratch/giffordale95/projects/neural_encoding_simulation_toolkit', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012/', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding stats | In silico EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
pairwise_decoding_exemplars = []
pairwise_decoding_animacy = []

for sub in args.subjects:

	results_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
		'pairwise_decoding_eeg', 'sub-'+format(sub,'02'), 'channels-'+
		args.channels, 'pairwise_decoding_in_silico_sub-'+format(sub,'02')+
		'_channels-'+args.channels+'.npy')
	results = np.load(results_dir, allow_pickle=True).item()

	# Get the exemplars decoding results
	idx_tril = np.tril_indices(len(results['pairwise_decoding_exemplars']), -1)
	pairwise_decoding_exemplars.append(np.mean(
		results['pairwise_decoding_exemplars'][idx_tril], 0))

	# Get the animacy decoding results
	pairwise_decoding_animacy.append(results['pairwise_decoding_animacy'])

	# EEG metadata
	times = results['times']
	ch_names = results['ch_names']

pairwise_decoding_exemplars = np.asarray(pairwise_decoding_exemplars)
pairwise_decoding_animacy = np.asarray(pairwise_decoding_animacy)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_exemplars = np.zeros((2, len(times)))
ci_animacy = np.zeros((2, len(times)))

exemplars_dist = np.zeros((args.n_iter, len(times)))
animacy_dist = np.zeros((args.n_iter, len(times)))

for i in tqdm(range(args.n_iter)):
	idx = resample(np.arange(len(args.subjects)))
	exemplars_dist[i] = np.mean(pairwise_decoding_exemplars[idx], 0)
	animacy_dist[i] = np.mean(pairwise_decoding_animacy[idx], 0)

ci_exemplars[0] = np.percentile(exemplars_dist, 2.5, axis=0)
ci_exemplars[1] = np.percentile(exemplars_dist, 97.5, axis=0)
ci_animacy[0] = np.percentile(animacy_dist, 2.5, axis=0)
ci_animacy[1] = np.percentile(animacy_dist, 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'pairwise_decoding_exemplars': pairwise_decoding_exemplars,
	'pairwise_decoding_animacy': pairwise_decoding_animacy,
	'ci_exemplars': ci_exemplars,
	'ci_animacy': ci_animacy,
	'times': times,
	'ch_names': ch_names
}

save_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
	'pairwise_decoding_eeg', 'stats', 'channels-'+args.channels)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results_dict)