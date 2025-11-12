"""Plot the tripartitie organization effect (Konkle & Caramazza, 2013) on in
silico fMRI responses.

Parameters
----------
subjects : list of int
	Number of all the used NSD subject.
image_background : str
	Whether the images have a 'natural' or 'artificial' image background.
nest_dir : str
	Neural encoding simulation toolkit directory.
nsd_dir : str
	Directory of the Natural Scenes Dataset.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--image_background', type=str, default='artificial')
parser.add_argument('--nest_dir', default='/home/ale/aaa_stuff/PhD/projects/neural_encoding_simulation_toolkit', type=str)
parser.add_argument('--nsd_dir', default='/home/ale/scratch/datasets/natural-scenes-dataset/', type=str)
#parser.add_argument('--nest_dir', default='/scratch/giffordale95/projects/neural_encoding_simulation_toolkit', type=str)
args = parser.parse_args()


# =============================================================================
# Plot parameters
# =============================================================================
subject = 'fsaverage_nsd'
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
custom_cmap = mcolors.ListedColormap([(143/255, 25/255, 250/255),
	(43/255, 141/255, 248/255), (243/255, 85/255, 20/255)])


# =============================================================================
# Load the tripartite fMRI responses of all subjects
# =============================================================================
lh_animals = []
rh_animals = []
lh_big_objects = []
rh_big_objects = []
lh_small_objects = []
rh_small_objects = []
lh_ncsnr = []
rh_ncsnr = []

data_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
	'fmri_tripartite_organization', 'single_subject_results')

for sub in args.subjects:
	data_file = 'fmri_tripartite_organization_image_background-' + \
		args.image_background + '_sub-' + format(sub,'02') + '.npy'
	data = np.load(os.path.join(data_dir, data_file), allow_pickle=True).item()
	lh_animals.append(data['lh_animals'])
	rh_animals.append(data['rh_animals'])
	lh_big_objects.append(data['lh_big_objects'])
	rh_big_objects.append(data['rh_big_objects'])
	lh_small_objects.append(data['lh_small_objects'])
	rh_small_objects.append(data['rh_small_objects'])
	lh_ncsnr.append(data['lh_ncsnr'])
	rh_ncsnr.append(data['rh_ncsnr'])

lh_animals = np.asarray(lh_animals)
rh_animals = np.asarray(rh_animals)
lh_big_objects = np.asarray(lh_big_objects)
rh_big_objects = np.asarray(rh_big_objects)
lh_small_objects = np.asarray(lh_small_objects)
rh_small_objects = np.asarray(rh_small_objects)
lh_ncsnr = np.asarray(lh_ncsnr)
rh_ncsnr = np.asarray(rh_ncsnr)


# =============================================================================
# Compute the overlap of different ROIs with the categorical zones
# =============================================================================
FFA_animal = np.zeros((len(args.subjects)))
OFA_animal = np.zeros((len(args.subjects)))
EBA_animal = np.zeros((len(args.subjects)))
FBA_animal = np.zeros((len(args.subjects)))

PPA_big_objects = np.zeros((len(args.subjects)))
OPA_big_objects = np.zeros((len(args.subjects)))
RSC_big_objects = np.zeros((len(args.subjects)))

for s, sub in enumerate(args.subjects):
	# Load the ROI indices
	metadata_dir = os.path.join(args.nest_dir, 'model_training_datasets',
		'train_dataset-nsd_fsaverage', 'metadata_subject-'+str(sub)+'.npy')
	metadata = np.load(metadata_dir, allow_pickle=True).item()

	# FFA
	count = 0
	# LH
	lh_idx = np.append(metadata['lh_fsaverage_rois']['FFA-1'],
		metadata['lh_fsaverage_rois']['FFA-2'])
	lh_idx.sort()
	for v in lh_idx:
		if lh_animals[s,v] > lh_small_objects[s,v] and lh_animals[s,v] > lh_big_objects[s,v]:
			count += 1
	# RH
	rh_idx = np.append(metadata['rh_fsaverage_rois']['FFA-1'],
		metadata['rh_fsaverage_rois']['FFA-2'])
	rh_idx.sort()
	for v in rh_idx:
		if rh_animals[s,v] > rh_small_objects[s,v] and rh_animals[s,v] > rh_big_objects[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	FFA_animal[s] = count / tot_vertices

	# OFA
	count = 0
	# LH
	lh_idx = metadata['lh_fsaverage_rois']['OFA']
	for v in lh_idx:
		if lh_animals[s,v] > lh_small_objects[s,v] and lh_animals[s,v] > lh_big_objects[s,v]:
			count += 1
	# RH
	rh_idx = metadata['rh_fsaverage_rois']['OFA']
	for v in rh_idx:
		if rh_animals[s,v] > rh_small_objects[s,v] and rh_animals[s,v] > rh_big_objects[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	OFA_animal[s] = count / tot_vertices

	# EBA
	count = 0
	# LH
	lh_idx = metadata['lh_fsaverage_rois']['EBA']
	for v in lh_idx:
		if lh_animals[s,v] > lh_small_objects[s,v] and lh_animals[s,v] > lh_big_objects[s,v]:
			count += 1
	# RH
	rh_idx = metadata['rh_fsaverage_rois']['EBA']
	for v in rh_idx:
		if rh_animals[s,v] > rh_small_objects[s,v] and rh_animals[s,v] > rh_big_objects[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	EBA_animal[s] = count / tot_vertices

	# FBA
	count = 0
	# LH
	lh_idx = np.append(metadata['lh_fsaverage_rois']['FBA-1'],
		metadata['lh_fsaverage_rois']['FBA-2'])
	lh_idx.sort()
	for v in lh_idx:
		if lh_animals[s,v] > lh_small_objects[s,v] and lh_animals[s,v] > lh_big_objects[s,v]:
			count += 1
	# RH
	rh_idx = np.append(metadata['rh_fsaverage_rois']['FBA-1'],
		metadata['rh_fsaverage_rois']['FBA-2'])
	rh_idx.sort()
	for v in rh_idx:
		if rh_animals[s,v] > rh_small_objects[s,v] and rh_animals[s,v] > rh_big_objects[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	FBA_animal[s] = count / tot_vertices

	# PPA
	count = 0
	# LH
	lh_idx = metadata['lh_fsaverage_rois']['PPA']
	for v in lh_idx:
		if lh_big_objects[s,v] > lh_small_objects[s,v] and lh_big_objects[s,v] > lh_animals[s,v]:
			count += 1
	# RH
	rh_idx = metadata['rh_fsaverage_rois']['PPA']
	for v in rh_idx:
		if rh_big_objects[s,v] > rh_small_objects[s,v] and rh_big_objects[s,v] > rh_animals[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	PPA_big_objects[s] = count / tot_vertices

	# OPA
	count = 0
	# LH
	lh_idx = metadata['lh_fsaverage_rois']['OPA']
	for v in lh_idx:
		if lh_big_objects[s,v] > lh_small_objects[s,v] and lh_big_objects[s,v] > lh_animals[s,v]:
			count += 1
	# RH
	rh_idx = metadata['rh_fsaverage_rois']['OPA']
	for v in rh_idx:
		if rh_big_objects[s,v] > rh_small_objects[s,v] and rh_big_objects[s,v] > rh_animals[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	OPA_big_objects[s] = count / tot_vertices

	# RSC
	count = 0
	# LH
	lh_idx = metadata['lh_fsaverage_rois']['RSC']
	for v in lh_idx:
		if lh_big_objects[s,v] > lh_small_objects[s,v] and lh_big_objects[s,v] > lh_animals[s,v]:
			count += 1
	# RH
	rh_idx = metadata['rh_fsaverage_rois']['RSC']
	for v in rh_idx:
		if rh_big_objects[s,v] > rh_small_objects[s,v] and rh_big_objects[s,v] > rh_animals[s,v]:
			count += 1
	# Store results
	tot_vertices = len(lh_idx) + len(rh_idx)
	RSC_big_objects[s] = count / tot_vertices

# Average across subjects
FFA_animal_mean = np.mean(FFA_animal)
OFA_animal_mean = np.mean(OFA_animal)
EBA_animal_mean = np.mean(EBA_animal)
FBA_animal_mean = np.mean(FBA_animal)
PPA_big_objects_mean = np.mean(PPA_big_objects)
OPA_big_objects_mean = np.mean(OPA_big_objects)
RSC_big_objects_mean = np.mean(RSC_big_objects)


# =============================================================================
# Tripartite organization analysis (between-subjects)
# =============================================================================
# Perform the tripartite organization analysis on the subject-average fMRI
# responses across subjects.

# For each condition, subtract the mean of the other two conditions
lh_animals_b = np.mean(lh_animals, 0) - \
	((np.mean(lh_big_objects, 0) + np.mean(lh_small_objects, 0)) / 2)
rh_animals_b = np.mean(rh_animals, 0) - \
	((np.mean(rh_big_objects, 0) + np.mean(rh_small_objects, 0)) / 2)
lh_big_objects_b = np.mean(lh_big_objects, 0) - \
	((np.mean(lh_animals, 0) + np.mean(lh_small_objects, 0)) / 2)
rh_big_objects_b = np.mean(rh_big_objects, 0) - \
	((np.mean(rh_animals, 0) + np.mean(rh_small_objects, 0)) / 2)
lh_small_objects_b = np.mean(lh_small_objects, 0) - \
	((np.mean(lh_animals, 0) + np.mean(lh_big_objects, 0)) / 2)
rh_small_objects_b = np.mean(rh_small_objects, 0) - \
	((np.mean(rh_animals, 0) + np.mean(rh_big_objects, 0)) / 2)
lh_ncsnr_b = np.mean(lh_ncsnr, 0)
rh_ncsnr_b = np.mean(rh_ncsnr, 0)

# Data without mean subtraction
# lh_animals_b = np.mean(lh_animals, 0)
# rh_animals_b = np.mean(rh_animals, 0)
# lh_big_objects_b = np.mean(lh_big_objects, 0)
# rh_big_objects_b = np.mean(rh_big_objects, 0)
# lh_small_objects_b = np.mean(lh_small_objects, 0)
# rh_small_objects_b = np.mean(rh_small_objects, 0)

# Tripartite organization analysis
lh_tripartite_organization = np.zeros((len(lh_animals_b)), dtype=np.float32)
rh_tripartite_organization = np.zeros((len(rh_animals_b)), dtype=np.float32)
threshold_resp = -10 # 0.025 # threshold based on an univariate response arbitrary value
threshold_ncsnr = 0.2
for v in range(len(lh_tripartite_organization)):
	# Left hemisphere
	lh_data = [lh_animals_b[v], lh_big_objects_b[v], lh_small_objects_b[v]]
	lh_tripartite_organization[v] = np.argsort(lh_data)[-1]
	if max(lh_data) < threshold_resp: # Threshold with univariate response magnitude
		lh_tripartite_organization[v] = np.nan
	if lh_ncsnr_b[v] < threshold_ncsnr: # Threshold with ncsnr
		lh_tripartite_organization[v] = np.nan
	# Right hemisphere
	rh_data = [rh_animals_b[v], rh_big_objects_b[v], rh_small_objects_b[v]]
	rh_tripartite_organization[v] = np.argsort(rh_data)[-1]
	if max(rh_data) < threshold_resp: # Threshold with univariate response magnitude
		rh_tripartite_organization[v] = np.nan
	if rh_ncsnr_b[v] < threshold_ncsnr: # Threshold with ncsnr
		rh_tripartite_organization[v] = np.nan

# Set early/intermediate stream vertices to nan
metadata_dir = os.path.join(args.nest_dir, 'model_training_datasets',
	'train_dataset-nsd_fsaverage', 'metadata_subject-1.npy')
metadata = np.load(metadata_dir, allow_pickle=True).item()
lh_early = metadata['lh_fsaverage_rois']['early']
rh_early = metadata['rh_fsaverage_rois']['early']
lh_tripartite_organization[lh_early] = np.nan
rh_tripartite_organization[rh_early] = np.nan

# Plot the results
tripartite_organization = np.append(lh_tripartite_organization,
	rh_tripartite_organization)
vertex_data = cortex.Vertex(tripartite_organization, subject,
	cmap=custom_cmap, vmin=0, vmax=2, with_colorbar=False)
fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	with_rois=True,
	#roi_list=['FFA', 'PPA', 'OFA'],
	roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
	linewidth=5,
	linecolor=(1, 1, 1),
	with_labels=False,
	labelsize=20,
	curvature_brightness=0.5,
	with_colorbar=False
	)

# Save the figure
file_name = 'fmri_tripartite_effect_between_subjects_background-artificial.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')


# =============================================================================
# Tripartite organization analysis (within-subjects)
# =============================================================================
# Perform the tripartite organization within subjects, and then assign each
# vertex based on the tripartitie organization of all subjects (only when there
# is an agreement of at least N subjects, so that this will also serve as a
# threshold).

# Tripartite organization analysis
# lh_tripartite_organization_subjects = np.zeros((len(args.subjects),
# 	lh_animals.shape[1]), dtype=np.float32)
# rh_tripartite_organization_subjects = np.zeros((len(args.subjects),
# 	rh_animals.shape[1]), dtype=np.float32)
# threshold_ncsnr = 0.2
# for s in tqdm(range(len(args.subjects))):
# 	for v in range(lh_tripartite_organization_subjects.shape[1]):
# 		# Left hemisphere
# 		lh_data = [lh_animals[s,v], lh_big_objects[s,v], lh_small_objects[s,v]]
# 		lh_tripartite_organization_subjects[s,v] = np.argsort(lh_data)[-1]
# 		if lh_ncsnr[s,v] < threshold_ncsnr: # Threshold with ncsnr
# 			lh_tripartite_organization_subjects[s,v] = np.nan
# 		# Right hemisphere
# 		rh_data = [rh_animals[s,v], rh_big_objects[s,v], rh_small_objects[s,v]]
# 		rh_tripartite_organization_subjects[s,v] = np.argsort(rh_data)[-1]
# 		if rh_ncsnr[s,v] < threshold_ncsnr: # Threshold with ncsnr
# 			rh_tripartite_organization_subjects[s,v] = np.nan

# # Threshold the responses based on subject prevalence
# lh_tripartite_organization = np.zeros((lh_animals.shape[1]), dtype=np.float32)
# lh_tripartite_organization[:] = np.nan
# rh_tripartite_organization = np.zeros((rh_animals.shape[1]), dtype=np.float32)
# rh_tripartite_organization[:] = np.nan
# threshold_sub = 3 # threshold based on subjects showing the effect
# for v in range(lh_tripartite_organization_subjects.shape[1]):
# 	# Left hemisphere
# 	unique, counts = np.unique(lh_tripartite_organization_subjects[:,v],
# 		return_counts=True, equal_nan=False)
# 	if max(counts) >= threshold_sub: # Threshold with number of subjects showing the effect
# 		idx = np.argsort(counts)[-1]
# 		lh_tripartite_organization[v] = unique[idx]
# 	# Right hemisphere
# 	unique, counts = np.unique(rh_tripartite_organization_subjects[:,v],
# 		return_counts=True, equal_nan=False)
# 	if max(counts) >= threshold_sub: # Threshold with number of subjects showing the effect
# 		idx = np.argsort(counts)[-1]
# 		rh_tripartite_organization[v] = unique[idx]

# # Plot the results
# tripartite_organization = np.append(lh_tripartite_organization,
# 	rh_tripartite_organization)
# vertex_data = cortex.Vertex(tripartite_organization, subject,
# 	cmap=custom_cmap, vmin=0, vmax=2, with_colorbar=True)
# fig = cortex.quickshow(vertex_data,
# #	height=500, # Increase resolution of map and ROI contours
# 	with_curvature=True,
# 	curvature_brightness=0.5,
# 	with_rois=True,
# 	with_labels=True,
# 	linewidth=2,
# 	linecolor=(1, 1, 1),
# 	with_colorbar=True
# 	)






# Plot results on inflated surfaces # !!!

