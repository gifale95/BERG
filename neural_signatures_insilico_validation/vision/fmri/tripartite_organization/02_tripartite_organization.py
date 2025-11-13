"""Test the tripartite organization effect (Konkle & Caramazza, 2013) on in
silico fMRI responses, and plot the results.

Parameters
----------
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) to
    consider a vertex for the tripartite organization analysis.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nilearn
from nilearn.plotting import view_surf
from nilearn.datasets import load_fsaverage # type: ignore


parser = argparse.ArgumentParser()
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
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
# Load the in silico fMRI responses for the tripartite organization images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri_tripartite_organization', 'insilico_fmri_responses',
    'insilico_fmri_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

lh_animals = data['lh_animals']
rh_animals = data['rh_animals']
lh_big_objects = data['lh_big_objects']
rh_big_objects = data['rh_big_objects']
lh_small_objects = data['lh_small_objects']
rh_small_objects = data['rh_small_objects']
metadata = data['metadata']
del data


# =============================================================================
# Set vertices with NCSNR below threshold to NaN
# =============================================================================
for s in range(len(metadata)):
    # Left hemisphere
    lh_ncsnr = metadata[s]['fmri']['lh_ncsnr']
    idx = np.where(lh_ncsnr < args.ncsnr_threshold)[0]
    lh_animals[s,idx] = np.nan
    lh_big_objects[s,idx] = np.nan
    lh_small_objects[s,idx] = np.nan
    # Right hemisphere
    rh_ncsnr = metadata[s]['fmri']['rh_ncsnr']
    idx = np.where(rh_ncsnr < args.ncsnr_threshold)[0]
    rh_animals[s,idx] = np.nan
    rh_big_objects[s,idx] = np.nan
    rh_small_objects[s,idx] = np.nan


# =============================================================================
# Compute the overlap of different ROIs with the categorical zones
# =============================================================================
# For each subject, compute the overlap of face/body-selective and
# scene-selective ROIs with the animal-selective and big object-selective
# cortical zones, respectively.

# Empty arrays of shape (n_subjects,) for face/body-selective ROIs
FFA_animal = np.zeros((len(metadata)))
OFA_animal = np.zeros((len(metadata)))
EBA_animal = np.zeros((len(metadata)))
FBA_animal = np.zeros((len(metadata)))

# Empty arrays of shape (n_subjects,) for scene-selective ROIs
PPA_big_objects = np.zeros((len(metadata)))
OPA_big_objects = np.zeros((len(metadata)))
RSC_big_objects = np.zeros((len(metadata)))

# Loop across subjects
for s in range(len(metadata)):

    # Loop across ROIs
    rois = ['FFA', 'OFA', 'EBA', 'FBA', 'PPA', 'OPA', 'RSC']
    for roi in rois:

        # Initialize counters
        tot_vertices = 0
        count = 0

        # Loop across hemispheres
        for hem in ['lh', 'rh']:

            # Get the vertex indices for the ROI
            if roi == 'FFA' or roi == 'FBA':
                # Get the vertex indices for both parts of the ROI
                lh_idx = np.append(metadata[hem+'_fsaverage_rois'][f'{roi}-1'],
                    metadata[hem+'_fsaverage_rois'][f'{roi}-2'])
                lh_idx.sort()
            else:
                # Get the vertex indices for the ROI
                lh_idx = metadata[hem+'_fsaverage_rois'][roi]

            # Calculate the count of vertices selective for animals or big
            # objects
            for v in lh_idx:
                if np.isnan(lh_animals[s,v]) or np.isnan(lh_small_objects[s,v]) or np.isnan(lh_big_objects[s,v]):
                    continue
                else:
                    tot_vertices += 1
                    if roi in ['FFA', 'OFA', 'EBA', 'FBA']:
                        if lh_animals[s,v] > lh_small_objects[s,v] and lh_animals[s,v] > lh_big_objects[s,v]:
                            count += 1
                    elif roi in ['PPA', 'OPA', 'RSC']:
                        if lh_big_objects[s,v] > lh_small_objects[s,v] and lh_big_objects[s,v] > lh_animals[s,v]:
                            count += 1

        # Store the results for each ROI
        if roi == 'FFA':
            FFA_animal[s] = count / tot_vertices * 100
        elif roi == 'OFA':
            OFA_animal[s] = count / tot_vertices * 100
        elif roi == 'EBA':
            EBA_animal[s] = count / tot_vertices * 100
        elif roi == 'FBA':
            FBA_animal[s] = count / tot_vertices * 100
        if roi == 'PPA':
            PPA_big_objects[s] = count / tot_vertices * 100
        elif roi == 'OPA':
            OPA_big_objects[s] = count / tot_vertices * 100
        elif roi == 'RSC':
            RSC_big_objects[s] = count / tot_vertices * 100

# Average across subjects
FFA_animal_mean = np.mean(FFA_animal)
OFA_animal_mean = np.mean(OFA_animal)
EBA_animal_mean = np.mean(EBA_animal)
FBA_animal_mean = np.mean(FBA_animal)
PPA_big_objects_mean = np.mean(PPA_big_objects)
OPA_big_objects_mean = np.mean(OPA_big_objects)
RSC_big_objects_mean = np.mean(RSC_big_objects)

# Print results
print('Overlap of face/body-selective ROIs with animal-selective zone:')
print(f'FFA: {FFA_animal_mean:.2f}%')
print(f'OFA: {OFA_animal_mean:.2f}%')
print(f'EBA: {EBA_animal_mean:.2f}%')
print(f'FBA: {FBA_animal_mean:.2f}%')
print('')
print('Overlap of scene-selective ROIs with big object-selective zone:')
print(f'PPA: {PPA_big_objects_mean:.2f}%')
print(f'OPA: {OPA_big_objects_mean:.2f}%')
print(f'RSC: {RSC_big_objects_mean:.2f}%')


# =============================================================================
# Tripartite organization analysis (across-subjects)
# =============================================================================
# Perform the tripartite organization analysis on the fMRI responses averaged
# across subjects.

# For each condition, subtract the mean of the other two conditions, and
# average across subjects
lh_animals_avg = np.nanmean(lh_animals, 0) - \
    ((np.nanmean(lh_big_objects, 0) + np.nanmean(lh_small_objects, 0)) / 2)
rh_animals_avg = np.nanmean(rh_animals, 0) - \
    ((np.nanmean(rh_big_objects, 0) + np.nanmean(rh_small_objects, 0)) / 2)
lh_big_objects_avg = np.nanmean(lh_big_objects, 0) - \
    ((np.nanmean(lh_animals, 0) + np.nanmean(lh_small_objects, 0)) / 2)
rh_big_objects_avg = np.nanmean(rh_big_objects, 0) - \
    ((np.nanmean(rh_animals, 0) + np.nanmean(rh_small_objects, 0)) / 2)
lh_small_objects_avg = np.nanmean(lh_small_objects, 0) - \
    ((np.nanmean(lh_animals, 0) + np.nanmean(lh_big_objects, 0)) / 2)
rh_small_objects_avg = np.nanmean(rh_small_objects, 0) - \
    ((np.nanmean(rh_animals, 0) + np.nanmean(rh_big_objects, 0)) / 2)

# Data without mean subtraction # !!! DELETE?
# lh_animals_avg = np.mean(lh_animals, 0)
# rh_animals_avg = np.mean(rh_animals, 0)
# lh_big_objects_avg = np.mean(lh_big_objects, 0)
# rh_big_objects_avg = np.mean(rh_big_objects, 0)
# lh_small_objects_avg = np.mean(lh_small_objects, 0)
# rh_small_objects_avg = np.mean(rh_small_objects, 0)

# Append the in silico fMRI responses for the three conditions
lh_data = np.array([lh_animals_avg, lh_big_objects_avg, lh_small_objects_avg])
rh_data = np.array([rh_animals_avg, rh_big_objects_avg, rh_small_objects_avg])

# For each vertex, select the condition leading to highest response
lh_tripartite_organization = np.argsort(lh_data, axis=0)[-1].astype(np.float32)
rh_tripartite_organization = np.argsort(rh_data, axis=0)[-1].astype(np.float32)

# Threshold with univariate response magnitude
threshold_resp = -10 # 0.025 # !!! WHICH VALUE?
lh_idx_nan = np.where(np.max(lh_data, 0) < threshold_resp)[0]
rh_idx_nan = np.where(np.max(rh_data, 0) < threshold_resp)[0]
lh_tripartite_organization[lh_idx_nan] = np.nan
rh_tripartite_organization[rh_idx_nan] = np.nan

# Threshold with ncsnr
lh_idx_nan = np.where(np.isnan(lh_animals_avg))[0]
rh_idx_nan = np.where(np.isnan(rh_animals_avg))[0]
lh_tripartite_organization[lh_idx_nan] = np.nan
rh_tripartite_organization[rh_idx_nan] = np.nan

# Set early/intermediate stream vertices to NaN # !!! DELETE?
# metadata_dir = os.path.join(args.nest_dir, 'model_training_datasets',
#     'train_dataset-nsd_fsaverage', 'metadata_subject-1.npy')
# metadata = np.load(metadata_dir, allow_pickle=True).item()
# lh_early = metadata['lh_fsaverage_rois']['early']
# rh_early = metadata['rh_fsaverage_rois']['early']
# lh_tripartite_organization[lh_early] = np.nan
# rh_tripartite_organization[rh_early] = np.nan

# Append the results across left and right hemispheres
data = np.append(lh_tripartite_organization, rh_tripartite_organization)

# Plot the results on flat surfaces
vertex_data = cortex.Vertex(data, subject, cmap=custom_cmap, vmin=0, vmax=2,
    with_colorbar=False)
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
file_name = 'fmri_tripartite_effect_between_subjects_flat.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
    format='svg')

# Plot results on inflated surfaces # !!!
# Get the fsaverage mesh
fsaverage_meshes = load_fsaverage(mesh='fsaverage')
# Create the inflated surface plot
view = view_surf(
    surf_mesh=fsaverage_meshes["inflated"],
    surf_map=data,
    hemi="both", # type: ignore
    title=None
)
view
view.save_as_html("inflated_surface_plot.html")
# Save the figure
file_name = 'fmri_tripartite_effect_between_subjects_inflated.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
    format='svg')


# =============================================================================
# Tripartite organization analysis (within-subjects) # !!! DELETE?
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