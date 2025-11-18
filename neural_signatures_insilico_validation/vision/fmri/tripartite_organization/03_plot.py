"""Test the tripartite organization effect (Konkle & Caramazza, 2013) on in
silico fMRI responses, and plot the results.

Parameters
----------
images : str
    Whether to use 'naturalistic' or 'texforms' images.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) to
    consider a vertex for the tripartite organization analysis.
encoding_threshold : float
    The threshold on the encoding models explained variance to consider a
    vertex for the tripartite organization analysis (in % units).
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

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, default='naturalistic')
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot parameters
# =============================================================================
subject = 'fsaverage'
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
custom_cmap = mcolors.ListedColormap([(143/255, 25/255, 250/255),
    (43/255, 141/255, 248/255), (243/255, 85/255, 20/255)])


# =============================================================================
# Load the results # !!!
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'insilico_fmri_responses',
    'insilico_fmri_responses_images-'+args.images+'.npy')

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
# Plot the ROI vertex overlap with categorical zones # !!!
# =============================================================================


# =============================================================================
# Plot the tripartite organization results on a brain surface # !!!
# =============================================================================
# Perform the tripartite organization analysis on the fMRI responses averaged
# across subjects.

# Average the responses across subjects
lh_animals_avg = np.nanmean(lh_animals, 0)
rh_animals_avg = np.nanmean(rh_animals, 0)
lh_big_objects_avg = np.nanmean(lh_big_objects, 0)
rh_big_objects_avg = np.nanmean(rh_big_objects, 0)
lh_small_objects_avg = np.nanmean(lh_small_objects, 0)
rh_small_objects_avg = np.nanmean(rh_small_objects, 0)

# Append the in silico fMRI responses for the three conditions
lh_data = np.array([lh_animals_avg, lh_big_objects_avg, lh_small_objects_avg])
rh_data = np.array([rh_animals_avg, rh_big_objects_avg, rh_small_objects_avg])

# For each vertex, select the condition leading to highest response
lh_tripartite_organization = np.argsort(lh_data, axis=0)[-1].astype(np.float32)
rh_tripartite_organization = np.argsort(rh_data, axis=0)[-1].astype(np.float32)

# Threshold with univariate response magnitude
threshold_resp = -.25
lh_idx_nan = np.where(np.max(lh_data, 0) < threshold_resp)[0]
rh_idx_nan = np.where(np.max(rh_data, 0) < threshold_resp)[0]
lh_tripartite_organization[lh_idx_nan] = np.nan
rh_tripartite_organization[rh_idx_nan] = np.nan

# Threshold with ncsnr
lh_idx_nan = np.where(np.isnan(lh_animals_avg))[0]
rh_idx_nan = np.where(np.isnan(rh_animals_avg))[0]
lh_tripartite_organization[lh_idx_nan] = np.nan
rh_tripartite_organization[rh_idx_nan] = np.nan

# Append the results across left and right hemispheres
data = np.append(lh_tripartite_organization, rh_tripartite_organization)

# Plot the results on flat surfaces
vertex_data = cortex.Vertex(data, subject, cmap=custom_cmap, vmin=0, vmax=2,
    with_colorbar=False)
fig = cortex.quickshow(vertex_data,
    height=2000, # Increase resolution of map and ROI contours
    with_curvature=True,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    linewidth=2,
    linecolor=(1, 1, 1),
    with_labels=True,
    labelsize=15,
    curvature_brightness=0.5,
    with_colorbar=False
    )
# Save the figure
file_name = os.path.join(save_dir, 'tripartite_organization_flat_images-'+
    args.images+'.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
    format='svg')

# Plot results on inflated surfaces # !!! DELETE?
#import nilearn
#from nilearn.plotting import view_surf
#from nilearn.datasets import load_fsaverage # type: ignore
# # Get the fsaverage mesh
# fsaverage_meshes = load_fsaverage(mesh='fsaverage')
# # Create the inflated surface plot
# view = view_surf(
#     surf_mesh=fsaverage_meshes["inflated"],
#     surf_map=data,
#     hemi="both", # type: ignore
#     title=None
# )
# view
# view.save_as_html("inflated_surface_plot.html")
# # Save the figure
# file_name = os.path.join(save_dir, 'tripartite_organization_inflated_images-'+
#     args.images+'.svg')
# fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
#     format='svg')