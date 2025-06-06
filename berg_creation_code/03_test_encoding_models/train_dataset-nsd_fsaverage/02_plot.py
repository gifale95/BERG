"""Plot the encoding models' prediction accuracy for the test stimuli.

Parameters
----------
subjects : list
	List with all used NSD subjects.
model : str
	Name of the used encoding model.
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from copy import copy
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
r2 = []
noise_ceiling = []
explained_variance = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
	'train_dataset-nsd_fsaverage', 'model-'+args.model, 'metadata')

for sub in args.subjects:
	file_name = f'metadata_subject-{int(sub):02d}.npy'
	metadata = np.load(os.path.join(metadata_dir, file_name),
		allow_pickle=True).item()
	# Append the encoding accuracies across hemispheres
	r2.append(np.append(
		metadata['encoding_models']['lh_r2'],
		metadata['encoding_models']['rh_r2']))
	noise_ceiling.append(np.append(
		metadata['encoding_models']['lh_noise_ceiling'],
		metadata['encoding_models']['rh_noise_ceiling']))
	explained_variance.append(np.append(
		metadata['encoding_models']['lh_explained_variance'],
		metadata['encoding_models']['rh_explained_variance']))


# =============================================================================
# Plot parameters for colorbar
# =============================================================================
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage'


# =============================================================================
# Plot the r2 scores (averaged across subjects)
# =============================================================================
vertex_data = cortex.Vertex(np.mean(r2, 0), subject, cmap='hot', vmin=0,
	vmax=1, with_colorbar=True)

fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	curvature_brightness=0.5,
	with_rois=True,
	with_labels=True,
	labelsize=20,
	linewidth=5,
	linecolor=(1, 1, 1),
	with_colorbar=True
	)

plt.title("$r²$ | Subject average", fontsize=30)

plt.show()

fig.savefig('r2_subject-average.svg', bbox_inches='tight', transparent=True,
	format='svg')
fig.savefig('r2_subject-average.png', dpi=300, bbox_inches='tight',
	transparent=True, format='png')


# =============================================================================
# Plot the noise ceiling scores (averaged across subjects)
# =============================================================================
vertex_data = cortex.Vertex(np.mean(noise_ceiling, 0), subject, cmap='hot',
	vmin=0, vmax=1, with_colorbar=True)

fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	curvature_brightness=0.5,
	with_rois=True,
	with_labels=True,
	labelsize=20,
	linewidth=5,
	linecolor=(1, 1, 1),
	with_colorbar=True
	)

plt.title("Noise ceiling ($r²$) | Subject average", fontsize=30)

plt.show()

fig.savefig('noise_ceiling_subject-average.svg', bbox_inches='tight',
	transparent=True, format='svg')
fig.savefig('noise_ceiling_subject-average.png', dpi=300, bbox_inches='tight',
	transparent=True, format='png')


# =============================================================================
# Plot the noise-ceiling-normalized explained variance (averaged across subjects)
# =============================================================================
# Remove vertices with noise ceiling values below a certain threshold, since
# they cannot be interpreted in terms of modeling
expl_var_threshold = []
for s in range(len(explained_variance)):
	expl_var = copy(explained_variance[s])
	idx = noise_ceiling[s] < 0.1
	expl_var[idx] = np.nan
	expl_var_threshold.append(expl_var)

vertex_data = cortex.Vertex(np.nanmean(expl_var_threshold, 0), subject,
	cmap='hot', vmin=0, vmax=100, with_colorbar=True)

fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	curvature_brightness=0.5,
	with_rois=True,
	with_labels=True,
	labelsize=20,
	linewidth=5,
	linecolor=(1, 1, 1),
	with_colorbar=True
	)

plt.title("Noise-ceiling-normalized explained variance (%) | Subject average",
	fontsize=30)

plt.show()

fig.savefig('noise_ceiling_normalized_explained_variance_subject-average.svg',
	bbox_inches='tight', transparent=True, format='svg')
fig.savefig('noise_ceiling_normalized_explained_variance_subject-average.png',
	dpi=300, bbox_inches='tight', transparent=True, format='png')


# =============================================================================
# Plot the r2 scores (for single subjects)
# =============================================================================
for s, sub in enumerate(args.subjects):

	vertex_data = cortex.Vertex(r2[s], subject, cmap='hot', vmin=0, vmax=1,
		with_colorbar=True)

	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=True,
		labelsize=20,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)

	plt.title("$r²$ | Subject "+str(sub), fontsize=30)

	plt.show()

	fig.savefig('r2_subject-'+str(sub)+'.svg', bbox_inches='tight',
		transparent=True, format='svg')
	fig.savefig('r2_subject-'+str(sub)+'.png', dpi=300, bbox_inches='tight',
		transparent=True, format='png')
	plt.close()


# =============================================================================
# Plot the noise ceiling scores (for single subjects)
# =============================================================================
for s, sub in enumerate(args.subjects):

	vertex_data = cortex.Vertex(noise_ceiling[s], subject, cmap='hot', vmin=0,
		vmax=1, with_colorbar=True)

	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=True,
		labelsize=20,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)

	plt.title("Noise ceiling ($r²$) | Subject "+str(sub), fontsize=30)

	plt.show()

	fig.savefig('noise_ceiling_subject-'+str(sub)+'.svg', bbox_inches='tight',
		transparent=True, format='svg')
	fig.savefig('noise_ceiling_subject-'+str(sub)+'.png', dpi=300,
		bbox_inches='tight', transparent=True, format='png')
	plt.close()


# =============================================================================
# Plot the noise-ceiling-normalized explained variance (for single subjects)
# =============================================================================
# Remove vertices with noise ceiling values below a certain threshold, since
# they cannot be interpreted in terms of modeling
expl_var_threshold = []
for s in range(len(explained_variance)):
	expl_var = copy(explained_variance[s])
	idx = noise_ceiling[s] < 0.1
	expl_var[idx] = np.nan
	expl_var_threshold.append(expl_var)

for s, sub in enumerate(args.subjects):

	vertex_data = cortex.Vertex(expl_var_threshold[s], subject, cmap='hot',
		vmin=0, vmax=100, with_colorbar=True)

	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=True,
		labelsize=20,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)

	plt.title("Noise-ceiling-normalized explained variance (%) | Subject "+
		str(sub), fontsize=30)

	plt.show()

	fig.savefig('noise_ceiling_normalized_explained_variance_subject-'+str(sub)+
		'.svg', bbox_inches='tight', transparent=True, format='svg')
	fig.savefig('noise_ceiling_normalized_explained_variance_subject-'+str(sub)+
		'.png', dpi=300, bbox_inches='tight', transparent=True, format='png')
	plt.close()


# =============================================================================
# Plot noise-ceiling-normalized explained variance on barplots, for each ROI
# =============================================================================
# ROI list
rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA-1',
	'FFA-2', 'OFA', 'PPA', 'OPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2', 'early',
	'midventral', 'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral']

# Format the results for plotting
acc = np.zeros((len(args.subjects), len(rois)))
for r, roi in enumerate(rois):
	for s, sub in enumerate(args.subjects):
		# Load the metadata
		file_name = f'metadata_subject-{int(sub):02d}.npy'
		metadata = np.load(os.path.join(metadata_dir, file_name),
			allow_pickle=True).item()
		# Get the ROI mask
		lh_roi_mask = metadata['fmri']['lh_fsaverage_rois'][roi]
		rh_roi_mask = metadata['fmri']['rh_fsaverage_rois'][roi]
		# Select the ROI data
		lh_explained_variance = metadata['encoding_models']\
			['lh_explained_variance'][lh_roi_mask]
		rh_explained_variance = metadata['encoding_models']\
			['rh_explained_variance'][rh_roi_mask]
		lh_noise_ceiling = metadata['encoding_models']['lh_noise_ceiling']\
			[lh_roi_mask]
		rh_noise_ceiling = metadata['encoding_models']['rh_noise_ceiling']\
			[rh_roi_mask]
		# Remove vertices with noise ceiling values below a certain threshold,
		# since they cannot be interpreted in terms of modeling
		lh_idx = lh_noise_ceiling < 0.1
		rh_idx = rh_noise_ceiling < 0.1
		lh_explained_variance = np.delete(lh_explained_variance, lh_idx)
		rh_explained_variance = np.delete(rh_explained_variance, rh_idx)
		# Average the encoding accuracy across vertices from both hemispheres
		explained_variance = np.append(lh_explained_variance,
			rh_explained_variance)
		acc[s,r] = np.mean(explained_variance)

# Plot parameters
fontsize = 15
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(170/255, 118/255, 186/255)]

# Plot the encoding accuracy results
fig, axs = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
x = np.arange(len(acc))
width = 0.4
for r, roi in enumerate(rois):
	# Plot the encoding accuracies
	axs[r].bar(x, acc[:,r], width=width, color=colors[0])
	# Plot the encoding accuracies subject-mean
	y = np.mean(acc[:,r], 0)
	axs[r].plot([min(x), max(x)], [y, y], '--', color='k', linewidth=2,
		alpha=0.4, label='Subjects mean')
	# y-axis
	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('Noise-ceiling-normalized\nexplained variance (%)',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		axs[r].set_yticks(yticks)
		axs[r].set_yticklabels(yticks)
		axs[r].set_ylim(bottom=0, top=100)
	# x-axis
	if r in [18, 19, 20, 21, 22, 23]:
		axs[r].set_xlabel('Subjects', fontsize=fontsize)
		xticks = np.arange(len(args.subjects))  # Make sure this matches the actual x data
		xlabels = [str(i) for i in args.subjects]  # Use actual subject numbers
		axs[r].set_xticks(xticks)
		axs[r].set_xticklabels(xlabels)
	# Title
	axs[r].set_title(roi, fontsize=fontsize)
# Save the figure
fig.savefig('noise_ceiling_normalized_explained_variance_roi_barplot.svg',
	bbox_inches='tight', transparent=True, format='svg')
fig.savefig('noise_ceiling_normalized_explained_variance_roi_barplot.png',
	dpi=300, bbox_inches='tight', transparent=True, format='png')
