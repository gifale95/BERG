"""Plot the encoding models' in-distribution (using the 515 NSD-core test
stimuli) and out-of-distribution (using the 286 NSD-synthetic stimuli)
prediction accuracy.

Parameters
----------
subjects : list
	List with all used NSD subjects.
model_id : str
	Unique identifier of the model to load.
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
import ast


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=ast.literal_eval, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--model_id', type=str, default='fmri-nsd_fsaverage-<your_model_name>')
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
r2_nsdcore = []
noise_ceiling_nsdcore = []
explained_variance_nsdcore = []
r2_nsdsynthetic = []
noise_ceiling_nsdsynthetic = []
explained_variance_nsdsynthetic = []

model_name = args.model_id.split('-')[-1]
metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
	'train_dataset-nsd_fsaverage', 'model-'+model_name, 'metadata')

for sub in args.subjects:
	file_name = 'metadata_subject-' + format(sub, '02') + '.npy'
	metadata = np.load(os.path.join(metadata_dir, file_name),
		allow_pickle=True).item()
	# Append the encoding accuracies across hemispheres
	r2_nsdcore.append(np.append(
		metadata['encoding_models']['lh_r2_nsdcore'],
		metadata['encoding_models']['rh_r2_nsdcore']))
	noise_ceiling_nsdcore.append(np.append(
		metadata['encoding_models']['lh_noise_ceiling_nsdcore'],
		metadata['encoding_models']['rh_noise_ceiling_nsdcore']))
	explained_variance_nsdcore.append(np.append(
		metadata['encoding_models']['lh_explained_variance_nsdcore'],
		metadata['encoding_models']['rh_explained_variance_nsdcore']))
	r2_nsdsynthetic.append(np.append(
		metadata['encoding_models']['lh_r2_nsdsynthetic'],
		metadata['encoding_models']['rh_r2_nsdsynthetic']))
	noise_ceiling_nsdsynthetic.append(np.append(
		metadata['encoding_models']['lh_noise_ceiling_nsdsynthetic'],
		metadata['encoding_models']['rh_noise_ceiling_nsdsynthetic']))
	explained_variance_nsdsynthetic.append(np.append(
		metadata['encoding_models']['lh_explained_variance_nsdsynthetic'],
		metadata['encoding_models']['rh_explained_variance_nsdsynthetic']))


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
# Plot the in-distribution prediction accuracy (averaged across subjects)
# =============================================================================
# r2 scores
vertex_data = cortex.Vertex(np.mean(r2_nsdcore, 0), subject, cmap='hot', vmin=0,
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
plt.title("$r²$, in-distribution generalization (NSD-core) | Subject average",
	fontsize=30)
plt.show()
fig.savefig('r2_nsdcore_subject-average.svg', bbox_inches='tight',
	transparent=False, format='svg')
fig.savefig('r2_nsdcore_subject-average.png', dpi=300, bbox_inches='tight',
	transparent=False, format='png')
plt.close()

# Noise ceiling scores
vertex_data = cortex.Vertex(np.mean(noise_ceiling_nsdcore, 0), subject,
	cmap='hot', vmin=0, vmax=1, with_colorbar=True)
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
plt.title("Noise ceiling ($r²$), in-distribution stimuli (NSD-core) | Subject average",
	fontsize=30)
plt.show()
fig.savefig('noise_ceiling_nsdcore_subject-average.svg', bbox_inches='tight',
	transparent=False, format='svg')
fig.savefig('noise_ceiling_nsdcore_subject-average.png', dpi=300,
	bbox_inches='tight', transparent=False, format='png')
plt.close()

# Noise-ceiling-normalized explained variance scores
# Remove vertices with noise ceiling values below a certain threshold, since
# they cannot be interpreted in terms of modeling
expl_var_threshold = []
for s in range(len(explained_variance_nsdcore)):
	expl_var = copy(explained_variance_nsdcore[s])
	idx = noise_ceiling_nsdcore[s] < 0.1
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
plt.title("Noise-ceiling-normalized explained variance (%),\nin-distribution generalization (NSD-core) | Subject average",
	fontsize=30)
plt.show()
fig.savefig('noise_ceiling_normalized_explained_variance_nsdcore_subject-average.svg',
	bbox_inches='tight', transparent=False, format='svg')
fig.savefig('noise_ceiling_normalized_explained_variance_nsdcore_subject-average.png',
	dpi=300, bbox_inches='tight', transparent=False, format='png')
plt.close()


# =============================================================================
# Plot the out-of-distribution prediction accuracy (averaged across subjects)
# =============================================================================
# r2 scores
vertex_data = cortex.Vertex(np.mean(r2_nsdsynthetic, 0), subject, cmap='hot',
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
plt.title("$r²$, out-of-distribution generalization (NSD-synthetic) | Subject average",
	fontsize=30)
plt.show()
fig.savefig('r2_nsdsynthetic_subject-average.svg', bbox_inches='tight',
	transparent=False, format='svg')
fig.savefig('r2_nsdsynthetic_subject-average.png', dpi=300, bbox_inches='tight',
	transparent=False, format='png')
plt.close()

# Noise ceiling scores
vertex_data = cortex.Vertex(np.mean(noise_ceiling_nsdsynthetic, 0), subject,
	cmap='hot', vmin=0, vmax=1, with_colorbar=True)
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
plt.title("Noise ceiling ($r²$), out-of-distribution stimuli (NSD-synthetic) | Subject average",
	fontsize=30)
plt.show()
fig.savefig('noise_ceiling_nsdsynthetic_subject-average.svg',
	bbox_inches='tight', transparent=False, format='svg')
fig.savefig('noise_ceiling_nsdsynthetic_subject-average.png', dpi=300,
	bbox_inches='tight', transparent=False, format='png')
plt.close()

# Noise-ceiling-normalized explained variance scores
# Remove vertices with noise ceiling values below a certain threshold, since
# they cannot be interpreted in terms of modeling
expl_var_threshold = []
for s in range(len(explained_variance_nsdsynthetic)):
	expl_var = copy(explained_variance_nsdsynthetic[s])
	idx = noise_ceiling_nsdsynthetic[s] < 0.3
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
plt.title("Noise-ceiling-normalized explained variance (%),\nout-of-distribution generalization (NSD-synthetic) | Subject average",
	fontsize=30)
plt.show()
fig.savefig('noise_ceiling_normalized_explained_variance_nsdsynthetic_subject-average.svg',
	bbox_inches='tight', transparent=False, format='svg')
fig.savefig('noise_ceiling_normalized_explained_variance_nsdsynthetic_subject-average.png',
	dpi=300, bbox_inches='tight', transparent=False, format='png')
plt.close()


# =============================================================================
# Plot the in-distribution prediction accuracy (for single subjects)
# =============================================================================
# r2 scores
for s, sub in enumerate(args.subjects):
	vertex_data = cortex.Vertex(r2_nsdcore[s], subject, cmap='hot', vmin=0,
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
	plt.title("$r²$, in-distribution generalization (NSD-core) | Subject "+str(sub),
		fontsize=30)
	plt.show()
	fig.savefig('r2_nsdcore_subject-'+str(sub)+'.svg', bbox_inches='tight',
		transparent=False, format='svg')
	fig.savefig('r2_nsdcore_subject-'+str(sub)+'.png', dpi=300,
		bbox_inches='tight', transparent=False, format='png')
	plt.close()

# Noise ceiling scores
for s, sub in enumerate(args.subjects):
	vertex_data = cortex.Vertex(noise_ceiling_nsdcore[s], subject, cmap='hot',
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
	plt.title("Noise ceiling ($r²$), in-distribution stimuli (NSD-core) | Subject "+
		str(sub), fontsize=30)
	plt.show()
	fig.savefig('noise_ceiling_nsdcore_subject-'+str(sub)+'.svg',
		bbox_inches='tight', transparent=False, format='svg')
	fig.savefig('noise_ceiling_nsdcore_subject-'+str(sub)+'.png', dpi=300,
		bbox_inches='tight', transparent=False, format='png')
	plt.close()

# Noise-ceiling-normalized explained variance scores
# Remove vertices with noise ceiling values below a certain threshold, since
# they cannot be interpreted in terms of modeling
expl_var_threshold = []
for s in range(len(explained_variance_nsdcore)):
	expl_var = copy(explained_variance_nsdcore[s])
	idx = noise_ceiling_nsdcore[s] < 0.1
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
	plt.title("Noise-ceiling-normalized explained variance (%),\nin-distribution generalization (NSD-core) | Subject "+
		str(sub), fontsize=30)
	plt.show()
	fig.savefig('noise_ceiling_normalized_explained_variance_nsdcore_subject-'+
		str(sub)+ '.svg', bbox_inches='tight', transparent=False, format='svg')
	fig.savefig('noise_ceiling_normalized_explained_variance_nsdcore_subject-'+
		str(sub)+ '.png', dpi=300, bbox_inches='tight', transparent=False,
		format='png')
	plt.close()


# =============================================================================
# Plot the out-of-distribution prediction accuracy (for single subjects)
# =============================================================================
# r2 scores
for s, sub in enumerate(args.subjects):
	vertex_data = cortex.Vertex(r2_nsdsynthetic[s], subject, cmap='hot', vmin=0,
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
	plt.title("$r²$, out-of-distribution generalization (NSD-synthetic) | Subject "+str(sub),
		fontsize=30)
	plt.show()
	fig.savefig('r2_nsdsynthetic_subject-'+str(sub)+'.svg', bbox_inches='tight',
		transparent=False, format='svg')
	fig.savefig('r2_nsdsynthetic_subject-'+str(sub)+'.png', dpi=300,
		bbox_inches='tight', transparent=False, format='png')
	plt.close()

# Noise ceiling scores
for s, sub in enumerate(args.subjects):
	vertex_data = cortex.Vertex(noise_ceiling_nsdsynthetic[s], subject,
		cmap='hot', vmin=0, vmax=1, with_colorbar=True)
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
	plt.title("Noise ceiling ($r²$), out-of-distribution stimuli (NSD-synthetic) | Subject "+
		str(sub), fontsize=30)
	plt.show()
	fig.savefig('noise_ceiling_nsdsynthetic_subject-'+str(sub)+'.svg',
		bbox_inches='tight', transparent=False, format='svg')
	fig.savefig('noise_ceiling_nsdsynthetic_subject-'+str(sub)+'.png', dpi=300,
		bbox_inches='tight', transparent=False, format='png')
	plt.close()

# Noise-ceiling-normalized explained variance scores
# Remove vertices with noise ceiling values below a certain threshold, since
# they cannot be interpreted in terms of modeling
expl_var_threshold = []
for s in range(len(explained_variance_nsdsynthetic)):
	expl_var = copy(explained_variance_nsdsynthetic[s])
	idx = noise_ceiling_nsdsynthetic[s] < 0.3
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
	plt.title("Noise-ceiling-normalized explained variance (%),\nout-of-distribution generalization (NSD-synthetic) | Subject "+
		str(sub), fontsize=30)
	plt.show()
	fig.savefig('noise_ceiling_normalized_explained_variance_nsdsynthetic_subject-'+
		str(sub)+ '.svg', bbox_inches='tight', transparent=False, format='svg')
	fig.savefig('noise_ceiling_normalized_explained_variance_nsdsynthetic_subject-'+
		str(sub)+ '.png', dpi=300, bbox_inches='tight', transparent=False,
		format='png')
	plt.close()


# =============================================================================
# Plot noise-ceiling-normalized explained variance on barplots, for each ROI
# =============================================================================
# ROI list
rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA-1',
	'FFA-2', 'OFA', 'PPA', 'OPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2', 'early',
	'midventral', 'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral']

# Format the results for plotting
acc_nsdcore = np.zeros((len(args.subjects), len(rois)))
acc_nsdsynthetic = np.zeros((len(args.subjects), len(rois)))
for s, sub in enumerate(args.subjects):
	# Load the metadata
	file_name = 'metadata_subject-' + format(sub, '02') + '.npy'
	metadata = np.load(os.path.join(metadata_dir, file_name),
		allow_pickle=True).item()
	for r, roi in enumerate(rois):
		# Get the ROI mask
		lh_roi_mask = metadata['fmri']['lh_fsaverage_rois'][roi]
		rh_roi_mask = metadata['fmri']['rh_fsaverage_rois'][roi]
		# Select the ROI data
		lh_explained_variance_nsdcore = metadata['encoding_models']\
			['lh_explained_variance_nsdcore'][lh_roi_mask]
		rh_explained_variance_nsdcore = metadata['encoding_models']\
			['rh_explained_variance_nsdcore'][rh_roi_mask]
		lh_noise_ceiling_nsdcore = metadata['encoding_models']\
			['lh_noise_ceiling_nsdcore'][lh_roi_mask]
		rh_noise_ceiling_nsdcore = metadata['encoding_models']\
			['rh_noise_ceiling_nsdcore'][rh_roi_mask]
		lh_explained_variance_nsdsynthetic = metadata['encoding_models']\
			['lh_explained_variance_nsdsynthetic'][lh_roi_mask]
		rh_explained_variance_nsdsynthetic = metadata['encoding_models']\
			['rh_explained_variance_nsdsynthetic'][rh_roi_mask]
		lh_noise_ceiling_nsdsynthetic = metadata['encoding_models']\
			['lh_noise_ceiling_nsdsynthetic'][lh_roi_mask]
		rh_noise_ceiling_nsdsynthetic = metadata['encoding_models']\
			['rh_noise_ceiling_nsdsynthetic'][rh_roi_mask]
		# Remove vertices with noise ceiling values below a certain threshold,
		# since they cannot be interpreted in terms of modeling
		lh_idx_nsdcore = lh_noise_ceiling_nsdcore < 0.1
		rh_idx_nsdcore = rh_noise_ceiling_nsdcore < 0.1
		lh_explained_variance_nsdcore = np.delete(lh_explained_variance_nsdcore,
			lh_idx_nsdcore)
		rh_explained_variance_nsdcore = np.delete(rh_explained_variance_nsdcore,
			rh_idx_nsdcore)
		lh_idx_nsdsynthetic = lh_noise_ceiling_nsdsynthetic < 0.1
		rh_idx_nsdsynthetic = rh_noise_ceiling_nsdsynthetic < 0.1
		lh_explained_variance_nsdsynthetic = np.delete(
			lh_explained_variance_nsdsynthetic, lh_idx_nsdsynthetic)
		rh_explained_variance_nsdsynthetic = np.delete(
			rh_explained_variance_nsdsynthetic, rh_idx_nsdsynthetic)
		# Average the encoding accuracy across vertices from both hemispheres
		explained_variance_nsdcore = np.append(lh_explained_variance_nsdcore,
			rh_explained_variance_nsdcore)
		acc_nsdcore[s,r] = np.mean(explained_variance_nsdcore)
		explained_variance_nsdsynthetic = np.append(
			lh_explained_variance_nsdsynthetic,
			rh_explained_variance_nsdsynthetic)
		acc_nsdsynthetic[s,r] = np.mean(explained_variance_nsdsynthetic)

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
colors = [(230/255, 159/255, 0/255), (86/255, 180/255, 233/255)]

# Plot the encoding accuracy results
# Increase figure size to accommodate all text properly
fig, axs = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True,
	figsize=(30, 30))
axs = np.reshape(axs, (-1))

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.3) # Add proper spacing

x = np.arange(len(acc_nsdcore))
width = 0.3

for r, roi in enumerate(rois):
	# Plot the encoding accuracies
	axs[r].bar(x-width/2, acc_nsdcore[:,r], width=width, color=colors[0],
		label='In-distribution generalization (NSD-core)')
	axs[r].bar(x+width/2, acc_nsdsynthetic[:,r], width=width, color=colors[1],
		label='Out-of-distribution generalization (NSD-synthetic)')
	# Plot the encoding accuracies subject-mean
	y_nsdcore = np.mean(acc_nsdcore[:,r], 0)
	axs[r].plot([min(x), max(x)], [y_nsdcore, y_nsdcore], '--', color=colors[0],
		linewidth=2, alpha=0.4,
		label='In-distribution generalization (subject average)')
	y_nsdsynthetic = np.mean(acc_nsdsynthetic[:,r], 0)
	axs[r].plot([min(x), max(x)], [y_nsdsynthetic, y_nsdsynthetic], '--',
		color=colors[1], linewidth=2, alpha=0.4,
		label='Out-of-distribution generalization (subject average)')

	# y-axis - Fix the yticks setting
	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('Noise-ceiling-normalized\nexplained variance (%)',
			fontsize=fontsize)

	# Set y-axis properties for all subplots
	axs[r].set_ylim(bottom=0, top=100)
	yticks = np.arange(0, 101, 20)
	ylabels = np.arange(0, 101, 20)
	axs[r].set_yticks(ticks=yticks)  # Use axs[r] instead of plt
	axs[r].set_yticklabels(labels=ylabels)  # Use axs[r] instead of plt

	# x-axis
	if r in [18, 19, 20, 21, 22, 23]:
		axs[r].set_xlabel('Subjects', fontsize=fontsize)

	# Set x-axis properties for all subplots
	xticks = x
	xlabels = ['1', '2', '3', '4', '5', '6', '7', '8']
	axs[r].set_xticks(ticks=xticks)  # Use axs[r] instead of plt
	axs[r].set_xticklabels(labels=xlabels, fontsize=fontsize)  # Use axs[r] instead of plt

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

# Legend
axs[0].legend(ncol=4, fontsize=fontsize, bbox_to_anchor=(8.05, -4.5),
	frameon=False, markerscale=2)

# Save the figure
fig.savefig('noise_ceiling_normalized_explained_variance_roi_barplot.svg',
	bbox_inches='tight', transparent=False, format='svg')
fig.savefig('noise_ceiling_normalized_explained_variance_roi_barplot.png',
	dpi=300, bbox_inches='tight', transparent=False, format='png')
plt.close()
