"""Plot the encoding models' prediction accuracy for the test stimuli.

Parameters
----------
subjects : list
	List with all used NSD subjects.
rois : list of str
	List of all modeled ROIs.
model : str
	Name of the used encoding model.
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'EBA', 'FBA-2', 'OFA', 'FFA-1', 'FFA-2', 'PPA', 'RSC', 'OPA',
	'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'early', 'midventral',
	'midlateral', 'midparietal', 'parietal', 'lateral', 'ventral'])
parser.add_argument('--model', type=str, default='fwrf')
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()


# =============================================================================
# Plot parameters
# =============================================================================
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


# =============================================================================
# Plot noise-ceiling-normalized explained variance on barplots, for each ROI
# =============================================================================
# Load the encoding models' encoding accuracy
explained_variance = np.zeros((len(args.subjects), len(args.rois)))
metadata_dir = os.path.join(args.nest_dir, 'encoding_models', 'modality-fmri',
	'train_dataset-nsd', 'model-'+args.model, 'metadata')
for s, sub in enumerate(args.subjects):
	for r, roi in enumerate(args.rois):
		file_name = 'metadata_sub-' + format(sub, '02') + '_roi-' + roi + '.npy'
		metadata = np.load(os.path.join(metadata_dir, file_name),
			allow_pickle=True).item()
		explained_variance[s,r] = np.mean(metadata['encoding_models']\
			['explained_variance'])

# Create the plot
fig, axs = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
x = np.arange(len(args.subjects))
width = 0.4

for r, roi in enumerate(args.rois):

	# Plot the encoding accuracies
	axs[r].bar(x, explained_variance[:,r], width=width, color=colors[0])

	# Plot the encoding accuracies subject-mean
	y = np.mean(explained_variance[:,r], 0)
	axs[r].plot([min(x), max(x)], [y, y], '--', color='k', linewidth=2,
		alpha=0.4, label='Subjects mean')

	# y-axis
	if r in [0, 6, 12, 18]:
		axs[r].set_ylabel('Noise-ceiling-normalized\nexplained variance (%)',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		ylabels = np.arange(0, 101, 20)
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=0, top=100)

	# x-axis
	if r in [18, 19, 20, 21, 22, 23]:
		axs[r].set_xlabel('Subjects', fontsize=fontsize)
		xticks = x
		xlabels = ['1', '2', '3', '4', '5', '6', '7', '8']
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=fontsize)

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

# y-axis
axs[23].set_xlabel('Subjects', fontsize=fontsize)

# Save the figure
fig.savefig('noise_ceiling_normalized_explained_variance_roi_barplot.svg',
	bbox_inches='tight', transparent=True, format='svg')
fig.savefig('noise_ceiling_normalized_explained_variance_roi_barplot.png',
	dpi=300, bbox_inches='tight', transparent=False, format='png')


# =============================================================================
# Plot noise-ceiling-normalized explained variance on scatterplots, for each ROI
# =============================================================================
# Loop across subjects
for s, sub in enumerate(args.subjects):

	s = 7
	sub = s+1

	# Load the encoding models' encoding accuracy
	noise_ceiling = {}
	r2 = {}
	metadata_dir = os.path.join(args.nest_dir, 'encoding_models',
		'modality-fmri', 'train_dataset-nsd', 'model-'+args.model, 'metadata')
	for r, roi in enumerate(args.rois):
		file_name = 'metadata_sub-' + format(sub, '02') + '_roi-' + roi + '.npy'
		metadata = np.load(os.path.join(metadata_dir, file_name),
			allow_pickle=True).item()
		noise_ceiling[roi] = metadata['encoding_models']['noise_ceiling']
		r2[roi] = metadata['encoding_models']['r2']

	# Create the plot
	fig, axs = plt.subplots(nrows=4, ncols=6, sharex=True, sharey=True)
	axs = np.reshape(axs, (-1))

	for r, roi in enumerate(args.rois):

		# Plot diagonal dashed line
		axs[r].plot(np.arange(-1,1.1,.1), np.arange(-1,1.1,.1), '--k',
			linewidth=2, alpha=.5, label='_nolegend_')

		# Plot the results
		axs[r].scatter(noise_ceiling[roi], r2[roi], color=colors[0], alpha=.3)
		axs[r].set_aspect('equal')

		# y-axis
		ticks = np.arange(0, 1.1, .2)
		labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
		if r in [0, 6, 12, 18]:
			axs[r].set_ylabel('$r²$', fontsize=fontsize)
			plt.yticks(ticks=ticks, labels=labels)
		axs[r].set_ylim(bottom=-.1, top=.9)

		# x-axis
		if r in [18, 19, 20, 21, 22, 23]:
			axs[r].set_xlabel('Noise ceiling', fontsize=fontsize)
			plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize)
		axs[r].set_xlim(left=-.1, right=.9)

		# Title
		axs[r].set_title(roi, fontsize=fontsize)

	# y-axis
	axs[23].set_xlabel('Noise ceiling', fontsize=fontsize)

	# Save the figure
	file_name = 'noise_ceiling_normalized_explained_variance_roi_' + \
		'scatterplot_sub-0' + str(sub) + '.svg'
	fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
	file_name = 'noise_ceiling_normalized_explained_variance_roi_' + \
		'scatterplot_sub-0' + str(sub) + '.png'
	fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=False,
		format='png')
	plt.close()
