"""Plot the encoding models' prediction accuracy for the test stimuli.

Parameters
----------
subjects : list
	List with all used THINGS EEG2 subjects.
channels : str
	Used EEG channels.
model : str
	Name of the used encoding model.
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG

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
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--channels', type=str, default='all') # ['O', 'P', 'T', 'C', 'F', 'all']
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator/', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
correlation_averaged_repetitions = []
correlation_single_repetitions = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-eeg',
	'train_dataset-things_eeg_2', 'model-'+args.model, 'metadata')

for sub in args.subjects:
	file_name = 'metadata_subject-' + format(sub, '02') + '.npy'
	metadata = np.load(os.path.join(metadata_dir, file_name),
		allow_pickle=True).item()
	correlation_averaged_repetitions.append(
		metadata['encoding_models']['correlation_averaged_repetitions'])
	correlation_single_repetitions.append(
		metadata['encoding_models']['correlation_single_repetitions'])
	ch_names = metadata['eeg']['ch_names']
	times = metadata['eeg']['times']

correlation_averaged_repetitions = np.asarray(correlation_averaged_repetitions)
correlation_single_repetitions = np.asarray(correlation_single_repetitions)


# =============================================================================
# Channels selection
# =============================================================================
if args.channels != 'OP' and args.channels != 'all':
	kept_ch_names = []
	idx_ch = []
	for c, chan in enumerate(ch_names):
		if args.channels in chan:
			kept_ch_names.append(chan)
			idx_ch.append(c)
	idx_ch = np.asarray(idx_ch)
	ch_names_new = kept_ch_names
elif args.channels == 'OP':
	kept_ch_names = []
	idx_ch = []
	for c, chan in enumerate(ch_names):
		if 'O' in chan or 'P' in chan:
			kept_ch_names.append(chan)
			idx_ch.append(c)
	idx_ch = np.asarray(idx_ch)
	ch_names_new = kept_ch_names
elif args.channels == 'all':
	ch_names_new = ch_names
	idx_ch = np.arange(0, len(ch_names))

if args.channels == 'O':
	chan = 'Occipital'
elif args.channels == 'P':
	chan = 'Parietal'
elif args.channels == 'T':
	chan = 'Temporal'
elif args.channels == 'C':
	chan = 'Central'
elif args.channels == 'F':
	chan = 'Frontal'
elif args.channels == 'all':
	chan = 'All'

# Average the encoding accuracies across the selected channels
correlation_averaged_repetitions = np.mean(
	correlation_averaged_repetitions[:,idx_ch], 1)
correlation_single_repetitions = np.mean(
	correlation_single_repetitions[:,:,idx_ch], 2)


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
color_noise_ceiling = (150/255, 150/255, 150/255)


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
fig, axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))

for s, sub in enumerate(args.subjects):

	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
		linewidth=3, alpha=.5, label='_nolegend_')

	# Plot the correlation results (averaged repeats)
	axs[s].plot(times, correlation_averaged_repetitions[s], color=colors[0],
		linewidth=3)

	# Plot the correlation results (single repeats)
	for r in range(correlation_single_repetitions.shape[1]):
		if r == 0:
			axs[s].plot(times, correlation_single_repetitions[s,r], '--',
				color='k', linewidth=2, alpha=0.5)
		else:
			axs[s].plot(times, correlation_single_repetitions[s,r], '--',
				color='k', linewidth=2, alpha=0.5, label='_nolegend_')

	# x-axis parameters
	if s in [5, 6, 7, 8, 9]:
		axs[s].set_xlabel('Time (s)', fontsize=fontsize)
		xticks = [0, .1, .2, .3, .4, .5]
		xlabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
		plt.xticks(ticks=xticks, labels=xlabels)
	axs[s].set_xlim(left=min(times), right=max(times))

	# y-axis parameters
	if s in [0, 5]:
		axs[s].set_ylabel('Pearson\'s $r$', fontsize=fontsize)
		yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
		ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[s].set_ylim(bottom=-.075, top=1)

	# Title
	tit = chan + ' channels, subject ' + str(sub)
	axs[s].set_title(tit, fontsize=fontsize)

	# Legend
	if s in [0]:
		labels = ['Averaged repetitions', 'Single repetitions']
		axs[s].legend(labels, ncol=2, fontsize=fontsize, frameon=False,
			bbox_to_anchor=(1.5, -1.35))

# Save the figure
fig.savefig('encoding_accuracy_channels-' + args.channels + '.svg',
	bbox_inches='tight', transparent=True, format='svg')
fig.savefig('encoding_accuracy_channels-' + args.channels + '.png',
	dpi=300, bbox_inches='tight', transparent=True, format='png')
