"""Plot the decoding accuracy for the in silico EEG responses.

Parameters
----------
all_subjects : list of int
	List with all subject numbers.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'], occipital/parital ['OP'], or all ['all']
	channels.
nest_dir : str
	Neural encoding simulation toolkit directory.

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
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--channels', type=str, default='OP')
parser.add_argument('--nest_dir', default='/home/ale/aaa_stuff/PhD/projects/neural_encoding_simulation_toolkit', type=str)
#parser.add_argument('--nest_dir', default='/home/ale/scratch/projects/neural_encoding_simulation_toolkit', type=str)
args = parser.parse_args()


# =============================================================================
# Load the decoding results
# =============================================================================
results_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
	'pairwise_decoding_eeg', 'stats', 'channels-'+args.channels, 'stats.npy')

results = np.load(results_dir, allow_pickle=True).item()

decoding_exemplars = results['pairwise_decoding_exemplars'] * 100
decoding_animacy = results['pairwise_decoding_animacy'] * 100
ci_exemplars = results['ci_exemplars'] * 100
ci_animacy = results['ci_animacy'] * 100
times = results['times']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
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
colors = [(169/255, 5/255, 3/255), (228/255, 145/255, 142/255)]


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
	figsize=(13, 7))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
	linewidth=3, alpha=.5, label='_nolegend_')

# Plot the decoding subject-average results
# Exemplar decoding
label = 'Exemplar'
peak = times[np.argsort(np.mean(decoding_exemplars, 0))[::-1][0]]
max_dec = max(np.mean(decoding_exemplars, 0))
axs[0].plot([peak, peak], [max_dec, -100], '--', linewidth=3, color=colors[0],
	alpha=.5)
axs[0].plot(times, np.mean(decoding_exemplars, 0), color=colors[0], linewidth=3,
	label=label)
axs[0].fill_between(times, ci_exemplars[1], ci_exemplars[0], color=colors[0],
	alpha=.2)
# Animacy decoding
label = 'Animacy'
peak = times[np.argsort(np.mean(decoding_animacy, 0))[::-1][0]]
max_dec = max(np.mean(decoding_animacy, 0))
axs[0].plot([peak, peak], [max_dec, -100], '--', linewidth=3, color=colors[1],
	alpha=.5)
axs[0].plot(times, np.mean(decoding_animacy, 0), color=colors[1], linewidth=3,
	label=label)
axs[0].fill_between(times, ci_animacy[1], ci_animacy[0], color=colors[1],
	alpha=.2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel('Decoding accuracy (%)', fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=45, top=100)

# Legend
axs[0].legend(ncol=1, fontsize=fontsize, loc=1, frameon=False)

# Save the figure
file_name = 'decoding_accuray_in_silico_eeg_channels-' + args.channels + '.png'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='png')
file_name = 'decoding_accuray_in_silico_eeg_channels-' + args.channels + '.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')