"""Plot the encoding models' prediction accuracy for the test stimuli.

Parameters
----------
subjects : list
    List with all used THINGS MEG1 subjects.
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
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4])
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
correlation_single_splits_avg_rep = []
correlation_single_splits_single_rep = []
correlation_all_splits = []
noise_ceiling = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', 'model-vit_b_32', 'metadata')

for sub in args.subjects:
    file_name = 'metadata_P' + str(sub) + '.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name),
        allow_pickle=True).item()
    correlation_single_splits_avg_rep.append(
        metadata['encoding_accuracy_new']['correlation_single_splits_avg_rep'])
    correlation_single_splits_single_rep.append(
        metadata['encoding_accuracy_new']['correlation_single_splits_single_rep'])
    correlation_all_splits.append(
        metadata['encoding_accuracy_new']['correlation_all_splits'])
    noise_ceiling.append(
        metadata['encoding_accuracy_new']['noise_ceiling'])
    sensor_regions = metadata['sensors']['sensor_regions']
    times = metadata['meg']['times']
    del metadata

# Format the loaded data as numpy arrays, and convert the Pearson correlation
# scores to r2 scores
correlation_single_splits_avg_rep = np.asarray(
    correlation_single_splits_avg_rep) ** 2
correlation_single_splits_single_rep = np.asarray(
    correlation_single_splits_single_rep) ** 2
correlation_all_splits = np.asarray(correlation_all_splits) ** 2
noise_ceiling = np.asarray(noise_ceiling)


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 25
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
idx_sensor = np.where(sensor_regions == 'Occipital')[0]

plt.figure()

# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=1, alpha=.5, label='_nolegend_')

# Plot the correlation results
plt.plot(times, np.mean(correlation_single_splits_avg_rep, 0), color='b',
    linewidth=1, label='correlation_single_splits_avg_rep')
plt.plot(times, np.mean(correlation_single_splits_single_rep, (0, 1)), color='o',
    linewidth=1, label='correlation_single_splits_single_rep')
plt.plot(times, np.mean(correlation_all_splits, 0), color='g',
    linewidth=1, label='correlation_all_splits')
plt.plot(times, np.mean(noise_ceiling, 0), '--k',
    linewidth=1, label='noise_ceiling')

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
yticks = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
ylabels = [0, 200, 400, 600, 800, 1000, 1200]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel('Explained variance ($r²$)', fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.075, top=1)

# Legend
plt.legend(ncol=1, fontsize=fontsize, frameon=False)
























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
    bbox_inches='tight', transparent=False, format='svg')
fig.savefig('encoding_accuracy_channels-' + args.channels + '.png',
    dpi=300, bbox_inches='tight', transparent=False, format='png')