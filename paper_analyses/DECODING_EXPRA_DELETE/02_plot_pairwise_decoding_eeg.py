"""Plot the pairwise decoding of THINGS EEG2 test data with different amounts
of image conditions, repetitions, and with or without cross-validation.

Parameters
----------
sub : int
    Used subject.
project_dir : str
    Directory of the project folder.

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
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/decoding_expra', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 25
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 0
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 0
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


# =============================================================================
# Plot the decoding results (baseline) # !!!
# =============================================================================
# Load the data
data_dir = os.path.join(args.project_dir, 'pairwise_decoding')
file_name = (f'decoding_sub-{args.sub:02d}_conditions-200_'
            f'repeats-80_cv-1.npy')
data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
decoding = data['decoding'] * 100
times = data['times']

# Format the decoding results for plotting
idx_triu = np.tril_indices(decoding.shape[0], k=-1)
decoding = decoding[idx_triu]

# Create the figure
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the pairwise decoding
axs[0].plot(times, np.mean(decoding, 0), color='k', linewidth=2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .6]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=-0.1, right=0.6)

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=47, top=100)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_baseline.png')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='png')


# =============================================================================
# Plot the decoding results (low stimulus conditions)
# =============================================================================
# Load the data
data_dir = os.path.join(args.project_dir, 'pairwise_decoding')
file_name = (f'decoding_sub-{args.sub:02d}_conditions-010_'
            f'repeats-80_cv-1.npy')
data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
decoding = data['decoding'] * 100
times = data['times']

# Format the decoding results for plotting
idx_triu = np.tril_indices(decoding.shape[0], k=-1)
decoding = decoding[idx_triu]

# Create the figure
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the pairwise decoding
axs[0].plot(times, np.mean(decoding, 0), color='k', linewidth=2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .6]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=-0.1, right=0.6)

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=47, top=100)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_low_stimulus_conditions.png')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='png')


# =============================================================================
# Plot the decoding results (low stimulus repeats) # !!!
# =============================================================================
# Load the data
data_dir = os.path.join(args.project_dir, 'pairwise_decoding')
file_name = (f'decoding_sub-{args.sub:02d}_conditions-200_'
            f'repeats-08_cv-1.npy')
data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
decoding = data['decoding'] * 100
times = data['times']

# Format the decoding results for plotting
idx_triu = np.tril_indices(decoding.shape[0], k=-1)
decoding = decoding[idx_triu]

# Create the figure
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the pairwise decoding
axs[0].plot(times, np.mean(decoding, 0), color='k', linewidth=2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .6]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=-0.1, right=0.6)

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=47, top=100)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_low_stimulus_repeats.png')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='png')


# =============================================================================
# Plot the decoding results (no CV)
# =============================================================================
# Load the data
data_dir = os.path.join(args.project_dir, 'pairwise_decoding')
file_name = (f'decoding_sub-{args.sub:02d}_conditions-200_'
            f'repeats-80_cv-0.npy')
data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
decoding = data['decoding'] * 100
times = data['times']

# Format the decoding results for plotting
idx_triu = np.tril_indices(decoding.shape[0], k=-1)
decoding = decoding[idx_triu]

# Create the figure
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the pairwise decoding
axs[0].plot(times, np.mean(decoding, 0), color='k', linewidth=2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .6]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=-0.1, right=0.6)

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=47, top=100)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_no_cv.png')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='png')