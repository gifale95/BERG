"""Plot the encoding accuracy results of the THINGS EEG2 encoding models
trained with neural responses optionally denoised with PSN.

PSN GitHub: https://github.com/jacob-prince/PSN

Parameters
----------
subjects : list
    List of subject identifier sfor the THINGS EEG2 data. Valid subject
    identifiers are integers from 1 to 10.
psn_mode : int
    PSN mode, randing from 1 to 5.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--psn_mode', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg', 'plots')
os.makedirs(save_dir, exist_ok=True)

# =============================================================================
# Load the results, and reshape to (channels, times)
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg', 'test_encoding',
    f'psn_mode-{args.psn_mode}', 'test_encoding_stats.npy')
results = np.load(data_dir, allow_pickle=True).item()

ncsnr = results['ncsnr']
noise_ceiling = results['noise_ceiling']
correlation = results['correlation']
ci_ncsnr = results['ci_ncsnr']
ci_noise_ceiling = results['ci_noise_ceiling']
ci_correlation = results['ci_correlation']
ch_names = results['ch_names']
times = results['times']

n_sub = len(args.subjects)
n_chan = len(ch_names)
n_time = len(times)
for key in ncsnr.keys():
    ncsnr[key] = ncsnr[key].reshape(n_sub, n_chan, n_time)
    noise_ceiling[key] = noise_ceiling[key].reshape(n_sub, n_chan, n_time)
    ci_ncsnr[key] = ci_ncsnr[key].reshape(2, n_chan, n_time)
    ci_noise_ceiling[key] = ci_noise_ceiling[key].reshape(2, n_chan, n_time)

for key in correlation.keys():
    correlation[key] = correlation[key].reshape(n_sub, n_chan, n_time)
    ci_correlation[key] = ci_correlation[key].reshape(2, n_chan, n_time)


PLOT ERPS!!!!!


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
colors = [
    (31/255, 119/255, 180/255),
    (255/255, 127/255, 14/255),
    (44/255, 160/255, 44/255),
    (214/255, 39/255, 40/255),
    (148/255, 103/255, 189/255),
    (140/255, 86/255, 75/255),
    (227/255, 119/255, 194/255),
    (127/255, 127/255, 127/255)
    ]


# =============================================================================
# Plot the ncsnr
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')

# Plot the results
for i, (key, val) in enumerate(ncsnr.items()):
    axs[0].plot(times, np.mean(val, 0), color=colors[i],
        linewidth=2, label=key)

# Plot the confidence intervals
for i, (key, val) in enumerate(ci_ncsnr.items()):
    axs[0].fill_between(times, np.mean(val[1], 0),
        np.mean(val[0], 0), color=colors[i], alpha=.2,
        label='_nolegend_')

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
axs[0].set_xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("NCSNR", fontsize=fontsize)
yticks = [0, 0.5, 1]
ylabels = [0, 0.5, 1]
axs[0].set_yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=0, top=1)

# Legend
axs[0].legend(loc=2, ncol=1, fontsize=fontsize,
    bbox_to_anchor=(1.525, -6.5), frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'ncsnr_psn_mode-'+str(args.psn_mode)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the noise ceiling
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')

# Plot the results
for i, (key, val) in enumerate(noise_ceiling.items()):
        axs[0].plot(times, np.mean(val, 0), color=colors[i],
            linewidth=2, label=key)

# Plot the confidence intervals
for i, (key, val) in enumerate(ci_noise_ceiling.items()):
    axs[0].fill_between(times, np.mean(val[1], 0),
        np.mean(val[0], 0), color=colors[i], alpha=.2,
        label='_nolegend_')

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
axs[0].set_xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Noise ceiling", fontsize=fontsize)
yticks = [0, 0.5, 1]
ylabels = [0, 0.5, 1]
axs[0].set_yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=0, top=1)

axs[0].legend(loc=2, ncol=1, fontsize=fontsize,
    bbox_to_anchor=(1.525, -6.5), frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'noise_ceiling_psn_mode-'+str(args.psn_mode)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the correlation
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')

# Plot the results
for i, (key, val) in enumerate(correlation.items()):
    axs[0].plot(times, np.mean(val, 0), color=colors[i],
        linewidth=2, label=key)

# Plot the confidence intervals
for i, (key, val) in enumerate(ci_correlation.items()):
    axs[0].fill_between(times, np.mean(val[1], 0),
        np.mean(val[0], 0), color=colors[i], alpha=.2,
        label='_nolegend_')

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
axs[0].set_xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.5, 1]
ylabels = [0, 0.5, 1]
axs[0].set_yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=0, top=1)

axs[0].legend(loc=2, ncol=2, fontsize=fontsize,
    bbox_to_anchor=(1.525, -6.5), frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'correlation_psn_mode-'+str(args.psn_mode)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')