"""Plot the RSA scores between in silico EEG responses and behavioral
embeddings.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : list
    List containing the EEG channel type(s) retained for the analyses.
    Possible values are: 'O' (occipital), 'P' (posterior), 'T' (temporal),
    'C' (central), 'F' (frontal). Alternatively, the list can also contain the
    names of the individual channels used.
berg_dir : str
    Directory of the BERG.

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
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--psn_mode', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'behavioral_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'behavioral_modeling', 'stats', 'stats_psn_mode-'+str(args.psn_mode)+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

decoding = results['decoding']
ci_decoding = results['ci_decoding']
sig_decoding = results['sig_decoding']
rsa = results['rsa']
ci_rsa = results['ci_rsa']
sig_rsa = results['sig_rsa']
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
colors = [
    (103/255, 78/255, 167/255),
    (166/255, 77/255, 121/255),
    (105/255, 105/255, 105/255),
    (169/255, 169/255, 169/255),
    (100/255, 149/255, 237/255),
    (90/255, 130/255, 200/255),
    (40/255, 65/255, 150/255)
    ]


# =============================================================================
# Plot the decoding results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1)) # type: ignore

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=3, alpha=.5, label='_nolegend_')

# Loop across EEG data types
for c, key in enumerate(decoding.keys()):

    # Plot the subject-average results
    axs[0].plot(times, np.mean(decoding[key], 0), color=colors[c], linewidth=2,
        label=key)

    # Plot the confidence intervals
    axs[0].fill_between(times, ci_decoding[key][1], ci_decoding[key][0],
        color=colors[c], alpha=.2)

    # Plot the significance time points
    # sig = np.empty(len(times))
    # sig[:] = np.nan
    # sig[sig_decoding[key]] = 50 - 0.75 * c
    # plt.scatter(times, sig, s=100, color=colors[c])

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=45, top=100)

# Legend
axs[0].legend(loc=0, ncol=1, fontsize=20, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_psn_mode-'+str(args.psn_mode)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the RSA results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1)) # type: ignore

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=3, alpha=.5, label='_nolegend_')

# Loop across EEG data types
for c, key in enumerate(rsa.keys()):

    # Plot the subject-average results
    axs[0].plot(times, np.mean(rsa[key], 0), color=colors[c], linewidth=2,
        label=key)

    # Plot the confidence intervals
    axs[0].fill_between(times, ci_rsa[key][1], ci_rsa[key][0],
        color=colors[c], alpha=.2)

    # Plot the significance time points
    sig = np.empty(len(times))
    sig[:] = np.nan
    sig[sig_rsa[key]] = -.015 - 0.2 * c
    plt.scatter(times, sig, s=100, color=colors[c])

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("RSA (Pearson's $r$)", fontsize=fontsize)
yticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
ylabels = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-.03, top=.25)

# Legend
axs[0].legend(loc=0, ncol=1, fontsize=20, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'rsa_psn_mode-'+str(args.psn_mode)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')







fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1)) # type: ignore

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=3, alpha=.5, label='_nolegend_')

# Loop across EEG data types
for c, key in enumerate(['insilico_eeg_vtr-0_ste-1', 'insilico_eeg_vtr-1_ste-0']):

    # Plot the subject-average results
    axs[0].plot(times, np.mean(decoding[key], 0), color=colors[c], linewidth=2,
        label=key)

    # Plot the confidence intervals
    axs[0].fill_between(times, ci_decoding[key][1], ci_decoding[key][0],
        color=colors[c], alpha=.2)

    # Plot the significance time points
    # sig = np.empty(len(times))
    # sig[:] = np.nan
    # sig[sig_decoding[key]] = 50 - 0.75 * c
    # plt.scatter(times, sig, s=100, color=colors[c])

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=45, top=100)

# Legend
axs[0].legend(loc=0, ncol=1, fontsize=20, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_psn_mode-'+str(args.psn_mode)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')