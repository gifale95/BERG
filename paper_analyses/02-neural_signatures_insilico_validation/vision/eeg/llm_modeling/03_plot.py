"""Plot the RSA scores between in silico EEG responses and LLM embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
channels : str
    String containing the EEG channel type retained for the analyses.
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
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O-P', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'llm_modeling', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg', 'llm_modeling',
    'stats', args.encoding_model, 'stats_channels-'+args.channels+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

rsa = results['rsa']
ci_rsa = results['ci_rsa']
ci_peak_latency_ci_rsa = results['ci_peak_latency_ci_rsa']
decoding = results['decoding']
ci_decoding = results['ci_decoding']
ci_peak_latency_ci_decoding = results['ci_peak_latency_ci_decoding']
times = results['times']


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
# Plot the EEG pairwise decoding results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the RSA subject-average results
axs[0].plot(times, np.mean(decoding, 0), color='k', linewidth=2)

# Plot the peak time point
peak = times[np.argmax(np.mean(decoding, 0))]
max_dec = max(np.mean(decoding, 0))
axs[0].scatter(peak, max_dec, color='k', s=200, marker='o', edgecolors='k',
    linewidths=1, zorder=3)
ci_low = peak - ci_peak_latency_ci_decoding[0]
ci_up = ci_peak_latency_ci_decoding[1] - peak
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
axs[0].errorbar(peak, max_dec, xerr=conf_int, fmt="none", ecolor='k',
    elinewidth=1, capsize=3)

# Plot the confidence intervals
axs[0].fill_between(times, ci_decoding[1], ci_decoding[0], color='k', alpha=.1)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=47, top=100)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_channels-'+args.channels+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the RSA results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the RSA subject-average results
axs[0].plot(times, np.mean(rsa, 0), color='k', linewidth=2)

# Plot the peak time point
peak = times[np.argmax(np.mean(rsa, 0))]
max_rsa = max(np.mean(rsa, 0))
axs[0].scatter(peak, max_rsa, color='k', s=200, marker='o', edgecolors='k',
    linewidths=1, zorder=3)
ci_low = peak - ci_peak_latency_ci_rsa[0]
ci_up = ci_peak_latency_ci_rsa[1] - peak
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
axs[0].errorbar(peak, max_rsa, xerr=conf_int, fmt="none", ecolor='k',
    elinewidth=1, capsize=3)

# Plot the confidence intervals
axs[0].fill_between(times, ci_rsa[1], ci_rsa[0], color='k', alpha=.1)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.05, 0.1, 0.15, 0.2]
ylabels = [0, 0.05, 0.1, 0.15, 0.2]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-.02, top=.16)

# Save the figure
file_name = os.path.join(save_dir, 'rsa_channels-'+args.channels+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')