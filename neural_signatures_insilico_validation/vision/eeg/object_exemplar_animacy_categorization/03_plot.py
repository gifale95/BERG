"""Plot the of object exemplar and animacy decoding accuracy of in silico EEG
responses.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : list
    List containing the EEG channel type(s) retained for the analyses.
    Possible values are: 'O' (occipital), 'P' (posterior), 'T' (temporal),
    'C' (central), 'F' (frontal).
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
parser.add_argument('--channels', default=['O', 'P'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_exemplar_animacy_categorization', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'object_exemplar_animacy_categorization', 'pairwise_decoding_results',
    'stats_'+'channels-'+''.join(args.channels)+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

decoding_exemplars = results['decoding_exemplars'] * 100
decoding_animacy = results['decoding_animacy'] * 100
ci_exemplars = results['ci_exemplars'] * 100
ci_animacy = results['ci_animacy'] * 100
peak_latency_diff = results['peak_latency_diff']
ci_peak_latency_diff = results['ci_peak_latency_diff']
pval_peak_latency_diff = results['pval_peak_latency_diff']
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
colors = [(169/255, 5/255, 3/255), (170/255, 118/255, 186/255)]


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1)) # type: ignore

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
plt.xticks(ticks=xticks, labels=xlabels) # type: ignore
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel('Decoding accuracy (%)', fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
axs[0].set_ylim(bottom=45, top=100)

# Legend
axs[0].legend(ncol=1, fontsize=fontsize, loc=1, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_accuray_channels-'+
    ''.join(args.channels)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Print the peak latency stats
# =============================================================================
print('>>> Mean peak latency diff (seconds): ' + \
    str(np.round(np.mean(peak_latency_diff), 3)))
print('>>> CI peak latency diff (seconds): ' + str(ci_peak_latency_diff))
print('>>> P-val peak latency diff: ' + str(pval_peak_latency_diff))