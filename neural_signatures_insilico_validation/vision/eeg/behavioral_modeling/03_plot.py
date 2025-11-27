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
parser.add_argument('--channels', default=['O', 'P'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'behavioral_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'behavioral_modeling', 'stats', 'stats_'+'channels-'+
    '-'.join(args.channels)+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

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
colors = [(139/255, 0/255, 0/255), (166/255, 77/255, 121/255),
    (103/255, 78/255, 167/255)]


# =============================================================================
# Plot the RSA results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1)) # type: ignore

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=3, alpha=.5, label='_nolegend_')

# Plot the RSA subject-average results
axs[0].plot(times, np.mean(rsa, 0), color=colors[0], linewidth=3)

# Plot the peak time point
peak = times[np.argsort(np.mean(rsa, 0))[::-1][0]]
text = str(int(peak * 1000)) + ' ms'
max_rsa = max(np.mean(rsa, 0))
axs[0].text(peak, max_rsa+0.015, text, color='k', ha='center')
#axs[0].plot([peak, peak], [max_rsa, -100], '--', linewidth=3, color=colors[0],
#    alpha=.5)

# Plot the confidence intervals
axs[0].fill_between(times, ci_rsa[1], ci_rsa[0], color=colors[0], alpha=.2)

# Plot the significance markers
sig = np.empty(len(times))
sig[:] = np.nan
sig[sig_rsa] = -.025
plt.scatter(times, sig, s=100, color=colors[0])

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels) # type: ignore
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("RSA (Pearson's $r$)", fontsize=fontsize)
yticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
ylabels = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
axs[0].set_ylim(bottom=-.05, top=.25)

# Save the figure
file_name = os.path.join(save_dir, 'rsa_channels-'+'-'.join(args.channels)+
    '.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')