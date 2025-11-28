"""Plot the ERPs for faces and objects.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
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
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'erps', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the ERP results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',  'erps',
    'erps', 'erps.npy')

results = np.load(results_dir, allow_pickle=True).item()

insilico_erps_chan_avg = results['insilico_erps_chan_avg']
invivo_erps_chan_avg = results['invivo_erps_chan_avg']
ci_insilico_erps_chan_avg = results['ci_insilico_erps_chan_avg']
ci_invivo_erps_chan_avg = results['ci_invivo_erps_chan_avg']
corr_erps_chan_avg = results['corr_erps_chan_avg']
pval_corr_erps_chan_avg = results['pval_corr_erps_chan_avg']
times = results['metadata'][0]['eeg']['times']


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
    (147/255, 112/255, 219/255),
    (186/255, 85/255, 211/255),
    (105/255, 105/255, 105/255),
    (169/255, 169/255, 169/255),
    (90/255, 130/255, 200/255),
    (40/255, 65/255, 150/255)
    ]


# =============================================================================
# Plot the ERPs
# =============================================================================
fig= plt.figure(figsize=(13, 7))

# Plot the stimulus onset dashed line
plt.plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.5, label='_nolegend_')

# Loop across channel groups
chan_groups = ['O', 'P', 'T', 'C', 'F']
for c, chan in enumerate(chan_groups):

    # Plot the ERPs
    plt.plot(times, np.mean(invivo_erps_chan_avg[chan], 0), color=colors[c],
        linewidth=2, label='In vivo - '+chan)
    plt.plot(times, np.mean(insilico_erps_chan_avg[chan], 0), '--',
        color=colors[c], linewidth=2, label='In silico - '+chan)

    # Plot the CIs
    plt.fill_between(times, ci_invivo_erps_chan_avg[chan][1],
        ci_invivo_erps_chan_avg[chan][0], color=colors[c], alpha=.1)
    plt.fill_between(times, ci_insilico_erps_chan_avg[chan][1],
        ci_insilico_erps_chan_avg[chan][0], color=colors[c], alpha=.1)

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels) # type: ignore
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel('μV', fontsize=fontsize)
yticks = [-.5, 0, .5]
ylabels = [-.5, 0, .5]
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
plt.ylim(bottom=-.5, top=.65)

# Legend
plt.legend(ncol=1, fontsize=fontsize, loc=0, ncols=5, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'erps.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')