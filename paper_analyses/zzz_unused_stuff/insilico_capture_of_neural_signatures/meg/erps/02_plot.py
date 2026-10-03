"""Plot the MEG ERPs.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of MEG subject identifiers.
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
parser.add_argument('--encoding_model', type=str, default='meg-things_meg_1-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'erps', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the ERP results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'meg', 'erps', 'erps',
    args.encoding_model, 'erps.npy')

results = np.load(results_dir, allow_pickle=True).item()

insilico_erps_chan_avg = results['insilico_erps_chan_avg']
invivo_erps_chan_avg = results['invivo_erps_chan_avg']
ci_insilico_erps_chan_avg = results['ci_insilico_erps_chan_avg']
ci_invivo_erps_chan_avg = results['ci_invivo_erps_chan_avg']
corr_erps_chan_avg = results['corr_erps_chan_avg']
pval_corr_erps_chan_avg = results['pval_corr_erps_chan_avg']
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
# Plot the ERPs
# =============================================================================
fig = plt.figure(figsize=(10, 7.5))

# Plot the stimulus onset dashed line
plt.plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.5, label='_nolegend_')

# Loop across channel groups
chan_groups = ['Occipital', 'Parietal', 'Temporal', 'Central', 'Frontal']
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
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .6]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel('fT', fontsize=fontsize)
yticks = [-.5, 0, .5]
ylabels = [-.5, 0, .5]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.65, top=.65)

# Legend
plt.legend(ncol=1, fontsize=10, loc=0, ncols=5, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'erps.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)