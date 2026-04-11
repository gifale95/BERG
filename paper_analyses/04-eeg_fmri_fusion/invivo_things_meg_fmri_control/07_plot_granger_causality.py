"""Plot the Granger Causality results computed on the t-fMRI responses.

Parameters
----------
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'stats_granger_causality', 'stats.npy')

results = np.load(results_dir, allow_pickle=True).item()

times = (results['times'] * 1000).astype(int) # Convert to ms
gc_scores = results['gc_scores']
ci_gc_scores = results['ci_gc_scores']

# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'granger_causality', 'plots')
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
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(139/255, 0/255, 0/255), (0/255, 115/255, 155/255),
    (0/255, 0/255, 0/255)]


# =============================================================================
# Plot the Granger Causality scores between t-fMRI ROIs
# =============================================================================
fig, axs = plt.subplots(2, 5, sharex=True, sharey=False, figsize=(50, 30)) # (10, 7.5)
axs = np.reshape(axs, (-1))

# Loop across ROI pairwise combinations
roi_pairs = [
    ['V1', 'V2'],
    ['V1', 'V3'],
    ['V1', 'hV4'],
    ['V1', 'IT'],
    ['V2', 'V3'],
    ['V2', 'hV4'],
    ['V2', 'IT'],
    ['V3', 'hV4'],
    ['V3', 'IT'],
    ['hV4', 'IT']
]
for i, roi_pair in enumerate(roi_pairs):

    # Plot the chance dashed line
    axs[i].plot([-1000, 1000], [0, 0], 'k--', linewidth=2, alpha=.25,
        label='_nolegend_')

    # Plot the correlation
    key_1 = f'{roi_pair[0]}_to_{roi_pair[1]}'
    key_2 = f'{roi_pair[1]}_to_{roi_pair[0]}'
    axs[i].plot(times,
        np.mean(gc_scores[key_1], 0),
        color=colors[0], linewidth=2, label=key_1)
    axs[i].plot(times,
        np.mean(gc_scores[key_2], 0),
        color=colors[1], linewidth=2, label=key_2)

    # Plot the CIs
    axs[i].fill_between(times, ci_gc_scores[key_1][1],
        ci_gc_scores[key_1][0], color=colors[0], alpha=.1)
    axs[i].fill_between(times, ci_gc_scores[key_2][1],
        ci_gc_scores[key_2][0], color=colors[1], alpha=.1)

    # Title
    title = f'{roi_pair[0]} vs. {roi_pair[1]}'
    axs[i].set_title(title, fontsize=fontsize)

    # x-axis parameters
    if i in [5, 6, 7, 8, 9]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
        # xticks = [-100, -50, 0, 50, 100, 150, 199]
        # xlabels = [-100, -50, 0, 50, 100, 150, 200]
        # axs[i].set_xticks(ticks=xticks, labels=xlabels)
        axs[i].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    if i in [0, 5]:
        axs[i].set_ylabel('Granger Causality', fontsize=fontsize)
        # yticks = [10, 15, 20, 25, 30]
        # ylabels = [10, 15, 20, 25, 30]
        # axs[i].set_yticks(ticks=yticks, labels=ylabels)
        # axs[i].set_ylim(bottom=gc_min, top=gc_max)

    # Legend
    axs[i].legend(ncol=1, fontsize=fontsize, loc=0, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'granger_causality.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()