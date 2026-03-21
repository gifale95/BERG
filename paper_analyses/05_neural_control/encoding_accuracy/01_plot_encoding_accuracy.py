"""Plot the encoding accuracy of BERG's encoding models trained on TVSD.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico neural
    responses.
subjects : list
    The subject identifiers for the monkey encoding models. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--n_iter', default=10000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Get the encoding accuracy for each ROI
# =============================================================================
# Empty result variables
correlation = {}

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Get the metadata
    metadata = berg.get_model_metadata(
        args.encoding_model,
        subject=sub
    )

    # Extract the encoding accuracy
    corr = metadata['encoding_model']['correlation_results']
    roi_assignments = metadata['roi']['roi_assignments']
    times = metadata['utah_array']['times']

    # Average the encoding accuracy across electrodes from the same ROI
    rois = ['V1', 'V4', 'IT']
    for r, roi in enumerate(rois):
        if s == 0:
            correlation[roi] = []
        idx = np.where(roi_assignments == r)[0]
        correlation[roi].append(np.mean(corr[idx], 0))


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

# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
# Plot colors
colors = sample_cmap(len(rois))

# Plot the encoding accuracy
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
    figsize=(20, 7.5))
axs = np.reshape(axs, (-1))

# Loop across monkeys
for s, sub in enumerate(args.subjects):
    
    # Plot the chance and stimulus onset dashed lines
    axs[s].plot([-1000, 1000], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')

    # Loop across ROIs
    for r, key in enumerate(rois):
        # Plot the encoding accuracy
        axs[s].plot(times, correlation[key][s], color=colors[r], linewidth=2,
            label=key)
        # Title
        axs[s].set_title(f'Monkey {sub}', fontsize=fontsize)

    # x-axis parameters
    axs[s].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-100, -50, 0, 50, 100, 150, 199]
    xlabels = [-100, -50, 0, 50, 100, 150, 200]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[s].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    if s == 0:
        axs[s].set_ylabel("Pearson's $r$", fontsize=fontsize)
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.yticks(ticks=yticks, labels=ylabels)
        axs[s].set_ylim(bottom=-.05, top=0.7)

# Legend
axs[0].legend(fontsize=fontsize, ncol=1, loc=0, frameon=False)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'neural_control', 'encoding_accuracy',
    'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)
file_name = os.path.join(save_dir, 'encoding_accuracy.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()