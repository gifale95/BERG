"""Plot the vertex-mean t-fMRI responses of high-level visual cortex ROIs for
images of different categories.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2], type=int)
parser.add_argument('--eeg_reps', default='single', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge',
    'hvc_selectivity', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge', 'hvc_selectivity',
    'stats', f'eeg_reps-{args.eeg_reps}', 'stats.npy')

data = np.load(data_dir, allow_pickle=True).item()

tfmri_roi_avg = data['tfmri_roi_avg']
sig_cat_diff = data['sig_cat_diff']
ci_tfmri_roi_avg = data['ci_tfmri_roi_avg']
times = data['times']

# DELETE # !!!
time_range = np.arange(20, 50)
times = times[time_range]
# DELETE # !!!


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
    (40/255, 65/255, 150/255),
    (0/255, 0/255, 0/255)
    ]


# =============================================================================
# Plot the vertex-mean responses of each ROI
# =============================================================================
categories = ['Bodies', 'Faces', 'Objects', 'Scenes']
rois = ['EBA', 'FBA', 'FFA', 'OFA', 'PPA', 'OPA', 'RSC']

# Loop across ROIs
for r, roi in enumerate(rois):

    # Create the plot
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(13, 7))
    axs = np.reshape(axs, (-1))

    # Plot the stimulus onset dashed lines
    axs[0].plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.5,
        label='_nolegend_')

    # Loop across categories
    for c, cat in enumerate(categories):

        # Plot the univariate response scores
        axs[0].plot(times, np.mean(tfmri_roi_avg[roi+'_'+cat], 0),
            color=colors[c], linewidth=2, label=cat)

        # Confidence intervals
        axs[0].fill_between(times, ci_tfmri_roi_avg[roi+'_'+cat][1],
            ci_tfmri_roi_avg[roi+'_'+cat][0], color=colors[c], alpha=.1)

    # x-axis parameters
    axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [0, .1, .2, .3, .4, .5]
    xlabels = [0, 100, 200, 300, 400, 500]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[0].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[0].set_ylabel("Univariate response ($z$-scored)", fontsize=fontsize)
    yticks = [-1, -0.5, 0, 0.5, 1]
    ylabels = ['-1', '-0.5', '0', '0.5', '1']
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=-.5, top=.25)

    # Title
    plt.title(roi)

    # Legend
    plt.legend(loc=0, ncol=1, fontsize=15, frameon=False)

    # Save the figure
    file_name = os.path.join(save_dir,
        f'univariate_resposes_{roi}_eeg_reps-{args.eeg_reps}.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
        format='svg')
    plt.close(fig)