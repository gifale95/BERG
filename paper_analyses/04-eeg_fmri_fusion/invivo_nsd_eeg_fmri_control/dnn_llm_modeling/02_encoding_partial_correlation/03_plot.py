"""Plot the encoding-based variance partitioning results between t-fMRI
responses and vision DNN features or LLM embeddings.

Parameters
----------
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import cortex
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Plot <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding_partial_correlation', 'stats',
    'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
times = results['times']

# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding_partial_correlation', 'plots',
    'partial_correlation_surfaceplots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the partial correlation results on brain surfaces (subject-average)
# =============================================================================
# Plot parameters
fontsize = 40
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'

# Loop over results types and EEG time points
for key, val in tqdm(results['partial_correlation'].items()):
    for t, time in enumerate(times):

        # Average the results across subjects, and append them across left and
        # right hemishperes
        data = np.append(np.nanmean(val[:,0,:,t], 0),
            np.nanmean(val[:,1,:,t], 0))

        # Create the flat brain surface
        vertex_data = cortex.Vertex(
            data,
            subject,
            cmap='RdGy_r', # !!! 'afmhot'
            vmin=-0.9, # !!!
            vmax=0.9, # !!!
            with_colorbar=True)

        # Plot the flat brain surface
        fig = cortex.quickshow(
            vertex_data,
            #height=2000, # Increase resolution of map and ROI contours
            with_curvature=True,
            with_rois=True,
            roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
            linewidth=3,
            linecolor=(1, 1, 1),
            with_labels=True,
            labelsize=25,
            curvature_brightness=0.4,
            with_colorbar=True
            )

        # Add title
        title = f'Time (ms): {time}'
        plt.title(title, fontsize=fontsize)

        # Save the plot
        file_name = f'{key}_time-{t:03d}.png'
        plot_file = os.path.join(save_dir, file_name)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
        plt.close()


# =============================================================================
# Plot the ROI-wise partial correlations between t-fMRI and in silico fMRI
# responses
# =============================================================================
# Plot parameters
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

# Define the ROIs to plot
rois = ['V1', 'V2', 'V3', 'hV4', 'ventral', 'lateral', 'parietal']

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(rois))

# Create the figure
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 10)) # (10, 7.5) # !!!
axs = np.reshape(axs, -1)

# Loop over result types
for i, (key, val) in enumerate(results['partial_correlation_roi'].items()):

    # Plot the stimulus onset and chance dashed line
    axs[i].plot([0, 0], [-1, 1], 'k--', [-1000, 1000], [0, 0], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')

    # Loop across ROIs
    for r, roi in enumerate(rois):

        # Plot the correlation
        axs[i].plot(times, np.mean(val[roi], 0), color=colors[r], linewidth=2,
            label=roi)

        # Plot the CIs
        axs[i].fill_between(times,
            results['ci_partial_correlation_roi'][key][roi][1],
            results['ci_partial_correlation_roi'][key][roi][0],
            color=colors[r], alpha=.1)

        # Plot the peak time point
        peak = times[np.argmax(np.mean(val[roi], 0))]
        max_corr = max(np.mean(val[roi], 0))
        axs[i].scatter(peak, max_corr, color=colors[r], s=200, marker='o',
            edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
        ci_low = peak - \
            results['ci_partial_correlation_roi_peak_lat'][key][roi][0]
        ci_up = \
            results['ci_partial_correlation_roi_peak_lat'][key][roi][1] - peak
        conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
        axs[i].errorbar(peak, max_corr, xerr=conf_int, fmt="none",
            ecolor='k', elinewidth=1, capsize=3)

    # x-axis parameters
    if i in [2, 3]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
        xticks = [-100, 0, 100, 200, 300, 400, 500, 600]
        xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
        axs[i].set_xticks(ticks=xticks, labels=xlabels)
        axs[i].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    if i in [0, 2]:
        axs[i].set_ylabel("Pearson's $r$", fontsize=fontsize)
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        axs[i].set_yticks(ticks=yticks, labels=ylabels)
        axs[i].set_ylim(bottom=-.05, top=0.9)

    # Legend
    if i == 0:
        axs[i].legend(fontsize=fontsize, loc=4, ncols=2, frameon=False)

    # Title
    axs[i].set_title(key, fontsize=fontsize)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'dnn_llm_modeling', 'encoding_partial_correlation', 'plots')
file_name = os.path.join(save_dir, 'partial_correlation_roi.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()