"""Plot the encoding accuracy of the EEG-fMRI fusion encoding models.

Parameters
----------
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'all' (all trials), 'even' (even trials), and 'odd' (odd
    trials).
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
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--eeg_train_trials', default='odd', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'stats',
    f'stats_eeg_train_trials-{args.eeg_train_trials}.npy')

results = np.load(results_dir, allow_pickle=True).item()

metadata = results['metadata']
times = results['times']
corr_tfmri_fmri = results['corr_tfmri_fmri']
corr_tfmri_fmri_roi = results['corr_tfmri_fmri_roi']
ci_corr_tfmri_fmri_roi = results['ci_corr_tfmri_fmri_roi']
ci_corr_tfmri_fmri_roi_peak_lat = results['ci_corr_tfmri_fmri_roi_peak_lat']

# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'plots', 'encoding_accuracy_surfaceplots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the encoding accuracy of the EEG-fMRI fusion encoding models on brain
# surfaces (subject-average)
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

# Loop over EEG time points
for t, time in enumerate(tqdm(times)):

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(corr_tfmri_fmri[:,0,:,t], 0),
        np.nanmean(corr_tfmri_fmri[:,1,:,t], 0))

    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject,
        cmap='afmhot',
        vmin=0,
        vmax=.5,
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
    file_name = (f'correlation_eeg_train_trials-{args.eeg_train_trials}_'
        f'time-{t:03d}.png')
    plot_file = os.path.join(save_dir, file_name)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()


# =============================================================================
# Plot the ROI-wise correlations between t-fMRI and in silico fMRI responses
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
rois = ['V1', 'V2', 'V3', 'hV4', 'ventral']

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(rois))

# Create the figure
fig = plt.figure(figsize=(10, 7.5))

# Plot the stimulus onset and chance dashed line
plt.plot([-1000, 1000], [0, 0], 'k--', [0, 0], [1, -1], 'k--', linewidth=2,
    alpha=.25, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Plot the correlation
    plt.plot(times, np.mean(corr_tfmri_fmri_roi[roi], 0),
        color=colors[r], linewidth=2, label=roi)

    # Plot the CIs
    plt.fill_between(times, ci_corr_tfmri_fmri_roi[roi][1],
        ci_corr_tfmri_fmri_roi[roi][0], color=colors[r], alpha=.1)

    # Plot the peak time point
    peak = times[np.argmax(np.mean(corr_tfmri_fmri_roi[roi], 0))]
    max_corr = max(np.mean(corr_tfmri_fmri_roi[roi], 0))
    plt.scatter(peak, max_corr, color=colors[r], s=200, marker='o',
        edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
    ci_low = peak - ci_corr_tfmri_fmri_roi_peak_lat[roi][0]
    ci_up = ci_corr_tfmri_fmri_roi_peak_lat[roi][1] - peak
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    plt.errorbar(peak, max_corr, xerr=conf_int, fmt="none",
        ecolor='k', elinewidth=1, capsize=3)

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [-100, 0, 100, 200, 300, 400, 500, 600]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3]
ylabels = [0, 0.1, 0.2, 0.3]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.025, top=.3)

# Legend
plt.legend(fontsize=fontsize, loc=4, ncols=2, frameon=False)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'plots')
file_name = os.path.join(save_dir,
    f'roi_correlation_eeg_train_trials-{args.eeg_train_trials}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()