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
parser.add_argument('--eeg_train_trials', default='all', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'stats',
    f'stats_eeg_train_trials-{args.eeg_train_trials}.npy')

results = np.load(results_dir, allow_pickle=True).item()

metadata = results['metadata']
times = np.round(results['times_new'], 3)
corr_tfmri_fmri = results['corr_tfmri_fmri']
corr_tfmri_fmri_roi = results['corr_tfmri_fmri_roi']
ci_corr_tfmri_fmri_roi = results['ci_corr_tfmri_fmri_roi']
ci_corr_tfmri_fmri_roi_peak_lat = results['ci_corr_tfmri_fmri_roi_peak_lat']
between_subj_pval = results['between_subj_pval']

# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'plots',
    f'eeg_train_trials-{args.eeg_train_trials}',
    'encoding_accuracy_surfaceplots')
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

# Time points to plot
plot_times = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Loop over EEG time points
for time in tqdm(plot_times):

    # Get the time index
    t = np.where(times == time)[0][0]

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(corr_tfmri_fmri[:,0,:,t], 0),
        np.nanmean(corr_tfmri_fmri[:,1,:,t], 0))

    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject,
        cmap='magma',
        vmin=0,
        vmax=0.8,
        )

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=False,
        with_rois=True,
        roi_list=['E', 'I', 'V', 'L', 'D'], # 'Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal''E', 'I', 'V', 'L', 'D'
        linewidth=5,
        linecolor=(1, 1, 1),
        with_labels=False,
        labelsize=60,
        curvature_brightness=0.4,
        with_colorbar=False
        )

    # Add title
    title = f'Time (ms): {np.round(time*1000)}'
    plt.title(title, fontsize=fontsize)

    # Save the plot
    time = int(np.round(time*1000))
    plot_file = os.path.join(save_dir, f'correlation_time-{time:04d}ms.png')
    plt.savefig(plot_file, dpi=150, transparent=True, bbox_inches='tight', format='png')
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
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'PPA']

# Print the peak time point of each ROI
for roi in rois:
    peak = times[np.argmax(np.mean(corr_tfmri_fmri_roi[roi], 0))]
    print(f'Peak time point of {roi}: {np.round(peak*1000)} ms')

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(rois))

# Format the significance for plotting. The time axis is upsampled with the
# midpoints between consecutive time points: the curves and CIs then change
# color halfway between time points, and isolated significant time points
# still get a line segment (a lone non-NaN point between NaNs is not drawn)
p_threshold = 0.01
times_up = np.sort(np.concatenate((times, (times[:-1] + times[1:]) / 2)))
# Shape: (2*n_times-1)
sig = {}
corr_sig = {}
corr_nonsig = {}
ci_sig = {}
ci_nonsig = {}
for roi in rois:
    sig[roi] = between_subj_pval[roi] < p_threshold
    corr_interp = np.interp(times_up, times,
        np.mean(corr_tfmri_fmri_roi[roi], 0))
    # Shape: (2*n_times-1)
    ci_interp = np.array([np.interp(times_up, times, ci)
        for ci in ci_corr_tfmri_fmri_roi[roi]])
    # Shape: (2, 2*n_times-1)
    # Significance of the time points on either side of each upsampled point
    # (for original time points, both sides are the time point itself)
    sig_rep = np.repeat(sig[roi], 2)
    sig_left = sig_rep[:-1]
    sig_right = sig_rep[1:]
    # Midpoints between a significant and a non-significant time point belong
    # to both parts, so that the significant and non-significant parts join
    # without gaps
    mask_sig = sig_left | sig_right
    mask_nonsig = ~sig_left | ~sig_right
    corr_sig[roi] = np.where(mask_sig, corr_interp, np.nan)
    corr_nonsig[roi] = np.where(mask_nonsig, corr_interp, np.nan)
    ci_sig[roi] = np.where(mask_sig, ci_interp, np.nan)
    ci_nonsig[roi] = np.where(mask_nonsig, ci_interp, np.nan)

# Create the figure
fig = plt.figure(figsize=(25, 3))

# Plot the chance dashed line
plt.plot([-10, 10], [0, 0], 'k--', linewidth=3, alpha=.25, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Plot the correlation: gray for non-significant time points, ROI color
    # for significant ones. zorder=2.5 draws the colored segments above the
    # gray segments of all ROIs (default zorder=2), but below the peak
    # markers (zorder=3)
    plt.plot(times_up, corr_nonsig[roi], color=colors[r], linewidth=3,
        alpha=0.25, label='_nolegend_')
    plt.plot(times_up, corr_sig[roi], color=colors[r], linewidth=3,
        zorder=2.5, label=roi)

    # Plot the CIs: gray for non-significant time points, ROI color for
    # significant ones (linewidth=0 removes the band outlines, which would
    # otherwise draw a vertical line at every significance transition)
    plt.fill_between(times_up, ci_nonsig[roi][1], ci_nonsig[roi][0],
        color=colors[r], alpha=.1, linewidth=0)
    plt.fill_between(times_up, ci_sig[roi][1], ci_sig[roi][0],
        color=colors[r], alpha=.2, linewidth=0)

    # Plot the peak time point
    peak = times[np.argmax(np.mean(corr_tfmri_fmri_roi[roi], 0))]
    max_corr = max(np.mean(corr_tfmri_fmri_roi[roi], 0))
    plt.scatter(peak, max_corr, color=colors[r], s=400, marker='o',
        edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
    ci_low = peak - ci_corr_tfmri_fmri_roi_peak_lat[roi][0]
    ci_up = ci_corr_tfmri_fmri_roi_peak_lat[roi][1] - peak
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    # plt.errorbar(peak, max_corr, xerr=conf_int, fmt="none",
    #     ecolor='k', elinewidth=1, capsize=3)

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
# xticks = [0, .1, .2, .3, .4]
# xlabels = [0, 100, 200, 300, 400]
xticks = [0, .05, .1, .15, .2, .25, .3, .35, .4]
xlabels = [0, 50, 100, 150, 200, 250, 300, 350, 400]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.4, 0.8]
ylabels = [0, 0.4, 0.8]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.1, top=.8)

# Legend
plt.legend(fontsize=fontsize, loc=4, ncols=2, frameon=False)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'plots',
    f'eeg_train_trials-{args.eeg_train_trials}')
file_name = os.path.join(save_dir, 'roi_correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()