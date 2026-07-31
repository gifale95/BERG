"""Plot the encoding accuracy of the MEG-fMRI fusion encoding models.

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
    'invivo_things_meg_fmri_control', 'stats', 'stats.npy')

results = np.load(results_dir, allow_pickle=True).item()

times = results['times']
corr_tfmri_fmri = results['corr_tfmri_fmri']
ci_corr_tfmri_fmri = results['ci_corr_tfmri_fmri']
ci_corr_tfmri_fmri_peak_lat = results['ci_corr_tfmri_fmri_peak_lat']

# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'plots')
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


# =============================================================================
# Plot the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# Define the ROIs to plot
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA', 'PPA']

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
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
    alpha=.25, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Plot the correlation
    plt.plot(times, np.mean(corr_tfmri_fmri[roi], 0),
        color=colors[r], linewidth=2, label=roi)

    # Plot the CIs
    plt.fill_between(times, ci_corr_tfmri_fmri[roi][1],
        ci_corr_tfmri_fmri[roi][0], color=colors[r], alpha=.1)

    # Plot the peak time point
    peak = times[np.argmax(np.mean(corr_tfmri_fmri[roi], 0))]
    max_corr = max(np.mean(corr_tfmri_fmri[roi], 0))
    plt.scatter(peak, max_corr, color=colors[r], s=200, marker='o',
        edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
    ci_low = peak - ci_corr_tfmri_fmri_peak_lat[roi][0]
    ci_up = ci_corr_tfmri_fmri_peak_lat[roi][1] - peak
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    plt.errorbar(peak, max_corr, xerr=conf_int, fmt="none",
        ecolor='k', elinewidth=1, capsize=3)

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4]
ylabels = [0, 0.1, 0.2, 0.3, 0.4]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.08, top=.4)

# Legend
plt.legend(fontsize=20, loc=0, ncols=4, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'roi_correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()