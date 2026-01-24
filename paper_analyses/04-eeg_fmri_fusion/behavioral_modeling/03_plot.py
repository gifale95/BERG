"""Plot RSA results between t-fMRI responses and behavioral embeddings.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
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


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling', 'plots', 'rsa_surfaceplots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling', 'stats', 'stats.npy')

results = np.load(data_dir, allow_pickle=True).item()

rsa = results['rsa']
rsa_roi = results['rsa_roi']
ci_rsa_roi = results['ci_rsa_roi']
ci_rsa_roi_peak_lat = results['ci_rsa_roi_peak_lat']
times = results['times']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 40
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'


# =============================================================================
# Plot the RSA results
# =============================================================================
# Loop over EEG time points
for t, time in enumerate(tqdm(times)):

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(rsa[:,0,:,t], 0), np.nanmean(rsa[:,1,:,t], 0))
    
    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject,
        cmap='afmhot',
        vmin=0,
        vmax=0.3,
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
    title = f'Time (ms): {np.round(time*1000)}'
    plt.title(title, fontsize=fontsize)

    # Save the plot
    plot_file = os.path.join(save_dir, f'rsa_time-{t:03d}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()


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
rois = ['early', 'intermediate', 'ventral', 'lateral', 'parietal']

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
    alpha=.5, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Plot the correlation
    plt.plot(times, np.mean(rsa_roi[roi], 0),
        color=colors[r], linewidth=2, label=roi)

    # Plot the CIs
    plt.fill_between(times, ci_rsa_roi[roi][1],
        ci_rsa_roi[roi][0], color=colors[r], alpha=.1)

    # Plot the peak time point
    # peak = times[np.argmax(np.mean(rsa_roi[roi], 0))]
    # max_corr = max(np.mean(rsa_roi[roi], 0))
    # plt.scatter(peak, max_corr, color=colors[r], s=200, marker='o',
    #     edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
    # ci_low = peak - ci_rsa_roi_peak_lat[roi][0]
    # ci_up = ci_rsa_roi_peak_lat[roi][1] - peak
    # conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    # plt.errorbar(peak, max_corr, xerr=conf_int, fmt="none",
    #     ecolor='k', elinewidth=1, capsize=3)

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
plt.ylim(bottom=-.03, top=.25)

# Legend
plt.legend(fontsize=15, loc=0, ncols=3, frameon=False)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling', 'plots')
file_name = os.path.join(save_dir, 'roi_rsa.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()