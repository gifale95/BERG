"""Plot the encoding accuracy of the EEG-fMRI fusion encoding models.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemisphere : list
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

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the results
# =============================================================================
# corr_tfmri_fmri = {}
# corr_tfmri_fmri_roi = {}
# ci_corr_tfmri_fmri_roi = {}
# ci_corr_tfmri_fmri_roi_peak_lat = {}

# for rep in args.eeg_reps:

#     results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
#         'stats', 'stats.npy')

#     results = np.load(results_dir, allow_pickle=True).item()

#     metadata = results['metadata']
#     times = results['times']
    # corr_tfmri_fmri[f'eeg_reps-{rep}'] = results['corr_tfmri_fmri']
    # corr_tfmri_fmri_roi[f'eeg_reps-{rep}'] = results['corr_tfmri_fmri_roi']
    # ci_corr_tfmri_fmri_roi[f'eeg_reps-{rep}'] = results['ci_corr_tfmri_fmri_roi']
    # ci_corr_tfmri_fmri_roi_peak_lat[f'eeg_reps-{rep}'] = results['ci_corr_tfmri_fmri_roi_peak_lat']


results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
    'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
metadata = results['metadata']
times = results['times']
corr_invivoeeg = results['corr_invivoeeg']
corr_insilicoeeg_avg_tfmri_avg = results['corr_insilicoeeg_avg_tfmri_avg']
corr_insilicoeeg_sing_tfmri_avg = results['corr_insilicoeeg_sing_tfmri_avg']
corr_insilicoeeg_sing_tfmri_sing = results['corr_insilicoeeg_sing_tfmri_sing']
corr_invivoeeg_roi = results['corr_invivoeeg_roi']
corr_insilicoeeg_avg_tfmri_avg_roi = results['corr_insilicoeeg_avg_tfmri_avg_roi']
corr_insilicoeeg_sing_tfmri_avg_roi = results['corr_insilicoeeg_sing_tfmri_avg_roi']
corr_insilicoeeg_sing_tfmri_sing_roi = results['corr_insilicoeeg_sing_tfmri_sing_roi']


# =============================================================================
# Vertex selection
# =============================================================================
for s, sub in enumerate(args.fmri_subjects):
    for h, hemi in enumerate(args.hemispheres):

        # NCSNR and encoding accuracy vertex selection
        ncsnr = metadata[s]['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = metadata[s]['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

        corr_invivoeeg[s,h,idx_nan] = np.nan
        corr_insilicoeeg_avg_tfmri_avg[s,h,idx_nan] = np.nan
        corr_insilicoeeg_sing_tfmri_avg[s,h,idx_nan] = np.nan
        corr_insilicoeeg_sing_tfmri_sing[s,h,idx_nan] = np.nan


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

# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge', 'plots',
    'encoding_accuracy_surfaceplots')
os.makedirs(save_dir, exist_ok=True)

# Loop over EEG time points
for key in corr_tfmri_fmri.keys():
    for t, time in enumerate(tqdm(times)):

        # Average the results across subjects, and append them across left and
        # right hemishperes
        data = np.append(np.nanmean(corr_tfmri_fmri[key][:,0,:,t], 0),
            np.nanmean(corr_tfmri_fmri[key][:,1,:,t], 0))
        
        # Create the flat brain surface
        vertex_data = cortex.Vertex(
            data,
            subject,
            cmap='afmhot',
            vmin=0,
            vmax=1,
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
            labelsize=15,
            curvature_brightness=0.5,
            with_colorbar=True
            )

        # Add title
        title = f'Time (s): {np.round(time, 3)}'
        plt.title(title, fontsize=fontsize)

        # Save the plot
        plot_file = os.path.join(save_dir, f'correlation_{key}_time-{t:03d}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
        plt.close()


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 20
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
# Plot the vertex-average correlations between t-fMRI and in silico fMRI test
# responses
# =============================================================================
# # Create the plots save directory
# save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge', 'plots')
# os.makedirs(save_dir, exist_ok=True)

# # Create the figure
# fig= plt.figure(figsize=(10, 5))

# # Plot the stimulus onset and chance dashed line
# plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
#     alpha=.5, label='_nolegend_')

# # Loop across modeling types
# for i, (key, val) in enumerate(corr_tfmri_fmri.items()):

#     # Plot the correlation
#     plt.plot(times, np.nanmean(val, (0, 1, 2)), color=colors[i], linewidth=2,
#         label=key)

# # x-axis parameters
# plt.xlabel('Time (ms)', fontsize=fontsize)
# xticks = [0, .1, .2, .3, .4, .5]
# xlabels = [0, 100, 200, 300, 400, 500]
# plt.xticks(ticks=xticks, labels=xlabels)
# plt.xlim(left=min(times), right=max(times))

# # y-axis parameters
# plt.ylabel("Pearson's $r$", fontsize=fontsize)
# yticks = [0, 0.2, 0.4, 0.6, 0.8]
# ylabels = [0, 0.2, 0.4, 0.6, 0.8]
# plt.yticks(ticks=yticks, labels=ylabels)
# plt.ylim(bottom=-.025, top=.6)

# # Legend
# plt.legend(ncol=1, fontsize=15, loc=0, frameon=False)

# # Save the figure
# file_name = os.path.join(save_dir, 'correlation_vertexAvg.svg')
# fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the ROI-wise correlations between t-fMRI and in silico fMRI responses
# =============================================================================
# # Create the figure
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# axs = np.reshape(axs, (-1))

# # Loop across modeling types
# for i, key in enumerate(corr_tfmri_fmri_roi.keys()):

#     # Plot the stimulus onset and chance dashed line
#     axs[i].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
#         alpha=.5, label='_nolegend_')

#     # Loop across ROIs
#     rois = ['early', 'intermediate', 'ventral', 'lateral', 'parietal']
#     rois = ['V1', 'V2', 'V3', 'hV4']
#     for r, roi in enumerate(rois):

#         # Plot the correlation
#         axs[i].plot(times, np.mean(corr_tfmri_fmri_roi[key][roi], 0),
#             color=colors[r], linewidth=2, label=roi)

#         # Plot the CIs
#         axs[i].fill_between(times, ci_corr_tfmri_fmri_roi[key][roi][1],
#             ci_corr_tfmri_fmri_roi[key][roi][0], color=colors[r], alpha=.1)

#         # Plot the peak time point
#         peak = times[np.argmax(np.mean(corr_tfmri_fmri_roi[key][roi], 0))]
#         max_corr = max(np.mean(corr_tfmri_fmri_roi[key][roi], 0))
#         axs[i].scatter(peak, max_corr, color=colors[r], s=200, marker='o',
#             edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
#         ci_low = peak - ci_corr_tfmri_fmri_roi_peak_lat[key][roi][0]
#         ci_up = ci_corr_tfmri_fmri_roi_peak_lat[key][roi][1] - peak
#         conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
#         axs[i].errorbar(peak, max_corr, xerr=conf_int, fmt="none",
#             ecolor='k', elinewidth=1, capsize=3)

#     # x-axis parameters
#     if i in [2, 3]:
#         axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
#         xticks = [0, .1, .2, .3, .4, .5]
#         xlabels = [0, 100, 200, 300, 400, 500]
#         axs[i].set_xticks(ticks=xticks, labels=xlabels)
#         axs[i].set_xlim(left=min(times), right=max(times))

#     # y-axis parameters
#     if i in [0, 2]:
#         axs[i].set_ylabel("Pearson's $r$", fontsize=fontsize)
#         yticks = [0, 0.2, 0.4, 0.6, 0.8]
#         ylabels = [0, 0.2, 0.4, 0.6, 0.8]
#         axs[i].set_yticks(ticks=yticks, labels=ylabels)
#         axs[i].set_ylim(bottom=-.025, top=.6)

#     # Title
#     axs[i].set_title(key, fontsize=10)

#     # Legend
#     if i in [0]:
#         plt.legend(ncol=4, fontsize=10, loc=4, ncols=2, frameon=False)

# # Save the figure
# file_name = os.path.join(save_dir, 'roi_correlation_streams.svg')
# file_name = os.path.join(save_dir, 'roi_correlation_evc.svg')
# fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the vertex-average correlations between t-fMRI and in silico fMRI test
# responses # !!! DELETE # !!!
# =============================================================================
# Create the plots save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new', 'plots')
os.makedirs(save_dir, exist_ok=True)

# Create the figure
fig= plt.figure(figsize=(10, 5))

# Plot the stimulus onset and chance dashed line
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
    alpha=.5, label='_nolegend_')

# Plot the correlation
plt.plot(times, np.nanmean(corr_invivoeeg, (0, 1, 2)), color=colors[0], linewidth=2,
    label='invivoeeg')
plt.plot(times, np.nanmean(corr_insilicoeeg_avg_tfmri_avg, (0, 1, 2)), color=colors[1], linewidth=2,
    label='insilicoeeg_avg_tfmri_avg')
plt.plot(times, np.nanmean(corr_insilicoeeg_sing_tfmri_avg, (0, 1, 2)), color=colors[2], linewidth=2,
    label='insilicoeeg_sing_tfmri_avg')
plt.plot(times, np.nanmean(corr_insilicoeeg_sing_tfmri_sing, (0, 1, 2)), color=colors[3], linewidth=2,
    label='insilicoeeg_sing_tfmri_sing')

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.2, 0.4, 0.6, 0.8]
ylabels = [0, 0.2, 0.4, 0.6, 0.8]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.025, top=.8)

# Legend
plt.legend(ncol=1, fontsize=15, loc=0, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'correlation_vertexAvg.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the ROI-wise correlations between t-fMRI and in silico fMRI responses # !!! DELETE # !!!
# =============================================================================
# Create the figure
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))

results = {}
results['invivoeeg'] = corr_invivoeeg_roi
results['insilicoeeg_avg_tfmri_avg'] = corr_insilicoeeg_avg_tfmri_avg_roi
results['insilicoeeg_sing_tfmri_avg'] = corr_insilicoeeg_sing_tfmri_avg_roi
results['insilicoeeg_sing_tfmri_sing'] = corr_insilicoeeg_sing_tfmri_sing_roi

# Loop across modeling types
for i, (key, val) in enumerate(results.items()):

    # Plot the stimulus onset and chance dashed line
    axs[i].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
        alpha=.5, label='_nolegend_')

    # Loop across ROIs
    rois = ['early', 'intermediate', 'ventral', 'lateral', 'parietal']
    rois = ['V1', 'V2', 'V3', 'hV4']
    for r, roi in enumerate(rois):

        # Plot the correlation
        axs[i].plot(times, np.mean(val[roi], 0),
            color=colors[r], linewidth=2, label=roi)

    # x-axis parameters
    if i in [2, 3]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
        xticks = [0, .1, .2, .3, .4, .5]
        xlabels = [0, 100, 200, 300, 400, 500]
        axs[i].set_xticks(ticks=xticks, labels=xlabels)
        axs[i].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    if i in [0, 2]:
        axs[i].set_ylabel("Pearson's $r$", fontsize=fontsize)
        yticks = [0, 0.2, 0.4, 0.6, 0.8]
        ylabels = [0, 0.2, 0.4, 0.6, 0.8]
        axs[i].set_yticks(ticks=yticks, labels=ylabels)
        axs[i].set_ylim(bottom=-.025, top=.6)

    # Title
    axs[i].set_title(key, fontsize=10)

    # Legend
    if i in [0]:
        plt.legend(ncol=4, fontsize=10, loc=4, ncols=2, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'roi_correlation_streams.svg')
file_name = os.path.join(save_dir, 'roi_correlation_evc.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')