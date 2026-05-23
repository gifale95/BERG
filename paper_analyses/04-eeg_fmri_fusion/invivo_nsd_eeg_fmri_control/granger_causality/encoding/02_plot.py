"""Plot the results of the Granger Causality analysis on t-fMRI ROI responses.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
eeg_train_trials : list
    List indicating which training EEG response trials are used. Possible
    values  are: 'even' (even trials), and 'odd' (odd trials).
rois : list
    List of ROIs used for the Granger Causality analysis.
tot_time_splits : int
    The total number of splits in which the EEG time points are divided.
regression : str
    The type of regression to use for computing Granger Causality. If 'linear',
    use linear regression. If 'ridge', use RidgeCV regression.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--eeg_train_trials', default=['even', 'odd'], type=list)
parser.add_argument('--rois', default=['V1', 'hV4'], type=list)
parser.add_argument('--tot_time_splits', default=10, type=int)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'granger_causality', 'encoding', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the GC results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'granger_causality', 'encoding',
    'gc_scores')

gc = {}

for fs, fsub in enumerate(args.fmri_subjects):

    gc[fsub] = {}

    for roi_1 in args.rois:
        for roi_2 in args.rois:
            if roi_1 != roi_2:

                gc_sub = []

                for et, eeg_train_tr in enumerate(args.eeg_train_trials):
                    for t, ts in enumerate(range(args.tot_time_splits)):

                        file_name = (f'gc_sub-{fsub:02d}_roi_source-{roi_1}_'
                            f'roi_target-{roi_2}_eeg_train_trials-{eeg_train_tr}_'
                            f'regression-{args.regression}_time_split-{ts:02d}.npy')

                        results = np.load(os.path.join(data_dir, file_name),
                            allow_pickle=True).item()

                        if t == 0:
                            gc_t = results['gc']
                        else:
                            gc_t = np.append(gc_t, results['gc'], 1)
                        
                    gc_sub.append(gc_t)
                    del gc_t

                gc[fsub][(roi_1, roi_2)] = np.mean(gc_sub, 0)
                del gc_sub


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
# Plot the results (single subjects)
# =============================================================================
roi_1 = 'V1'
other_rois = ['V2', 'V3', 'hV4', 'ventral', 'FFA', 'EBA', 'PPA']

for s, fsub in enumerate(args.fmri_subjects):

    fig, axs = plt.subplots(len(other_rois), 2, sharex=True,
        sharey=True, figsize=(20, 20)) # (10, 7.5)

    for r, roi_2 in enumerate(other_rois):

        # Plot the feedforward GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2]))
        axs[r,0].imshow(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2], cmap='RdGy_r',
            aspect='equal', vmin=-vlim, vmax=vlim)

        # Plot the feedback GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2]))
        axs[r,1].imshow(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2], cmap='RdGy_r',
            aspect='equal', vmin=-vlim, vmax=vlim)

        # Title
        axs[r,0].set_title(f'Sub {fsub}, {roi_1} to {roi_2}', fontsize=fontsize)
        axs[r,1].set_title(f'Sub {fsub}, {roi_2} to {roi_1}', fontsize=fontsize)

        # X-axis parameters # !!!
        if r == len(other_rois) - 1:
            xticks = [0, 20, 40, 60, 80, 100, 119]
            xlabels = [0, 100, 200, 300, 400, 500, 600]
            axs[r,0].set_xticks(ticks=xticks, labels=xlabels)
            axs[r,0].set_xlabel('Time target (ms)', fontsize=fontsize)
            axs[r,1].set_xticks(ticks=xticks, labels=xlabels)
            axs[r,1].set_xlabel('Time target (ms)', fontsize=fontsize)

        # Y-axis parameters
        yticks = [0, 5, 10, 15]
        ylabels = [-100, -75, -50, -25]
        axs[r,0].set_yticks(ticks=yticks, labels=ylabels)
        axs[r,0].set_ylabel('Time source\n(ms)', fontsize=fontsize)
    
    # Save the figure
    file_name = os.path.join(save_dir, f'gc_sub-{fsub}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the results (single subjects - zoomed in on the early time points)
# =============================================================================
roi_1 = 'V1'
other_rois = ['V2', 'V3', 'hV4', 'ventral', 'FFA', 'EBA', 'PPA']

for s, fsub in enumerate(args.fmri_subjects):

    fig, axs = plt.subplots(len(other_rois), 2, sharex=True,
        sharey=True, figsize=(20, 20)) # (10, 7.5)

    for r, roi_2 in enumerate(other_rois):

        # Plot the feedforward GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2]))
        axs[r,0].imshow(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2], cmap='RdGy_r',
            aspect='equal', vmin=-vlim, vmax=vlim)

        # Plot the feedback GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2]))
        axs[r,1].imshow(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2], cmap='RdGy_r',
            aspect='equal', vmin=-vlim, vmax=vlim)

        # Title
        axs[r,0].set_title(f'Sub {fsub}, {roi_1} to {roi_2}', fontsize=fontsize)
        axs[r,1].set_title(f'Sub {fsub}, {roi_2} to {roi_1}', fontsize=fontsize)

        # X-axis parameters # !!!
        axs[r,0].set_xlim(left=0, right=40)
        axs[r,1].set_xlim(left=0, right=40)
        if r == len(other_rois) - 1:
            xticks = [0, 5, 10, 15, 20, 25, 30, 35, 39]
            xlabels = [0, 25, 50, 75, 100, 125, 150, 175, 200]
            axs[r,0].set_xticks(ticks=xticks, labels=xlabels)
            axs[r,0].set_xlabel('Time target (ms)', fontsize=fontsize)
            axs[r,1].set_xticks(ticks=xticks, labels=xlabels)
            axs[r,1].set_xlabel('Time target (ms)', fontsize=fontsize)

        # Y-axis parameters
        yticks = [0, 5, 10, 15]
        ylabels = [-100, -75, -50, -25]
        axs[r,0].set_yticks(ticks=yticks, labels=ylabels)
        axs[r,0].set_ylabel('Time source\n(ms)', fontsize=fontsize)
    
    # Save the figure
    file_name = os.path.join(save_dir, f'gc_sub-{fsub}_zoom.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the results (subject average)
# =============================================================================
roi_1 = 'V1'
other_rois = ['hV4']

fig, axs = plt.subplots(2, len(other_rois), sharex=True,
    sharey=True, figsize=(20, 12)) # (10, 7.5)

for r, roi_2 in enumerate(other_rois):

    # Plot the feedforward GC results
    gc_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        gc_sub.append(gc[fsub][(roi_1, roi_2)][:-2])
    gc_sub = np.mean(gc_sub, 0)
    vlim = np.max(np.abs(gc_sub))
    # im = axs[0].imshow(gc_sub, cmap='RdGy_r', aspect='auto', vmin=-vlim,
    #     vmax=vlim)
    im = axs[0].imshow(gc_sub, cmap='Reds', aspect='auto', vmin=0,
        vmax=vlim)

    # Plot the feedback GC results
    gc_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        gc_sub.append(gc[fsub][(roi_2, roi_1)][:-2])
    gc_sub = np.mean(gc_sub, 0)
    vlim = np.max(np.abs(gc_sub))
    # im = axs[1].imshow(gc_sub, cmap='RdGy_r', aspect='auto', vmin=-vlim,
    #     vmax=vlim)
    im = axs[1].imshow(gc_sub, cmap='Reds', aspect='auto', vmin=0,
        vmax=vlim)

    # Title
    axs[0].set_title(f'Sub-avg, {roi_1} to {roi_2}', fontsize=fontsize)
    axs[1].set_title(f'Sub-avg, {roi_2} to {roi_1}', fontsize=fontsize)

    # X-axis parameters
    xticks = [0, 51, 102, 153, 205, 256, 307]
    xlabels = [0, 100, 200, 300, 400, 500, 600]
    axs[1].set_xticks(ticks=xticks, labels=xlabels)
    axs[1].set_xlabel('Time target (ms)', fontsize=fontsize)

    # Y-axis parameters
    if r == 0:
        yticks = [0, 12, 24, 36]
        ylabels = [-100, -75, -50, -25]
        axs[0].set_yticks(ticks=yticks, labels=ylabels)
        axs[0].set_ylabel('Time source (ms)\n[relative to target time]', fontsize=fontsize)
        axs[1].set_yticks(ticks=yticks, labels=ylabels)
        axs[1].set_ylabel('Time source (ms)\n[relative to target time]', fontsize=fontsize)
plt.show()

# Save the figure
file_name = os.path.join(save_dir, f'gc_sub-avg_regression-{args.regression}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the GC results (subject average - zoomed in on the early time points)
# =============================================================================
roi_1 = 'V1'
other_rois = ['V2', 'V3', 'hV4', 'ventral', 'FFA', 'EBA', 'PPA']

fig, axs = plt.subplots(len(other_rois), 2, sharex=True,
    sharey=True, figsize=(20, 10)) # (10, 7.5)

for r, roi_2 in enumerate(other_rois):

    # Plot the feedforward GC results
    gc_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        gc_sub.append(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2])
    gc_sub = np.mean(gc_sub, 0)
    vlim = np.max(np.abs(gc_sub))
    im = axs[r,0].imshow(gc_sub, cmap='RdGy_r', aspect='equal', vmin=-vlim,
        vmax=vlim)

    # Plot the feedback GC results
    gc_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        gc_sub.append(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2])
    gc_sub = np.mean(gc_sub, 0)
    vlim = np.max(np.abs(gc_sub))
    im = axs[r,1].imshow(gc_sub, cmap='RdGy_r', aspect='equal', vmin=-vlim,
        vmax=vlim)

    # Title
    axs[r,0].set_title(f'Sub-avg, {roi_1} to {roi_2}', fontsize=fontsize)
    axs[r,1].set_title(f'Sub-avg, {roi_2} to {roi_1}', fontsize=fontsize)

    # X-axis parameters # !!!
    axs[r,0].set_xlim(left=0, right=60)
    axs[r,1].set_xlim(left=0, right=60)
    if r == len(other_rois) - 1:
        xticks = [0, 10, 20, 30, 40, 50, 59]
        xlabels = [0, 50, 100, 150, 200, 250, 300]
        axs[r,0].set_xticks(ticks=xticks, labels=xlabels)
        axs[r,0].set_xlabel('Time target (ms)', fontsize=fontsize)
        axs[r,1].set_xticks(ticks=xticks, labels=xlabels)
        axs[r,1].set_xlabel('Time target (ms)', fontsize=fontsize)

    # Y-axis parameters
    yticks = [0, 5, 10, 15]
    ylabels = [-100, -75, -50, -25]
    axs[r,0].set_yticks(ticks=yticks, labels=ylabels)
    axs[r,0].set_ylabel('Time source\n(ms)', fontsize=fontsize)

# Save the figure
file_name = os.path.join(save_dir, f'gc_sub-avg_zoom.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()