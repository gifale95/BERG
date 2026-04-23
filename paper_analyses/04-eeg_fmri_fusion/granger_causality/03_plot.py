"""Plot the results of the Granger Causality analysis on t-fMRI ROI responses.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
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
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'granger_causality'
    'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the GC results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'granger_causality',
    'gc_scores', 'source_dataset-things_eeg_2')

gc = {}
rsa_times = {}
rsa_alexnet_pearson = {}
rsa_alexnet_cosyne = {}

for fs, fsub in enumerate(args.fmri_subjects):

    file_name = (f'gc_fmri_sub-{fsub:02d}_'
                f'regression-{args.regression}.npy')

    results = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()

    gc[fsub] = results['gc']
    rsa_times[fsub] = results['rsa_times']
    rsa_alexnet_pearson[fsub] = results['rsa_alexnet_pearson']
    rsa_alexnet_cosyne[fsub] = results['rsa_alexnet_cosyne']
    times_target = results['times_target']
    times = results['times']
    del results


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
other_rois = ['V2', 'hV4', 'ventral']

for s, fsub in enumerate(args.fmri_subjects):

    fig, axs = plt.subplots(len(other_rois), 2, sharex=True,
        sharey=True, figsize=(20, 20)) # (10, 7.5)

    for r, roi_2 in enumerate(other_rois):

        # Plot the feedforward GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2]))
        axs[r,0].imshow(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2], cmap='RdGy_r',
            aspect='auto', vmin=-vlim, vmax=vlim)

        # Plot the feedback GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2]))
        axs[r,1].imshow(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2], cmap='RdGy_r',
            aspect='auto', vmin=-vlim, vmax=vlim)

        # Title
        axs[r,0].set_title(f'Sub {fsub}, {roi_1} to {roi_2}', fontsize=fontsize)
        axs[r,1].set_title(f'Sub {fsub}, {roi_2} to {roi_1}', fontsize=fontsize)

        # X-axis parameters
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
        axs[r,0].set_ylabel('Time source (ms)', fontsize=fontsize)
    
    # Save the figure
    file_name = os.path.join(save_dir, f'gc_sub-{fsub}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the results (single subjects - zoomed in on the early time points)
# =============================================================================
roi_1 = 'V1'
other_rois = ['V2', 'hV4', 'ventral']

for s, fsub in enumerate(args.fmri_subjects):

    fig, axs = plt.subplots(len(other_rois), 2, sharex=True,
        sharey=True, figsize=(20, 20)) # (10, 7.5)

    for r, roi_2 in enumerate(other_rois):

        # Plot the feedforward GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2]))
        axs[r,0].imshow(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2], cmap='RdGy_r',
            aspect='auto', vmin=-vlim, vmax=vlim)

        # Plot the feedback GC results
        vlim = np.max(np.abs(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2]))
        axs[r,1].imshow(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2], cmap='RdGy_r',
            aspect='auto', vmin=-vlim, vmax=vlim)

        # Title
        axs[r,0].set_title(f'Sub {fsub}, {roi_1} to {roi_2}', fontsize=fontsize)
        axs[r,1].set_title(f'Sub {fsub}, {roi_2} to {roi_1}', fontsize=fontsize)

        # X-axis parameters
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
        axs[r,0].set_ylabel('Time source (ms)', fontsize=fontsize)
    
    # Save the figure
    file_name = os.path.join(save_dir, f'gc_sub-{fsub}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the results (subject average)
# =============================================================================
roi_1 = 'V1'
other_rois = ['V2', 'hV4', 'ventral']

fig, axs = plt.subplots(len(other_rois), 2, sharex=True,
    sharey=True, figsize=(20, 20)) # (10, 7.5)

for r, roi_2 in enumerate(other_rois):

    # Plot the feedforward GC results
    gc_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        gc_sub.append(gc[fsub][f'{roi_1}_to_{roi_2}'][:-2])
    gc_sub = np.mean(gc_sub, 0)
    vlim = np.max(np.abs(gc_sub))
    im = axs[r,0].imshow(gc_sub, cmap='RdGy_r', aspect='auto', vmin=-vlim,
        vmax=vlim)

    # Plot the feedback GC results
    gc_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        gc_sub.append(gc[fsub][f'{roi_2}_to_{roi_1}'][:-2])
    gc_sub = np.mean(gc_sub, 0)
    vlim = np.max(np.abs(gc_sub))
    im = axs[r,1].imshow(gc_sub, cmap='RdGy_r', aspect='auto', vmin=-vlim,
        vmax=vlim)

    # Title
    axs[r,0].set_title(f'Sub {fsub}, {roi_1} to {roi_2}', fontsize=fontsize)
    axs[r,1].set_title(f'Sub {fsub}, {roi_2} to {roi_1}', fontsize=fontsize)

    # X-axis parameters
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
    axs[r,0].set_ylabel('Time source (ms)', fontsize=fontsize)

# Save the figure
file_name = os.path.join(save_dir, f'gc_sub-avg.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the between-time-point RSA results (subject-average)
# =============================================================================
# Plot parameters
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False

rois = ['V1']

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7.5, 7.5))

for r, roi in enumerate(rois):

    # Plot RSA results
    rsa_times_sub = []
    for s, fsub in enumerate(args.fmri_subjects):
        rsa_times_sub.append(rsa_times[fsub][roi])
    rsa_times_sub = np.mean(np.flip(rsa_times_sub, 1), 0)
    vlim = np.max(np.abs(rsa_times_sub))
    cax = axs[r,0].imshow(rsa_times_sub, cmap='RdGy_r', aspect='equal',
        vmin=-vlim, vmax=vlim)
    cbar = plt.colorbar(cax, shrink=0.75, ticks=[-vlim, 0, vlim],
        label="Pearson\'s $r$", location='left')
    
    # Title
    axs[r,0].set_title(f'{roi}', fontsize=fontsize)

    # X-axis parameters
    xticks = [0, 20, 40, 60, 80, 100, 119]
    xlabels = [0, 100, 200, 300, 400, 500, 600]
    axs[r,0].set_xticks(ticks=xticks, labels=xlabels)
    axs[r,0].set_xlabel('Time (ms)', fontsize=fontsize)

    # Y-axis parameters
    yticks = abs(xticks - len(times))
    ylabels = [0, 100, 200, 300, 400, 500, 600]
    axs[r,0].set_yticks(ticks=yticks, labels=ylabels)
    axs[r,0].set_ylabel('Time (ms)', fontsize=fontsize)

# Save the figure
file_name = os.path.join(save_dir, f'rsa_time_sub-avg.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the AlexNet RSA results (subject-average)
# =============================================================================
# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
model_layers = [
    'features.2',
    'features.5',
    'features.7',
    'features.9',
    'features.12',
    'classifier.2',
    'classifier.5',
    'classifier.6'
    ]
colors = sample_cmap(len(model_layers))

# Plot parameters
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True

rois = ['V1']

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 7.5))

for r, roi in enumerate(rois):

    # Plot the chance and stimulus onset dashed lines
    axs[r,0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')

    # Loop across model layers
    for l, layer in enumerate(model_layers):

        # Average the RSA results across subjects
        rsa_alexnet_sub = []
        for s, fsub in enumerate(args.fmri_subjects):
            rsa_alexnet_sub.append(rsa_alexnet_pearson[fsub][(roi, layer)]) # !!! rsa_alexnet_pearson, rsa_alexnet_cosyne
        rsa_alexnet_sub = np.mean(rsa_alexnet_sub, 0)

        # Plot the RSA subject-average results
        axs[r,0].plot(times, rsa_alexnet_sub, color=colors[l], linewidth=2,
            label=layer)

    # x-axis parameters
    axs[r,0].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
    xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[r,0].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[r,0].set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.05, 0.1, 0.15, 0.2]
    ylabels = [0, 0.05, 0.1, 0.15, 0.2]
    # plt.yticks(ticks=yticks, labels=ylabels)
    # axs[r,0].set_ylim(bottom=-.02, top=.2)

    # Legend
    axs[r,0].legend(fontsize=15, ncol=1, loc=0, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, f'rsa_time_sub-avg.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()