"""Plot t-fMRI time-time RSA results.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
roi : str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest used to find the
    baseline and controlling images.
use_time_bins: int
    If '1', average the t-fMRI responses into four time bins (50-100ms,
    100-150ms, 150-200ms, 200-250ms). If '0', do not average the t-fMRI
    responses into time bins.
correlation_measure: str
    Whether to use 'pearson' or 'spearman' correlation.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.05-0.10__0.20-0.25', type=str)
parser.add_argument('--use_time_bins', default=1, type=int)
parser.add_argument('--correlation_measure', default='pearson', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Plot time-time RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the t-fMRI time-time RSA results, and average them across fMRI subjects
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'tfmri_time_time_rsa')

# Loop across fMRI subjects
for s, sub in enumerate(args.fmri_subjects):

    # Load the results
    file_name = (f'tfmri_time_time_rsa_sub-{sub:02d}_roi-{args.roi}_'
        f'image_window_pair-{args.time_window_pair}_'
        f'use_time_bins-{args.use_time_bins}_'
        f'corr-{args.correlation_measure}.npy')
    time_time_rsa_sub = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()

    # Sum the results across fMRI subjects
    if s == 0:
        time_time_rsa = time_time_rsa_sub
    else:
        for key, val in time_time_rsa.items():
            time_time_rsa[key] += time_time_rsa_sub[key]
    del time_time_rsa_sub

# Average the results across fMRI subjects
for key, val in time_time_rsa.items():
    time_time_rsa[key] /= len(args.fmri_subjects)


# =============================================================================
# Create the plot save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'plots', 'tfmri_time_time_rsa')
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
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot the RSA results
# =============================================================================
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(30, 15))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(time_time_rsa.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the time-time RSA results
    im = axs[i].imshow(val, aspect='auto', cmap='magma', origin='lower',
        vmin=0, vmax=1)

    # Plot title
    axs[i].set_title(key, fontsize=fontsize)

    # x-axis parameters
    if i in [3, 4, 5]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
        if args.use_time_bins == 0:
            xticks = [20, 60, 100, 139]
            xlabels = [0, 200, 400, 600]
        elif args.use_time_bins == 1:
            xticks = [0, 1, 2, 3]
            xlabels = ['50-\n100', '100-\n150', '150-\n200', '200-\n250']
        axs[i].set_xticks(ticks=xticks, labels=xlabels)

    # y-axis parameters
    if i in [0, 3]:
        axs[i].set_ylabel('Time (ms)', fontsize=fontsize)
        if args.use_time_bins == 0:
            yticks = [20, 60, 100, 139]
            ylabels = [0, 200, 400, 600]
        elif args.use_time_bins == 1:
            yticks = [0, 1, 2, 3]
            ylabels = ['50-100', '100-150', '150-200', '200-250']
        axs[i].set_yticks(ticks=yticks, labels=ylabels)

    # Colorbar
    if args.correlation_measure == 'pearson':
        label = "Pearson's $r$"
    if args.correlation_measure == 'spearman':
        label = "Spearman's $\\rho$"
    fig.colorbar(im, label=label, fraction=0.046, pad=0.04)

# Save the figure
file_name = os.path.join(save_dir, f'tfmri_time_time_rsa_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}_'
    f'use_time_bins-{args.use_time_bins}_'
    f'corr-{args.correlation_measure}.npy')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)