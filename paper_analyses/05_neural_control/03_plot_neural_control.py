"""Plot the neural control results.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subjects : list
    List of subject identifiers for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
rois: list
    List of ROIs used. Valid values are "V1", "V4", and "IT".
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
from berg import BERG
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--rois', default=['V1', 'V4', 'IT'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the neural control results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control',
    'quantitative_results', args.encoding_model)

cv_control_data = {}
p_val = {}
cv_control_data_avg = {}
p_val_avg = {}
ci_low_null_distribution = {}
ci_high_null_distribution = {}

for roi in args.rois:
    for control in ['drive', 'suppress']:
        file_name = f'roi-{roi}_control-{control}.npy'
        data = np.load(os.path.join(data_dir, file_name),
            allow_pickle=True).item()
        cv_control_data[f'{roi}_{control}'] = data['cv_control_data']
        p_val[f'{roi}_{control}'] = data['p_val_bh']
        cv_control_data_avg[f'{roi}_{control}'] = data['cv_control_data_avg']
        p_val_avg[f'{roi}_{control}'] = data['p_val_avg_bh']
        ci_low_null_distribution[f'{roi}_{control}'] = \
            data['ci_low_null_distribution']
        ci_high_null_distribution[f'{roi}_{control}'] = \
            data['ci_high_null_distribution']


# =============================================================================
# Get the times
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subjects[0]
)

times = metadata['utah_array']['times']


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
colors = [(139/255, 0/255, 0/255), (0/255, 115/255, 155/255),
    (0/255, 0/255, 0/255)]


# =============================================================================
# Plot the neural control results (time-resolved)
# =============================================================================
fig, axs = plt.subplots(len(args.subjects), len(args.rois), sharex=True,
    sharey=True, figsize=(30, 15))

for r, roi in enumerate(args.rois):
    for s, sub in enumerate(args.subjects):

        # Plot the stimulus onset dashed line
        axs[s,r].plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.25,
            label='_nolegend_')

        # Plot the baseline null distribution confidence intervals
        ci_low = (ci_low_null_distribution[f'{roi}_drive'][s] + \
            ci_low_null_distribution[f'{roi}_suppress'][s]) / 2
        ci_high = (ci_high_null_distribution[f'{roi}_drive'][s] + \
            ci_high_null_distribution[f'{roi}_suppress'][s]) / 2
        axs[s,r].fill_between(times, ci_low, ci_high, color='k', alpha=.25)

        # Plot the neural control results
        s_cv = np.delete((0, 1), s)[0]
        n_images = cv_control_data[f'{roi}_drive'][s_cv].shape[0]
        for i in range(n_images):
            if r == 0 and s == 0 and i == 0:
                axs[s,r].plot(times, np.transpose(
                    cv_control_data[f'{roi}_drive'][s_cv][i]),
                    color=colors[0], linewidth=1, label='Drive')
                axs[s,r].plot(times, np.transpose(
                    cv_control_data[f'{roi}_suppress'][s_cv][i]),
                    color=colors[2], linewidth=1, label='Suppress')
            else:
                axs[s,r].plot(times, np.transpose(
                    cv_control_data[f'{roi}_drive'][s_cv][i]),
                    color=colors[0], linewidth=1)
                axs[s,r].plot(times, np.transpose(
                    cv_control_data[f'{roi}_suppress'][s_cv][i]),
                    color=colors[2], linewidth=1)

        # Plot the significance markers
        sig_drive = np.empty(len(times))
        sig_suppress = np.empty(len(times))
        sig_drive[:] = np.nan
        sig_suppress[:] = np.nan
        sig_drive[p_val[f'{roi}_drive'][s]<0.05] = 28
        sig_suppress[p_val[f'{roi}_suppress'][s]<0.05] = 7
        plt.scatter(times, sig_drive, s=100, color=colors[0])
        plt.scatter(times, sig_suppress, s=100, color=colors[2])

        # Title
        title = f'{roi} - Subject {sub}'
        axs[s,r].set_title(title, fontsize=fontsize)

        # x-axis parameters
        if s == 1:
            axs[s,r].set_xlabel('Time (ms)', fontsize=fontsize)
            xticks = [-100, -50, 0, 50, 100, 150, 199]
            xlabels = [-100, -50, 0, 50, 100, 150, 200]
            axs[s,r].set_xticks(ticks=xticks, labels=xlabels)
            axs[s,r].set_xlim(left=min(times), right=max(times))

        # y-axis parameters
        if r == 0:
            axs[s,r].set_ylabel('MUA', fontsize=fontsize)
            yticks = [10, 15, 20, 25, 30]
            ylabels = [10, 15, 20, 25, 30]
            axs[s,r].set_yticks(ticks=yticks, labels=ylabels)
            axs[s,r].set_ylim(bottom=8, top=29)

        # Legend
        if s == 0 and r == 0:
            axs[s,r].legend(ncol=1, fontsize=fontsize, loc=1, frameon=False)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'neural_control', 'plots')
os.makedirs(save_dir, exist_ok=True)
file_name = os.path.join(save_dir, 'neural_control_time-resolved.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the neural control results (time-averaged)
# =============================================================================
fig, axs = plt.subplots(len(args.subjects), len(args.rois), sharex=True,
    sharey=True, figsize=(30, 15))

for r, roi in enumerate(args.rois):
    for s, sub in enumerate(args.subjects):

        # Plot the stimulus onset dashed line
        axs[s,r].plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.25,
            label='_nolegend_')

        # Plot the baseline null distribution confidence intervals
        ci_low = (ci_low_null_distribution[f'{roi}_drive'][s] + \
            ci_low_null_distribution[f'{roi}_suppress'][s]) / 2
        ci_high = (ci_high_null_distribution[f'{roi}_drive'][s] + \
            ci_high_null_distribution[f'{roi}_suppress'][s]) / 2
        axs[s,r].fill_between(times, ci_low, ci_high, color='k', alpha=.25)

        # Plot the neural control results
        s_cv = np.delete((0, 1), s)[0]
        n_images = cv_control_data_avg[f'{roi}_drive'][s_cv].shape[0]
        for i in range(n_images):
            if r == 0 and s == 0 and i == 0:
                axs[s,r].plot(times, np.transpose(
                    cv_control_data_avg[f'{roi}_drive'][s_cv][i]),
                    color=colors[0], linewidth=1, label='Drive')
                axs[s,r].plot(times, np.transpose(
                    cv_control_data_avg[f'{roi}_suppress'][s_cv][i]),
                    color=colors[2], linewidth=1, label='Suppress')
            else:
                axs[s,r].plot(times, np.transpose(
                    cv_control_data_avg[f'{roi}_drive'][s_cv][i]),
                    color=colors[0], linewidth=1)
                axs[s,r].plot(times, np.transpose(
                    cv_control_data_avg[f'{roi}_suppress'][s_cv][i]),
                    color=colors[2], linewidth=1)

        # Plot the significance markers
        sig_drive = np.empty(len(times))
        sig_suppress = np.empty(len(times))
        sig_drive[:] = np.nan
        sig_suppress[:] = np.nan
        sig_drive[p_val_avg[f'{roi}_drive'][s]<0.05] = 28
        sig_suppress[p_val_avg[f'{roi}_suppress'][s]<0.05] = 7
        plt.scatter(times, sig_drive, s=100, color=colors[0])
        plt.scatter(times, sig_suppress, s=100, color=colors[2])

        # Title
        title = f'{roi} - Subject {sub}'
        axs[s,r].set_title(title, fontsize=fontsize)

        # x-axis parameters
        if s == 1:
            axs[s,r].set_xlabel('Time (ms)', fontsize=fontsize)
            xticks = [-100, -50, 0, 50, 100, 150, 199]
            xlabels = [-100, -50, 0, 50, 100, 150, 200]
            axs[s,r].set_xticks(ticks=xticks, labels=xlabels)
            axs[s,r].set_xlim(left=min(times), right=max(times))

        # y-axis parameters
        if r == 0:
            axs[s,r].set_ylabel('MUA', fontsize=fontsize)
            yticks = [10, 15, 20, 25, 30]
            ylabels = [10, 15, 20, 25, 30]
            axs[s,r].set_yticks(ticks=yticks, labels=ylabels)
            axs[s,r].set_ylim(bottom=8, top=29)

        # Legend
        if s == 0 and r == 0:
            axs[s,r].legend(ncol=1, fontsize=fontsize, loc=1, frameon=False)

# Save the figure
save_dir = os.path.join(args.berg_dir, 'neural_control', 'plots')
os.makedirs(save_dir, exist_ok=True)
file_name = os.path.join(save_dir, 'neural_control_time-averaged.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)