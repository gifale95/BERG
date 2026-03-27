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

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--rois', default=['V1', 'V4', 'IT'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the neural control results
# =============================================================================
control_resp = {}
baseline_resp = {}
ci_low_null_distribution = {}
ci_high_null_distribution = {}
ci_low_control_resp = {}
ci_high_control_resp = {}
ci_low_baseline_resp = {}
ci_high_baseline_resp = {}
p_val_bh = {}

controls = ['early-drive_late-drive', 'early-suppress_late-suppress',
    'early-drive_late-suppress', 'early-suppress_late-drive']

for sub in args.subjects:
    for roi in args.rois:
        for control in controls:

            data_dir = os.path.join(args.berg_dir, 'neural_control',
                'single_rois', 'stats', args.encoding_model,
                f'sub-{sub}_roi-{roi}_{control}.npy')
            data = np.load(data_dir, allow_pickle=True).item()

            control_resp[f'{sub}_{roi}_{control}'] = data['control_resp']
            baseline_resp[f'{sub}_{roi}_{control}'] = data['baseline_resp']
            ci_low_null_distribution[f'{sub}_{roi}_{control}'] = data['ci_low_null_distribution']
            ci_high_null_distribution[f'{sub}_{roi}_{control}'] = data['ci_high_null_distribution']
            ci_low_control_resp[f'{sub}_{roi}_{control}'] = data['ci_low_control_resp']
            ci_high_control_resp[f'{sub}_{roi}_{control}'] = data['ci_high_control_resp']
            ci_low_baseline_resp[f'{sub}_{roi}_{control}'] = data['ci_low_baseline_resp']
            ci_high_baseline_resp[f'{sub}_{roi}_{control}'] = data['ci_high_baseline_resp']
            p_val_bh[f'{sub}_{roi}_{control}'] = data['p_val_bh']

            times = data['times']

            t_min_early = data['t_min_early']
            t_max_early = data['t_max_early']
            t_min_late = data['t_min_late']
            t_max_late = data['t_max_late']


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
# Plot the neural control results
# =============================================================================
for s, sub in enumerate(args.subjects):

    fig, axs = plt.subplots(len(args.rois), len(controls), sharex=True,
        sharey=True, figsize=(30, 30)) # (10, 7.5)

    for r, roi in enumerate(args.rois):
        for c, control in enumerate(controls):

            # Plot the stimulus onset dashed line
            axs[r,c].plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.25,
                label='_nolegend_')

            # Plot the neural control responses
            axs[r,c].plot(times, np.mean(
                control_resp[f'{sub}_{roi}_{control}'], 0),
                color=colors[0], linewidth=2, label='Control')
            axs[r,c].fill_between(times,
                ci_low_control_resp[f'{sub}_{roi}_{control}'],
                ci_high_control_resp[f'{sub}_{roi}_{control}'],
                color=colors[0], alpha=.1)

            # Plot the baseline responses
            axs[r,c].plot(times, np.mean(
                baseline_resp[f'{sub}_{roi}_{control}'], 0),
                color='k', linewidth=2, label='Baseline')
            axs[r,c].fill_between(times,
                ci_low_baseline_resp[f'{sub}_{roi}_{control}'],
                ci_high_baseline_resp[f'{sub}_{roi}_{control}'],
                color='k', alpha=.1)

            # Plot the significance markers
            sig_bool = (p_val_bh[f'{sub}_{roi}_{control}'] < 0.05).astype(np.float32)
            sig_early = sig_bool[t_min_early:t_max_early+1]
            sig_late = sig_bool[t_min_late:t_max_late+1]
            sig_early[sig_early==0] = np.nan
            sig_late[sig_late==0] = np.nan
            if control == 'early-drive_late-drive':
                sig_early[sig_early==1] = 28
                sig_late[sig_late==1] = 28
            elif control == 'early-suppress_late-suppress':
                sig_early[sig_early==1] = 9
                sig_late[sig_late==1] = 9
            elif control == 'early-drive_late-suppress':
                sig_early[sig_early==1] = 28
                sig_late[sig_late==1] = 9
            elif control == 'early-suppress_late-drive':
                sig_early[sig_early==1] = 9
                sig_late[sig_late==1] = 28
            sig = np.empty(len(times))
            sig[:] = np.nan
            sig[t_min_early:t_max_early+1] = sig_early
            sig[t_min_late:t_max_late+1] = sig_late
            axs[r,c].scatter(times, sig_early, s=100, color=colors[0])
            axs[r,c].scatter(times, sig_late, s=100, color=colors[0])

            # Title
            title = f'Subject {sub}, {roi}, {control}'
            axs[r,c].set_title(title, fontsize=fontsize)

            # x-axis parameters
            if r == len(args.rois)-1:
                axs[r,c].set_xlabel('Time (ms)', fontsize=fontsize)
                xticks = [-100, -50, 0, 50, 100, 150, 199]
                xlabels = [-100, -50, 0, 50, 100, 150, 200]
                axs[r,c].set_xticks(ticks=xticks, labels=xlabels)
                axs[r,c].set_xlim(left=min(times), right=max(times))

            # y-axis parameters
            if c == 0:
                axs[r,c].set_ylabel('MUA', fontsize=fontsize)
                yticks = [10, 15, 20, 25, 30]
                ylabels = [10, 15, 20, 25, 30]
                axs[r,c].set_yticks(ticks=yticks, labels=ylabels)
                axs[r,c].set_ylim(bottom=8, top=29)

            # Legend
            if r == 0 and c == 0:
                axs[r,c].legend(ncol=1, fontsize=fontsize, loc=1,
                    frameon=False)

    # Save the figure
    save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
        'plots')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'neural_control_sub-{sub}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close(fig)


# =============================================================================
# Scatterplots of in silico responses for the early and late parts of the epoch # !!!
# =============================================================================