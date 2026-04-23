"""Plot the neural control results, for single ROIs.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subjects : list
    List of subject identifiers for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
roi_1: str
    First ROI used. Valid values are "V1", "V4", and "IT".
roi_2: str
    Second ROI used. Valid values are "V1", "V4", and "IT". If None, then only
    one ROI (roi_1) is used for neural control.
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
parser.add_argument('--roi_1', default='V1', type=str)
parser.add_argument('--roi_2', default='V4', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the neural control results
# =============================================================================
control_resp_roi_1 = {}
base_resp_roi_1 = {}
ci_low_null_distribution_roi_1 = {}
ci_high_null_distribution_roi_1 = {}
ci_low_control_resp_roi_1 = {}
ci_high_control_resp_roi_1 = {}
ci_low_base_resp_roi_1 = {}
ci_high_base_resp_roi_1 = {}
# p_val_bh_roi_1 = {}

control_resp_roi_2 = {}
base_resp_roi_2 = {}
ci_low_null_distribution_roi_2 = {}
ci_high_null_distribution_roi_2 = {}
ci_low_control_resp_roi_2 = {}
ci_high_control_resp_roi_2 = {}
ci_low_base_resp_roi_2 = {}
ci_high_base_resp_roi_2 = {}
# p_val_bh_roi_2 = {}

controls = ['early-drive_late-drive', 'early-suppress_late-suppress',
    'early-drive_late-suppress', 'early-suppress_late-drive']

for sub in args.subjects:
    for c_roi_1 in controls:
        for c_roi_2 in controls:

            data_dir = os.path.join(args.berg_dir, 'neural_control',
                'neural_control', 'stats', args.encoding_model)
            file_name = (f'sub-{sub}_roi_1-{args.roi_1}_{c_roi_1}_'
                f'roi_2-{args.roi_2}_{c_roi_2}.npy')
            data = np.load(os.path.join(data_dir, file_name),
                allow_pickle=True).item()

            control_resp_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['control_resp_roi_1']
            base_resp_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['base_resp_roi_1']
            ci_low_null_distribution_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_low_null_distribution_roi_1']
            ci_high_null_distribution_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_high_null_distribution_roi_1']
            ci_low_control_resp_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_low_control_resp_roi_1']
            ci_high_control_resp_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_high_control_resp_roi_1']
            ci_low_base_resp_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_low_base_resp_roi_1']
            ci_high_base_resp_roi_1[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_high_base_resp_roi_1']

            control_resp_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['control_resp_roi_2']
            base_resp_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['base_resp_roi_2']
            ci_low_null_distribution_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_low_null_distribution_roi_2']
            ci_high_null_distribution_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_high_null_distribution_roi_2']
            ci_low_control_resp_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_low_control_resp_roi_2']
            ci_high_control_resp_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_high_control_resp_roi_2']
            ci_low_base_resp_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_low_base_resp_roi_2']
            ci_high_base_resp_roi_2[f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}'] = data['ci_high_base_resp_roi_2']

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

    fig, axs = plt.subplots(len(controls), len(controls), sharex=True,
        sharey=True, figsize=(40, 30)) # (10, 7.5)

    for c1, c_roi_1 in enumerate(controls):
        for c2, c_roi_2 in enumerate(controls):

            # Plot the stimulus onset dashed line
            axs[c1,c2].plot([0, 0], [100, -100], 'k--', linewidth=2,
                alpha=.25, label='_nolegend_')

            # Plot the neural control responses
            # ROI 1
            c_resp_roi_1 = control_resp_roi_1\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].plot(times, np.mean(c_resp_roi_1, 0),
                color=colors[0], linewidth=2, label=args.roi_1)
            ci_low_resp_roi_1 = ci_low_control_resp_roi_1\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            ci_high_resp_roi_1 = ci_high_control_resp_roi_1\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].fill_between(times, ci_low_resp_roi_1,
                ci_high_resp_roi_1, color=colors[0], alpha=.1)
            # ROI 2
            c_resp_roi_2 = control_resp_roi_2\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].plot(times, np.mean(c_resp_roi_2, 0),
                color=colors[1], linewidth=2, label=args.roi_2)
            ci_low_resp_roi_2 = ci_low_control_resp_roi_2\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            ci_high_resp_roi_2 = ci_high_control_resp_roi_2\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].fill_between(times, ci_low_resp_roi_2,
                ci_high_resp_roi_2, color=colors[1], alpha=.1)

            # Plot the baseline responses
            # ROI 1
            b_resp_roi_1 = base_resp_roi_1\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].plot(times, np.mean(b_resp_roi_1, 0), '--',
                color=colors[0], linewidth=2)
            ci_low_base_roi_1 = ci_low_base_resp_roi_1\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            ci_high_base_roi_1 = ci_high_base_resp_roi_1\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].fill_between(times, ci_low_base_roi_1,
                ci_high_base_roi_1, color=colors[0], alpha=.1)
            # ROI 2
            b_resp_roi_2 = base_resp_roi_2\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].plot(times, np.mean(b_resp_roi_2, 0), '--',
                color=colors[1], linewidth=2)
            ci_low_base_roi_2 = ci_low_base_resp_roi_2\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            ci_high_base_roi_2 = ci_high_base_resp_roi_2\
                [f'{sub}_{args.roi_1}_{c_roi_1}_{args.roi_2}_{c_roi_2}']
            axs[c1,c2].fill_between(times, ci_low_base_roi_2,
                ci_high_base_roi_2, color=colors[1], alpha=.1)

            # Plot the significance markers
            # sig_bool = (p_val_bh[f'{sub}_{roi}_{control}'] < 0.05).astype(np.float32)
            # sig_early = sig_bool[t_min_early:t_max_early+1]
            # sig_late = sig_bool[t_min_late:t_max_late+1]
            # sig_early[sig_early==0] = np.nan
            # sig_late[sig_late==0] = np.nan
            # if control == 'early-drive_late-drive':
            #     sig_early[sig_early==1] = 28
            #     sig_late[sig_late==1] = 28
            # elif control == 'early-suppress_late-suppress':
            #     sig_early[sig_early==1] = 9
            #     sig_late[sig_late==1] = 9
            # elif control == 'early-drive_late-suppress':
            #     sig_early[sig_early==1] = 28
            #     sig_late[sig_late==1] = 9
            # elif control == 'early-suppress_late-drive':
            #     sig_early[sig_early==1] = 9
            #     sig_late[sig_late==1] = 28
            # sig = np.empty(len(times))
            # sig[:] = np.nan
            # sig[t_min_early:t_max_early+1] = sig_early
            # sig[t_min_late:t_max_late+1] = sig_late
            # axs[r,c].scatter(times, sig, s=100, color=colors[0])

            # Title
            title = f'Subject {sub},\n{args.roi_1}: {c_roi_1},\n{args.roi_2}: {c_roi_2}'
            axs[c1,c2].set_title(title, fontsize=fontsize)

            # x-axis parameters
            if c1 == len(controls)-1:
                axs[c1,c2].set_xlabel('Time (ms)', fontsize=fontsize)
                xticks = [-100, -50, 0, 50, 100, 150, 199]
                xlabels = [-100, -50, 0, 50, 100, 150, 200]
                axs[c1,c2].set_xticks(ticks=xticks, labels=xlabels)
                axs[c1,c2].set_xlim(left=min(times), right=max(times))

            # y-axis parameters
            if c2 == 0:
                axs[c1,c2].set_ylabel('MUA', fontsize=fontsize)
                yticks = [10, 15, 20, 25, 30]
                ylabels = [10, 15, 20, 25, 30]
                axs[c1,c2].set_yticks(ticks=yticks, labels=ylabels)
                axs[c1,c2].set_ylim(bottom=8, top=29)

            # Legend
            if c1 == 0 and c2 == 0:
                axs[c1,c2].legend(ncol=1, fontsize=fontsize, loc=0,
                    frameon=False)

    # Save the figure
    save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
        'plots')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'neural_control_sub-{sub}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close(fig)


# =============================================================================
# Scatterplots of in silico responses for the early and late parts of the epoch # !!!
# =============================================================================



# =============================================================================
# Trajectories of responses in V1 and V4 space # !!!
# =============================================================================