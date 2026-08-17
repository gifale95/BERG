"""Plot the univariate RNC cross-subject validated results.

Parameters
----------
roi: str
    Used ROI.
time_window_pairs: list
    List with the time window pairs of interest.
imageset : str
    The image set to use for the analysis. Possible values are: 'imagenet'
    (ILSVRC-2012 validation split) and 'coco' (MS COCO 2017 test split).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--roi', default='FFA', type=str)
parser.add_argument('--time_window_pairs', default=['0.05-0.10__0.10-0.15', '0.05-0.10__0.15-0.20', '0.05-0.10__0.20-0.25', '0.10-0.15__0.15-0.20', '0.10-0.15__0.20-0.25', '0.15-0.20__0.20-0.25'], type=list)
parser.add_argument('--imageset', default='imagenet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the neural control results
# =============================================================================
res = {}

for tw in args.time_window_pairs:

    data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc', 'stats',
        'cv-1', tw, f'stats_roi-{args.roi}.npy')

    res[tw] = np.load(data_dir, allow_pickle=True).item()
    n_sub = len(res[tw]['tfmri_1'])


# =============================================================================
# Create the plot save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc', 'plots',
    'cv-1')
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
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(4/255, 178/255, 153/255), (130/255, 201/255, 240/255),
    (217/255, 214/255, 111/255), (214/255, 83/255, 117/255)]


# =============================================================================
# Plot the univariate responses for the controlling images on scatterplots
# =============================================================================
fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 25))
axs = np.reshape(axs, (-1))

titles = [
    '0.05-0.10 s\nvs.\n0.10-0.15 s',
    '0.05-0.10 s\nvs.\n0.15-0.20 s',
    '0.05-0.10 s\nvs.\n0.20-0.25 s',
    '0.10-0.15 s\nvs.\n0.15-0.20 s',
    '0.10-0.15 s\nvs.\n0.20-0.25 s',
    '0.15-0.20 s\nvs.\n0.20-0.25 s'
]

for t, tw in enumerate(args.time_window_pairs):

    # Break down the time windows
    time_window_1_start, time_window_1_end = map(
        float, tw.split('__')[0].split('-'))
    time_window_2_start, time_window_2_end = map(
        float, tw.split('__')[1].split('-'))

    # Diagonal dashed line
    # axs[t].plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2,
    #     alpha=.4, label='_nolegend_')

    # Baseline images dashed lines
    # base_1 = np.mean(data['base_resp_1'])
    # axs[t].plot([base_1, base_1], [-3, 3], '--w', linewidth=2, alpha=.6,
    #     label='_nolegend_')
    # base_2 = np.mean(data['base_resp_2'])
    # axs[t].plot([-3, 3], [base_2, base_2], '--w', linewidth=2, alpha=.6,
    #     label='_nolegend_')

    # Univariate responses for all images
    for s in range(n_sub):
        axs[t].scatter(res[tw]['tfmri_1'][s,:], res[tw]['tfmri_2'][s,:], c='w',
            alpha=.1, edgecolors='k', label='_nolegend_')

    # Univariate responses for the controlling images
    for s in range(n_sub):
        # 1. high_1_high_2
        axs[t].scatter(np.mean(res[tw]['cv_resp_1']['high_1_high_2'][s]),
            np.mean(res[tw]['cv_resp_2']['high_1_high_2'][s]), c=colors[0],
            s=400, alpha=0.8)
        # 2. low_1_low_2
        axs[t].scatter(np.mean(res[tw]['cv_resp_1']['low_1_low_2'][s]),
            np.mean(res[tw]['cv_resp_2']['low_1_low_2'][s]), c=colors[1],
            s=400, alpha=0.8)
        # 3. high_1_low_2
        axs[t].scatter(np.mean(res[tw]['cv_resp_1']['high_1_low_2'][s]),
            np.mean(res[tw]['cv_resp_2']['high_1_low_2'][s]), c=colors[2],
            s=400, alpha=0.8)
        # 4. low_1_high_2
        axs[t].scatter(np.mean(res[tw]['cv_resp_1']['low_1_high_2'][s]),
            np.mean(res[tw]['cv_resp_2']['low_1_high_2'][s]), c=colors[3],
            s=400, alpha=0.8)

    # Add the correlation scores the two ROI responses for all images
    x = -0.7
    y = 0.6
    s = '$r$=' + str(np.round(np.mean(res[tw]['time_window_pair_corr']), 2))
    # axs[t].text(x, y, s, fontsize=fontsize)

    # x-axis parameters
    xlabel = (f'Univariate response\n{time_window_1_start} s - '
        f'{time_window_1_end} s')
    axs[t].set_xlabel(xlabel, fontsize=fontsize)
    xticks = [1, -0.5, 0, 0.5, 1]
    xlabels = [1, -0.5, 0, 0.5, 1]
    # axs[t].set_xticks(ticks=xticks, labels=xlabels)
    # axs[t].set_xlim(left=-.75, right=.75)

    # y-axis parameters
    ylabel = (f'Univariate response\n{time_window_2_start} s - '
        f'{time_window_2_end} s')
    axs[t].set_ylabel(ylabel, fontsize=fontsize)
    yticks = [1, -0.5, 0, 0.5, 1]
    ylabels = [1, -0.5, 0, 0.5, 1]
    # axs[t].set_yticks(ticks=yticks, labels=ylabels)
    # axs[t].set_ylim(bottom=-1, top=.75)

    # Title
    axs[t].set_title(titles[t], fontsize=fontsize)

    # Aspect
    axs[t].set_aspect('equal')

# Save the figure
file_name = f'univariate_rnc_scatterplots_roi-{args.roi}.png'
fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
    transparent=True, format='png')
plt.close()


# =============================================================================
# Plot the univariate responses significance for the controlling images # !!!
# =============================================================================
'tfmri_1': tfmri_1,
'tfmri_2': tfmri_2,
'control_types': control_types,
'controlling_images': controlling_images,
'baseline_images': baseline_images,
'cv_resp_1': cv_resp_1,
'cv_resp_2': cv_resp_2,
'base_resp_1': base_resp_1,
'base_resp_2': base_resp_2,
'ci_cv_resp_1': ci_cv_resp_1,
'ci_cv_resp_2': ci_cv_resp_2,
'ci_base_resp_1': ci_base_resp_1,
'ci_base_resp_2': ci_base_resp_2,
'pval_1': pval_1,
'pval_2': pval_2,
'pval_corrected_1': pval_corrected_1,
'pval_corrected_2': pval_corrected_2,
'sig_1': sig_1,
'sig_2': sig_2,
'time_window_pair_corr': time_window_pair_corr


# Format the results for plotting
control_responses = {}
ci_control_responses = {}
sig_control_responses = {}
baseline_responses = {}
for roi_pair, stats_roi in stats.items():
    # Control responses
    control_resp = []
    control_resp.append(stats_roi['high_1_high_2_resp'])
    control_resp.append(stats_roi['low_1_low_2_resp'])
    control_resp.append(stats_roi['high_1_low_2_resp'])
    control_resp.append(stats_roi['low_1_high_2_resp'])
    control_responses[roi_pair] = control_resp
    del control_resp
    # Confidence intervals
    ci_control_resp = []
    ci_control_resp.append(stats_roi['ci_high_1_high_2'])
    ci_control_resp.append(stats_roi['ci_low_1_low_2'])
    ci_control_resp.append(stats_roi['ci_high_1_low_2'])
    ci_control_resp.append(stats_roi['ci_low_1_high_2'])
    ci_control_responses[roi_pair] = ci_control_resp
    del ci_control_resp
    # Significance
    sig_control_resp = []
    sig_control_resp.append(stats_roi['h1h2_between_subject_pval'])
    sig_control_resp.append(stats_roi['l1l2_between_subject_pval'])
    sig_control_resp.append(stats_roi['h1l2_between_subject_pval'])
    sig_control_resp.append(stats_roi['l1h2_between_subject_pval'])
    sig_control_responses[roi_pair] = sig_control_resp
    del sig_control_resp
    # Baseline responses
    baseline_responses[roi_pair] = stats_roi['baseline_resp']

# Plot parameters
lim_min = -1.75
lim_max = 1.25
padding = 0.4
x_dist = (abs(lim_min - lim_max) - (padding*2)) / 3
x_dist_within = float(0.25)
x_start = lim_min + padding
xticks = np.asarray((x_start, x_start+x_dist*1, x_start+x_dist*2, x_start+x_dist*3))
x_coord = xticks - (x_dist_within / 2)
alpha = 0.2
sig_bar_length = 0.1
sig_star_offset_top = 0.13
sig_star_offset_bottom = 0.26
fontsize_sig = 20
marker_roi_1 = 'd'
marker_roi_2 = 's'
null_width = 0.1

for roi_pair in args.roi_pairs:

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6,6))
    axs = np.reshape(axs, (-1))

    for c in range(len(control_responses[roi_pair])):

        # ROI 1 (baseline images univariate responses)
        x_null = np.repeat(x_coord[c], len(all_subjects))
        y = baseline_responses[roi_pair][:,0]
        axs[0].plot([x_null[0]-null_width, x_null[0]+null_width],
            [np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
            alpha=.4)

        # ROI 1 (controlling images univariate responses)
        x = np.repeat(x_coord[c], len(all_subjects))
        x_score = x[0]
        y = np.mean(control_responses[roi_pair][c][:,0], 1)
        axs[0].scatter(x, y, marker=marker_roi_1, s=200, color=colors[c],
            alpha=alpha)
        axs[0].scatter(x[0], np.mean(y), marker=marker_roi_1, s=400,
            color=colors[c])
        # ROI 1 (controlling images univariate responses CIs)
        ci_low = np.mean(y) - ci_control_responses[roi_pair][c][0,0]
        ci_up = ci_control_responses[roi_pair][c][1,0] - np.mean(y)
        conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
        axs[0].errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
            ecolor=colors[c], elinewidth=5, capsize=0)

        # ROI 1 (controlling images univariate responses significance)
        idx = roi_pair.find('-')
        roi_1 = roi_pair[:idx]
        roi_2 = roi_pair[idx+1:]
        if sig_control_responses[roi_pair][c][roi_1] < 0.05:
            if c in [0, 2]:
                y = max(np.mean(control_responses[roi_pair][c][:,0], 1)) + \
                    sig_star_offset_top
                axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
                    fontweight='bold', ha='center', va='center')
            elif c in [1, 3]:
                y = min(np.mean(control_responses[roi_pair][c][:,0], 1)) - \
                    sig_star_offset_bottom
                axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
                    fontweight='bold', ha='center', va='center')

        # ROI 2 (baseline images univariate responses)
        x_null = x + x_dist_within
        y = baseline_responses[roi_pair][:,1]
        axs[0].plot([x_null[0]-null_width, x_null[0]+null_width],
            [np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
            alpha=.4)

        # ROI 2 (controlling images univariate responses)
        x += x_dist_within
        x_score = x[0]
        y = np.mean(control_responses[roi_pair][c][:,1], 1)
        axs[0].scatter(x, y, marker=marker_roi_2, s=200, color=colors[c],
            alpha=alpha)
        axs[0].scatter(x[0], np.mean(y), marker=marker_roi_2, s=400,
            color=colors[c])
        # ROI 2 (controlling images univariate responses CIs)
        ci_low = np.mean(y) - ci_control_responses[roi_pair][c][0,1]
        ci_up = ci_control_responses[roi_pair][c][1,1] - np.mean(y)
        conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
        axs[0].errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
            ecolor=colors[c], elinewidth=5, capsize=0)

        # ROI 2 (controlling images univariate responses significance)
        if sig_control_responses[roi_pair][c][roi_2] < 0.05:
            if c in [0, 3]:
                y = max(np.mean(control_responses[roi_pair][c][:,1], 1)) + \
                    sig_star_offset_top
                axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
                    fontweight='bold', ha='center', va='center')
            elif c in [1, 2]:
                y = min(np.mean(control_responses[roi_pair][c][:,1], 1)) - \
                    sig_star_offset_bottom
                axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
                    fontweight='bold', ha='center', va='center')

    # x-axis parameters
    xlabels = ['', '', '', '']
    axs[0].set_xticks(ticks=xticks, labels=xlabels, rotation=45)
    xlabel = 'Neural control\nconditions'
    axs[0].set_xlabel(xlabel, fontsize=fontsize)
    axs[0].set_xlim(left=lim_min, right=lim_max)

    # y-axis parameters
    ylabel = 'Univariate\nresponse'
    axs[0].set_ylabel(ylabel, fontsize=fontsize)
    yticks = [-1, 0, 1, 2]
    ylabels = [-1, 0, 1, 2]
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=lim_min, top=lim_max)

    # Aspect
    axs[0].set_aspect('equal')

    # Save the figure
    file_name = 'univariate_rnc_significance_encoding_models_train_dataset-' + \
        args.encoding_models_train_dataset + '_imageset-' + args.imageset + \
        '_' + roi_pair + '.png'
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='png')
    plt.close()