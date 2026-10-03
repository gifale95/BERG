"""Plot the multivariate RNC cross-subject validated results, and the MDS
results

Parameters
----------
all_subjects : list
    List with the subject identifiers of the 10 THINGS EEG2 subjects.
times : list
    List of used time points.
time_pairs : list
    List of used pairwise time points combinations.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from scipy.stats import zscore
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--times', type=list, default=[0.1, 0.2, 0.3, 0.4])
parser.add_argument('--time_pairs', type=list, default=['0.1-0.2', '0.1-0.3', '0.1-0.4', '0.2-0.3', '0.2-0.4', '0.3-0.4'])
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'rnc_eeg',
    'multivariate_rnc', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the univariate RNC stats
# =============================================================================
stats = {}

for time in args.time_pairs:

    data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
        'stats', 'cv-1', time, 'stats.npy')

    stats[time] = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Set the plot parameters
# =============================================================================
fontsize = 40
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
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
colors = [(108/255, 0/255, 158/255), (153/255, 153/255, 153/255),
    (240/255, 84/255, 0/255)]


# =============================================================================
# Plot the multivariate RNC results from the final genetic optimization
# generation
# =============================================================================
# Plot parameters
x_dist_within = float(0.2)
alpha = 0.2
sig_offset = 0.05
sig_bar_length = 0.03
linewidth_sig_bar = 1
sig_star_offset_top = 0.02
sig_star_offset_bottom = 0.04
s = 600
s_mean = 800

# Plot
fig = plt.figure(figsize=(10,12))

for t, stats_time in enumerate(stats.values()):

    # Aligning images RSA (scores)
    x = np.repeat(t+1-x_dist_within, len(args.all_subjects))
    y = stats_time['best_generation_scores_test']['align'][:,-1]
    plt.scatter(x, y, s=s, color=colors[0], alpha=alpha)
    if t == 0:
        label = '↑ $r$ (Align)'
        plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[0],
            label=label)
    else:
        plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[0])
    # Aligning images RSA (CIs)
    ci_low = np.mean(y) - stats_time['ci_align'][0]
    ci_up = stats_time['ci_align'][1] - np.mean(y)
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
        ecolor=colors[0], elinewidth=5, capsize=0)
    # Aligning images RSA (significance)
    if stats_time['rsa_alignment_between_subject_pval'] < 0.05:
        y_max = max(y) + sig_offset
        plt.plot([x[0], x[0]], [y_max, y_max+sig_bar_length], 'k-',
            [x[0], t+1], [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
            [t+1, t+1], [y_max+sig_bar_length, y_max], 'k-',
            linewidth=linewidth_sig_bar)
        x_mean = np.mean(np.asarray((x[0], t+1)))
        y = y_max + sig_bar_length + sig_star_offset_top
        plt.text(x_mean, y, s='*', fontsize=30, color='k',
            fontweight='bold', ha='center', va='center')

    # Baseline images RSA (scores)
    x = np.repeat(t+1, len(args.all_subjects))
    y = stats_time['baseline_images_score_test']
    plt.scatter(x, y, s=s, color=colors[1], alpha=alpha)
    if t == 0:
        label = 'Baseline'
        plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[1],
            label=label)
    else:
        plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[1])
    # Baseline images RSA (CIs)
    ci_low = np.mean(y) - stats_time['ci_baseline'][0]
    ci_up = stats_time['ci_baseline'][1] - np.mean(y)
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
        ecolor=colors[1], elinewidth=5, capsize=0)

    # Disentangling images RSA (scores)
    x = np.repeat(t+1+x_dist_within, len(args.all_subjects))
    y = stats_time['best_generation_scores_test']['disentangle'][:,-1]
    plt.scatter(x, y, s=s, color=colors[2], alpha=alpha)
    if t == 0:
        label = '↓ $r$ (Disentangle)'
        plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[2],
            label=label)
    else:
        plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[2])
    # Disentangling images RSA (CIs)
    ci_low = np.mean(y) - stats_time['ci_disentangle'][0]
    ci_up = stats_time['ci_disentangle'][1] - np.mean(y)
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
        ecolor=colors[2], elinewidth=5, capsize=0)
    # Disentangling images RSA (significance)
    if stats_time['rsa_disentanglement_between_subject_pval'] < 0.05:
        y_min = min(y) - sig_offset
        plt.plot([x[0], x[0]], [y_min, y_min-sig_bar_length], 'k-',
            [x[0], t+1], [y_min-sig_bar_length, y_min-sig_bar_length], 'k-',
            [t+1, t+1], [y_min-sig_bar_length, y_min], 'k-',
            linewidth=linewidth_sig_bar)
        x_mean = np.mean(np.asarray((x[0], t+1)))
        y = y_min - sig_bar_length - sig_star_offset_bottom
        plt.text(x_mean, y, s='*', fontsize=30, color='k',
            fontweight='bold', ha='center', va='center')

# y-axis parameters
plt.ylabel('Pearson\'s $r$', fontsize=fontsize)
plt.ylim(top=1.15, bottom=-0.19)

# x-axis parameters
xticks = np.arange(1, len(args.time_pairs)+1)
labels = args.time_pairs
plt.xticks(ticks=xticks, labels=labels, rotation=45, fontsize=fontsize)
plt.xlim(left=0.5, right=6.5)

# Legend
plt.legend(loc=3, ncol=3, fontsize=fontsize)

# Save the figure
file_name = os.path.join(save_dir, 'multivariate_rnc_significance.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the multivariate RNC optimization curves
# =============================================================================
# Plot parameters
fontsize = 5
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)

# Make the figure
fig, axs = plt.subplots(len(args.time_pairs), len(args.all_subjects),
    sharex=True, sharey=True)

# Plot
for r, stats_roi in enumerate(stats.values()):

    for s, sub in enumerate(args.all_subjects):

        # Title
        if r == 0:
            title = 'Subject ' + str(sub)
            axs[r,s].set_title(title, fontsize=fontsize)

        # x-axis
        if r == len(args.time_pairs)-1:
            axs[r,s].set_xlabel('Generations', fontsize=fontsize)
            xticks = [1000]
            xlabels = ['1,000']
            axs[r,s].set_xticks(ticks=xticks, labels=xlabels)

        # x-axis
        if s == 0:
            y_label = args.time_pairs[r] + '\nPearson\'s $r$'
            axs[r,s].set_ylabel(y_label, fontsize=fontsize)

        x = np.arange(stats_roi['best_generation_scores_test']['align'].shape[1])

        # Plot the training curves (alignment)
        axs[r,s].plot(x, stats_roi['best_generation_scores_train']['align'][s],
            linewidth=1, color=colors[0])

        # Plot the test curves (alignment)
        axs[r,s].plot(x, stats_roi['best_generation_scores_test']['align'][s],
            '--', linewidth=1, color=colors[0])

        # Plot the baseline images scores
        control_scores = stats_roi['baseline_images_score_test'][s]
        axs[r,s].plot([x[0], x[-1]], [control_scores, control_scores], '--',
            linewidth=1, color=colors[1])

        # Plot the train curves (disentanglement)
        axs[r,s].plot(x,
            stats_roi['best_generation_scores_train']['disentangle'][s],
            linewidth=1, color=colors[2])

        # Plot the test curves (disentanglement)
        axs[r,s].plot(x,
            stats_roi['best_generation_scores_test']['disentangle'][s],
            '--', linewidth=1, color=colors[2])

        # Limits
        axs[r,s].set_xlim(min(x), max(x))
        axs[r,s].set_ylim(bottom=-.05, top=1)

        # Legend
        if r == 0 and s == 0:
            # Create custom lines with increased line width for the legend
            custom_lines = [
                Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], linewidth=4, color=colors[0], linestyle='--'),
                Line2D([0], [0], linewidth=4, color=colors[1], linestyle='--'),
                Line2D([0], [0], linewidth=4, color=colors[2]),
                Line2D([0], [0], linewidth=4, color=colors[2], linestyle='--')
                ]
            legend = [
                '↑ $r$ (train)',
                '↑ $r$ (test)',
                'Baseline',
                '↓ $r$ (train)',
                '↓ $r$ (test)'
                ]
            axs[r,s].legend(custom_lines, legend, loc=2, ncol=5,
                fontsize=fontsize, bbox_to_anchor=(1.525, -6.5), frameon=False,
                markerscale=2)

# Save the figure
file_name = os.path.join(save_dir, 'multivariate_rnc_optimization_curves.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Load the MDS results
# =============================================================================
data_dir = os.path.join(os.path.join(args.berg_dir, 'rnc_eeg',
    'multivariate_rnc', 'multidimensional_scaling',
    'mds_multivariate_responses.npy'))
results = np.load(data_dir, allow_pickle=True).item()

time_point_comb_names = results['time_point_comb_names']
time_point_comb = results['time_point_comb']
mds = {}

for t, time in enumerate(time_point_comb_names):

    mds['align_'+time] = results['mds_align'][time]
    mds['disentangle_'+time] = results['mds_disentangle'][time]

mds_all_images = results['mds_all_images']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
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
colors = [(108/255, 0/255, 158/255), (240/255, 84/255, 0/255)]


# =============================================================================
# Min-max alphas normalization to range [.1, 1], based on the MDS-space
# distance between each time point
# =============================================================================
# Standardize the coordinates within the two MDS dimensions, so that they will
# equally contribute to the distance computation
# Controlling images
for key, val in mds.items():
    for d in range(val.shape[1]):
        mds[key][:,d] = zscore(val[:,d])
# All images
for d in range(mds_all_images.shape[1]):
    mds_all_images[:,d] = zscore(mds_all_images[:,d])

# Compute the alphas for the controlling images
min_alpha = 0.1
max_alpha = 1
alphas = {}
for key, val in mds.items():
    # Compute the distance in MDS-space between each ROI
    dist = []
    for t1, t2 in time_point_comb:
        dist.append(abs(val[t1,0] - val[t2,0]) + abs(val[t1,1] - val[t2,1]))
    dist = np.asarray(dist)
    # Compute the alphas based on the distance
    min_dist, max_dist = dist.min(), dist.max()
    a = min_alpha + (dist - min_dist) * (max_alpha - min_alpha) / \
        (max_dist - min_dist)
    # Flip the scores, so that smaller distances are plotted with higher alphas
    alphas[key] = abs(a - max_alpha - min_alpha)

# Compute the alphas on the MDS results for all images
dist = []
for t1, t2 in time_point_comb:
    dist.append(abs(mds_all_images[t1,0] - mds_all_images[t2,0]) + \
        abs(mds_all_images[t1,1] - mds_all_images[t2,1]))
dist = np.asarray(dist)
# Compute the alphas based on the distance
min_dist, max_dist = dist.min(), dist.max()
a = min_alpha + (dist - min_dist) * (max_alpha - min_alpha) / \
    (max_dist - min_dist)
# Flip the scores, so that smaller distances are plotted with higher alphas
alphas_all_images = abs(a - max_alpha - min_alpha)


# =============================================================================
# Plot the results
# =============================================================================
# Create the figure
for key, val in tqdm(mds.items()):
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7, 7))
    axs = np.reshape(axs, (-1))
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    if key[:1] == 'a':
        color = colors[0]
    elif key[:1] == 'd':
        color = colors[1]

    # Plot the connections between ROIs
    for t, (t1, t2) in enumerate(time_point_comb):
        ax2.plot([val[t1,0], val[t2,0]], [val[t1,1], val[t2,1]],
            color=color, linewidth=5, alpha=alphas[key][t])

    # Plot each ROI in MDS space
    for t, time in enumerate(args.times):
        ax1.scatter(val[t,0], val[t,1], s=4500, c='w', linewidths=0,
            alpha=1)
        ax1.text(val[t,0], val[t,1], time, fontsize=fontsize, fontweight='bold',
            ha='center', va='center_baseline', color='k')

    # x-axis
    ax1.set_xticks([])
    ax2.set_xticks([])

    # y-axis
    ax1.set_yticks([])
    ax2.set_yticks([])

    # Save the figure
    file_name = os.path.join(save_dir, 'mds_multivariate_responses_'+key+'.svg')
    fig.savefig(file_name, bbox_inches='tight', format='svg')
    plt.close()


# =============================================================================
# Plot the results for all images
# =============================================================================
# Create the figure
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7, 7))
axs = np.reshape(axs, (-1))
ax1 = axs[0]
ax2 = ax1.twinx()
ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False)

# Plot the connections between ROIs
for t, (t1, t2) in enumerate(time_point_comb):
    ax2.plot([mds_all_images[t1,0], mds_all_images[t2,0]],
        [mds_all_images[t1,1], mds_all_images[t2,1]], color='k', linewidth=5,
        alpha=alphas_all_images[t])

# Plot each ROI in MDS space
for t, time in enumerate(args.times):
    ax1.scatter(mds_all_images[t,0], mds_all_images[t,1], s=4500, c='w',
        linewidths=0, alpha=1)
    ax1.text(mds_all_images[t,0], mds_all_images[t,1], time, fontsize=fontsize,
        fontweight='bold', ha='center', va='center_baseline', color='k')

# x-axis
ax1.set_xticks([])
ax2.set_xticks([])

# y-axis
ax1.set_yticks([])
ax2.set_yticks([])

# Save the figure
file_name = os.path.join(save_dir, 'mds_multivariate_responses_all_images.svg')
fig.savefig(file_name, bbox_inches='tight', format='svg')
plt.close()