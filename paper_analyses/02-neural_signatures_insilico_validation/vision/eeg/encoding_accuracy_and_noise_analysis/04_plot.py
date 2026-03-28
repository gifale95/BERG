"""Plot the encoding accuracy and noise analysis results for BERG's EEG
encoding models trained on THINGS EEG2.

Parameters
----------
subjects : list of int
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the results, and create the plot saving directory
# =============================================================================
# Load the results
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation',
'vision', 'eeg', 'encoding_accuracy', 'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
correlation = results['correlation']
ci_correlation = results['ci_correlation']
sig_correlation = results['sig_correlation']
diff_correlation = results['diff_correlation']
ci_diff_correlation = results['ci_diff_correlation']
sig_diff_correlation = results['sig_diff_correlation']
corr_iv_iv = results['corr_iv_iv']
corr_iv_is = results['corr_iv_is']
ci_corr_iv_is = results['ci_corr_iv_is']
ci_corr_iv_iv = results['ci_corr_iv_iv']
sig_less = results['sig_less']
sig_greater = results['sig_greater']
times = results['metadata'][0]['eeg']['times']
ch_names = results['metadata'][0]['eeg']['ch_names']

# Plot save directory
save_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'encoding_accuracy', 'plots')
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

# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
# Channel groups
channel_groups = ['Occipital', 'Parietal', 'Temporal', 'Central', 'Frontal']
colors = sample_cmap(len(channel_groups))

# Loop across models
for model in correlation.keys():

    # Plot the encoding accuracy
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(10, 7.5))
    axs = np.reshape(axs, (-1))

    # Plot the chance and stimulus onset dashed lines
    axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')

    # Loop across channel groups
    for c, key in enumerate(channel_groups):
        # Plot the encoding accuracy
        axs[0].plot(times, np.mean(correlation[model][:,c], 0),
            color=colors[c], linewidth=2, label=key)
        # Plot the confidence intervals
        axs[0].fill_between(times, ci_correlation[model][0,c],
            ci_correlation[model][1,c], color=colors[c], alpha=.1)
        # Plot the significance markers
        sig = np.empty(len(times))
        sig[:] = np.nan
        sig[sig_correlation[model][c]] = -0.015 * (c + 1) - 0.005
        plt.scatter(times, sig, s=30, color=colors[c])

    # x-axis parameters
    axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
    xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[0].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=-.1, top=0.85)

    # Legend
    axs[0].legend(fontsize=fontsize, ncol=3, loc=0, frameon=False)

    # Save the figure
    file_name = os.path.join(save_dir, f'encoding_accuracy_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the encoding accuracy difference results
# =============================================================================
# Channel groups
channel_groups = ['Occipital', 'Parietal', 'Temporal', 'Central', 'Frontal']
colors = sample_cmap(len(channel_groups))

# Loop across models
for model in diff_correlation.keys():

    # Plot the encoding accuracy
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(10, 7.5))
    axs = np.reshape(axs, (-1))

    # Plot the chance and stimulus onset dashed lines
    axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')

    # Loop across channel groups
    for c, key in enumerate(channel_groups):
        # Plot the encoding accuracy
        axs[0].plot(times, np.mean(diff_correlation[model][:,c], 0),
            color=colors[c], linewidth=2, label=key)
        # Plot the confidence intervals
        axs[0].fill_between(times, ci_diff_correlation[model][0,c],
            ci_diff_correlation[model][1,c], color=colors[c], alpha=.1)
        # Plot the significance markers
        sig = np.empty(len(times))
        sig[:] = np.nan
        sig[sig_diff_correlation[model][c]] = -0.005 * (c + 1) - 0.007
        plt.scatter(times, sig, s=30, color=colors[c])

    # x-axis parameters
    axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
    xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[0].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[0].set_ylabel("Δ Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.1, 0.2, 0.3, 0.4]
    ylabels = [0, 0.1, 0.2, 0.3, 0.4]
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=-0.045, top=0.23)

    # Legend
    axs[0].legend(fontsize=fontsize, ncol=3, loc=0, frameon=False)

    # Save the figure
    file_name = os.path.join(save_dir,
        f'diff_encoding_accuracy_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Channel and time-average encoding accuracy stats
# =============================================================================
# Compute the channel- and time-average encoding accuracy
correlation_avg = {}
pval_correlation_avg = {}
# Loop across models
for model in correlation.keys():
    # Average the encoding accuracy results across occipital and parietal
    # channels, and across time points from 60ms after stimulus onset
    idx_time = np.where(times >= 0.06)[0]
    correlation_avg[model] = np.mean(correlation[model][:,:2,idx_time], # only average across occipital and parietal channels
        (1, 2))
    pval_correlation_avg[model] = ttest_1samp(correlation_avg[model], 0,
        alternative='greater')[1]

# Compute the channel- and time-average encoding accuracy difference
diff_correlation_avg = {}
pval_diff_correlation_avg = {}
# Loop across models
for model in diff_correlation.keys():
    # Average the encoding accuracy difference results across occipital and
    # parietal channels, and across time points from 60ms after stimulus onset
    idx_time = np.where(times >= 0.06)[0]
    diff_correlation_avg[model] = np.mean(diff_correlation[model][:,:2,idx_time], # only average across occipital and parietal channels
        (1, 2))
    pval_diff_correlation_avg[model] = ttest_1samp(diff_correlation_avg[model],
        0, alternative='two-sided')[1]

# Print the encoding accuracy results
for model in correlation_avg.keys():
    print(f'Model: {model}')
    print(f'Encoding accuracy: {np.mean(correlation_avg[model])}')
    print(f'Encoding accuracy p-value: {pval_correlation_avg[model]}')

# Print the encoding accuracy result differences
for model in diff_correlation_avg.keys():
    print(f'Model: {model}')
    print(f'Encoding accuracy difference: {np.mean(diff_correlation_avg[model])}')
    print(f'Encoding accuracy difference p-value: {pval_diff_correlation_avg[model]}')


# =============================================================================
# Plot the noise analysis results
# =============================================================================
# Plot colors
colors = [(139/255, 0/255, 0/255), (0/255, 0/255, 0/255)]

# Loop across models
for model in corr_iv_iv.keys():

    # Plot the noise analysis results
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(10, 7.5))
    axs = np.reshape(axs, (-1))

    # Plot the in vivo vs. in silico correlations and CIs
    x = np.arange(corr_iv_iv[model].shape[1])
    y = np.repeat(np.mean(corr_iv_is[model]), len(x))
    axs[0].plot(x, y, color=colors[0], linewidth=2,
        label='In silico vs. in vivo target')
    axs[0].fill_between(x, ci_corr_iv_is[model][1], ci_corr_iv_is[model][0],
        color=colors[0], alpha=.1)

    # Plot the in vivo vs. in vivo correlations and CIs
    axs[0].plot(x, np.mean(corr_iv_iv[model], 0), color=colors[1], linewidth=2,
        label='In vivo vs. in vivo target')
    axs[0].fill_between(x, ci_corr_iv_iv[model][1], ci_corr_iv_iv[model][0],
        color=colors[1], alpha=.1)

    # Plot the significance
    sig_l = np.empty(len(sig_less[model]))
    sig_g = np.empty(len(sig_greater[model]))
    sig_l[:] = np.nan
    sig_g[:] = np.nan
    sig_l[sig_less[model]] = 0.37
    sig_g[sig_greater[model]] = 0.37
    plt.scatter(x, sig_l, s=50, color=colors[0])
    plt.scatter(x, sig_g, s=50, color=colors[1])

    # x-axis parameters
    axs[0].set_xlabel('In vivo trials', fontsize=fontsize)
    xticks = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    xlabels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[0].set_xlim(left=min(x), right=max(x))

    # y-axis parameters
    axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.1, 0.2, 0.3, 0.4]
    ylabels = [0, 0.1, 0.2, 0.3, 0.4]
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=0, top=0.4)

    # Legend
    plt.legend(ncol=1, fontsize=fontsize, loc=0, frameon=False)

    # Save the figure
    file_name = os.path.join(save_dir, f'noise_analysis_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()