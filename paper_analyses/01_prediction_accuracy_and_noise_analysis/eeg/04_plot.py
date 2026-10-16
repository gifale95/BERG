"""Plot the prediction accuracy and noise analysis results for BERG's EEG
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
results_dir = os.path.join(args.berg_dir, 'prediction_accuracy_noise_analysis',
    'eeg', 'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
correlation = results['correlation']
ci_correlation = results['ci_correlation']
sig_correlation = results['sig_correlation']
corr_iv_iv = results['corr_iv_iv']
corr_iv_is = results['corr_iv_is']
ci_corr_iv_is = results['ci_corr_iv_is']
ci_corr_iv_iv = results['ci_corr_iv_iv']
sig_less = results['sig_less']
sig_greater = results['sig_greater']
times = results['metadata'][0]['eeg']['times']
ch_names = results['metadata'][0]['eeg']['ch_names']

# Plot save directory
save_dir = os.path.join(args.berg_dir, 'prediction_accuracy_noise_analysis',
    'eeg', 'plots')
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


# =============================================================================
# Plot the prediction accuracy results
# =============================================================================
# Plot color
color = (30/255, 114/255, 179/255)

# Loop across models
for model in correlation.keys():

    # Plot the prediction accuracy
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(7, 5))
    axs = np.reshape(axs, (-1))

    # Plot the chance and stimulus onset dashed lines
    axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')

    # Plot the prediction accuracy
    axs[0].plot(times, np.mean(correlation[model], 0), color=color, linewidth=2)

    # Plot the confidence intervals
    axs[0].fill_between(times, ci_correlation[model][0],
        ci_correlation[model][1], color=color, alpha=.2)

    # Plot the significance markers
    sig = np.empty(len(times))
    sig[:] = np.nan
    sig[sig_correlation[model]] = -0.025
    plt.scatter(times, sig, s=30, color=color)

    # x-axis parameters
    axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [0, .2, .4, .595]
    xlabels = [0, 200, 400, 600]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[0].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=-.05, top=0.7)

    # Save the figure
    file_name = os.path.join(save_dir, f'prediction_accuracy_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Channel and time-average prediction accuracy stats
# =============================================================================
# Compute the channel- and time-average prediction accuracy
correlation_avg = {}
pval_correlation_avg = {}
# Loop across models
for model in correlation.keys():
    # Average the prediction accuracy results across occipital and parietal
    # channels, and across time points from 60ms after stimulus onset
    idx_time = np.where(times >= 0.06)[0]
    correlation_avg[model] = np.mean(correlation[model][:,idx_time], 1)
    pval_correlation_avg[model] = ttest_1samp(correlation_avg[model], 0,
        alternative='greater')[1]

# Print the prediction accuracy results
for model in correlation_avg.keys():
    print(f'Model: {model}')
    print(f'Prediction accuracy: {np.mean(correlation_avg[model])}')
    print(f'Prediction accuracy p-value: {pval_correlation_avg[model]}')


# =============================================================================
# Plot the noise analysis results
# =============================================================================
# Plot colors
colors = [(30/255, 114/255, 179/255), (23/255, 141/255, 32/255)]

# Loop across models
for model in corr_iv_iv.keys():

    # Plot the noise analysis results
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(7, 5))
    axs = np.reshape(axs, (-1))

    # Plot the in vivo vs. in silico correlations and CIs
    x = np.arange(corr_iv_iv[model].shape[1])
    y = np.repeat(np.mean(corr_iv_is[model]), len(x))
    axs[0].plot(x, y, color=colors[0], linewidth=2,
        label='In silico vs. in vivo target')
    axs[0].fill_between(x, ci_corr_iv_is[model][1], ci_corr_iv_is[model][0],
        color=colors[0], alpha=.2)

    # Plot the in vivo vs. in vivo correlations and CIs
    axs[0].plot(x, np.mean(corr_iv_iv[model], 0), color=colors[1], linewidth=2,
        label='In vivo vs. in vivo target')
    axs[0].fill_between(x, ci_corr_iv_iv[model][1], ci_corr_iv_iv[model][0],
        color=colors[1], alpha=.2)

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
    xticks = [9, 19, 29, 39, 49, 59]
    xlabels = [10, 20, 30, 40, 50, 60]
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