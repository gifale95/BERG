"""Plot the encoding accuracy and noise analysis results fro BERG's EEG
encoding models trained on THINGS EEG2.

Parameters
----------
encoding_models : list
    The names of BERG's encoding models used for generating the in silico EEG
    responses.
subjects : list of int
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['eeg-things_eeg_2-vit_b_32', 'eeg-things_eeg_2-alexnet', 'eeg-things_eeg_2-alexnet_untrained'])
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


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
# Load the encoding accuracy results, and create the plot saving directory
# =============================================================================
# Empty result variables
correlation = {}
noise_ceiling = {}

# Loop across encoding models
for encoding_model in tqdm(args.encoding_models):

    # Result directory
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'encoding_accuracy', 'encoding_accuracy', encoding_model,
        'encoding_accuracy.npy')
    results = np.load(results_dir, allow_pickle=True).item()

    # Load the results
    correlation[encoding_model] = np.array(results['correlation'])
    noise_ceiling[encoding_model] = np.array(results['noise_ceiling'])
    corr_iv_iv = results['corr_iv_iv']
    corr_iv_is = results['corr_iv_is']
    ch_names = results['metadata'][0]['eeg']['ch_names']
    times = results['metadata'][0]['eeg']['times']

    # Plot save directory
    save_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'encoding_accuracy', 'plots', encoding_model)
    os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
    # Average the correlation scores across channels from the same channel
    # group
    correlation_chan_avg = []
    # Loop across channel groups
    chan_groups = ['O', 'P', 'T', 'C', 'F']
    chan_groups_names = ['Occipital', 'Parietal', 'Temporal', 'Central', 'Frontal']
    for chan in chan_groups:
        # Loop across EEG channels, and select the ones from the channel group of
        # interest
        idx_chan = []
        for c, ch_name in enumerate(ch_names):
            if chan in ch_name:
                idx_chan.append(c)
        idx_chan = np.array(idx_chan)
        # Average the correlation scores across the selected channels
        correlation_chan_avg.append(np.mean(
            correlation[encoding_model][:,idx_chan], 1))
    # Convert to numpy array
    correlation_chan_avg = np.array(correlation_chan_avg)

    # Compute the confidence intervals
    n_iter = 100000
    ci = np.zeros((len(chan_groups), 2, len(times)))
    # Empty bootstrap distribution arrays
    ci_dist = np.zeros((n_iter, len(chan_groups), len(times)))
    # Compute the bootstrap distributions
    for i in range(n_iter):
        idx = resample(np.arange(len(args.subjects)))
        ci_dist[i] = np.mean(correlation_chan_avg[:,idx], 1)
    # Compute the confidence intervals
    ci[:,0] = np.percentile(ci_dist, 2.5, axis=0)
    ci[:,1] = np.percentile(ci_dist, 97.5, axis=0)

    # Plot colors
    colors = sample_cmap(len(chan_groups_names))

    # Plot the encoding accuracy
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(10, 7.5))
    axs = np.reshape(axs, (-1))
    # Plot the chance and stimulus onset dashed lines
    axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')
    # Loop across channel groups
    for c, key in enumerate(chan_groups_names):
        # Plot the encoding accuracy
        axs[0].plot(times, np.mean(correlation_chan_avg[c], 0), color=colors[c],
            linewidth=2, label=key)
        # Plot the confidence intervals
        axs[0].fill_between(times, ci[c][1], ci[c][0], color=colors[c],
            alpha=.1)
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
    axs[0].set_ylim(bottom=-.05, top=0.8)
    # Legend
    axs[0].legend(fontsize=fontsize, ncol=3, loc=0, frameon=False)
    # Save the figure
    file_name = os.path.join(save_dir, 'encoding_accuracy_nsdsynthetic.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the noise analysis results
# =============================================================================
    # Only select results from time points from 60ms after stimulus onset
    idx = np.where(times >= 0.06)[0]
    corr_iv_iv = corr_iv_iv[:,:,:,idx]
    corr_iv_is = corr_iv_is[:,:,idx]
    times = times[idx]

    # Only select results from occipital and parietal channels
    idx_chan = []
    for c, ch_name in enumerate(ch_names):
        if 'O' in ch_name or 'P' in ch_name:
            idx_chan.append(c)
    idx_chan = np.array(idx_chan)
    corr_iv_iv = corr_iv_iv[:,:,idx_chan]
    corr_iv_is = corr_iv_is[:,idx_chan]

    # Average the results across channels and times
    corr_iv_iv = np.mean(corr_iv_iv, axis=(2,3))
    corr_iv_is = np.mean(corr_iv_is, axis=(1,2))

    # Compute the confidence intervals
    n_iter = 100000
    ci_corr_iv_iv = np.zeros((2, corr_iv_iv.shape[1]), dtype=np.float32)
    ci_corr_iv_is = np.zeros((2), dtype=np.float32)
    # Empty bootstrap distribution arrays
    ci_dist_iv_iv = np.zeros((n_iter, corr_iv_iv.shape[1]))
    ci_dist_iv_is = np.zeros((n_iter))
    # Compute the bootstrap distributions
    for i in range(n_iter):
        idx = resample(np.arange(len(args.subjects)))
        ci_dist_iv_iv[i] = np.mean(corr_iv_iv[idx], 0)
        ci_dist_iv_is[i] = np.mean(corr_iv_is[idx], 0)
    # Compute the confidence intervals
    ci_corr_iv_iv[0] = np.percentile(ci_dist_iv_iv, 2.5, axis=0)
    ci_corr_iv_iv[1] = np.percentile(ci_dist_iv_iv, 97.5, axis=0)
    ci_corr_iv_is[0] = np.percentile(ci_dist_iv_is, 2.5, axis=0)
    ci_corr_iv_is[1] = np.percentile(ci_dist_iv_is, 97.5, axis=0)

    # Compute the significance
    p_val_less = ttest_rel(corr_iv_iv, np.repeat(np.reshape(
        corr_iv_is, (len(corr_iv_is), 1)), corr_iv_iv.shape[1], axis=1),
        axis=0, alternative='less')[1]
    p_val_greater = ttest_rel(corr_iv_iv, np.repeat(np.reshape(
        corr_iv_is, (len(corr_iv_is), 1)), corr_iv_iv.shape[1], axis=1),
        axis=0, alternative='greater')[1]
    # Multiple comparison correction
    idx_sig_less = multipletests(p_val_less, 0.05, 'fdr_bh')[0]
    idx_sig_greater = multipletests(p_val_greater, 0.05, 'fdr_bh')[0]
    sig_less = np.empty(len(idx_sig_less))
    sig_greater = np.empty(len(idx_sig_greater))
    sig_less[:] = np.nan
    sig_greater[:] = np.nan
    sig_less[idx_sig_less] = 0.35
    sig_greater[idx_sig_greater] = 0.35

    # Plot colors
    colors = [(139/255, 0/255, 0/255), (0/255, 0/255, 0/255)]

    # Plot the noise analysis results
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(10, 7.5))
    axs = np.reshape(axs, (-1))
    # Plot the in vivo vs. in silico correlations and CIs
    y = np.repeat(np.mean(corr_iv_is), len(x))
    axs[0].plot(x, y, color=colors[0], linewidth=2,
        label='In silico vs. in vivo target')
    axs[0].fill_between(x, ci_corr_iv_is[1], ci_corr_iv_is[0], color=colors[0],
        alpha=.1)
    # Plot the in vivo vs. in vivo correlations and CIs
    x = np.arange(corr_iv_iv.shape[1])
    axs[0].plot(x, np.mean(corr_iv_iv, 0), color=colors[1], linewidth=2,
        label='In vivo vs. in vivo target')
    axs[0].fill_between(x, ci_corr_iv_iv[1], ci_corr_iv_iv[0], color=colors[1],
        alpha=.1)
    # Plot the significance
    plt.scatter(x, sig_less, s=50, color=colors[0])
    plt.scatter(x, sig_greater, s=50, color=colors[1])
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
    file_name = os.path.join(save_dir, 'noise_analysis.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the encoding accuracy difference between models
# =============================================================================
# Comparisons:
# 1: eeg-things_eeg_2-vit_b_32 vs. eeg-things_eeg_2-alexnet
# 2: eeg-things_eeg_2-vit_b_32 vs. eeg-things_eeg_2-alexnet_untrained
# 3: eeg-things_eeg_2-alexnet vs. eeg-things_eeg_2-alexnet_untrained
comparisons = [['vit_b_32', 'alexnet'], ['vit_b_32', 'alexnet_untrained'],
    ['alexnet', 'alexnet_untrained']]

# Loop across comparisons
for comp in comparisons:

    # Average the correlation scores across channels from the same channel
    # group, and subtract them between encoding models
    diff_chan_avg = []
    # Loop across channel groups
    chan_groups = ['O', 'P', 'T', 'C', 'F']
    chan_groups_names = ['Occipital', 'Parietal', 'Temporal', 'Central', 'Frontal']
    for chan in chan_groups:
        # Loop across EEG channels, and select the ones from the channel group
        # of interest
        idx_chan = []
        for c, ch_name in enumerate(ch_names):
            if chan in ch_name:
                idx_chan.append(c)
        idx_chan = np.array(idx_chan)
        # Average the correlation scores across the selected channels
        diff = np.mean(correlation[f'eeg-things_eeg_2-{comp[0]}'][:,idx_chan], 1) - \
            np.mean(correlation[f'eeg-things_eeg_2-{comp[1]}'][:,idx_chan], 1)
        diff_chan_avg.append(diff)
    # Convert to numpy array
    diff_chan_avg = np.array(diff_chan_avg)

    # Compute the confidence intervals
    n_iter = 100000
    ci = np.zeros((len(chan_groups), 2, len(times)))
    # Empty bootstrap distribution arrays
    ci_dist = np.zeros((n_iter, len(chan_groups), len(times)))
    # Compute the bootstrap distributions
    for i in range(n_iter):
        idx = resample(np.arange(len(args.subjects)))
        ci_dist[i] = np.mean(diff_chan_avg[:,idx], 1)
    # Compute the confidence intervals
    ci[:,0] = np.percentile(ci_dist, 2.5, axis=0)
    ci[:,1] = np.percentile(ci_dist, 97.5, axis=0)

    # Compute the significance
    p_val = ttest_1samp(diff_chan_avg, 0, axis=1,
        alternative='greater')[1]
    # Multiple comparison correction
    p_val_shape = p_val.shape
    p_val = np.reshape(p_val, (-1))
    sig = multipletests(p_val, 0.05, 'fdr_bh')[0]
    sig = np.reshape(sig, p_val_shape)
    sig_plot = np.empty(sig.shape)
    sig_plot[:] = np.nan
    y = 0.27
    for c in range(len(chan_groups)):
        sig_plot[c,sig[c,:]] = y
        idx_sig = np.where(sig[c])[0]
        y += 0.01

    # Plot colors
    colors = sample_cmap(len(chan_groups_names))

    # Plot the encoding accuracy difference
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
        figsize=(10, 7.5))
    axs = np.reshape(axs, (-1))
    # Plot the chance and stimulus onset dashed lines
    axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
        linewidth=2, alpha=.25, label='_nolegend_')
    # Loop across channel groups
    for c, key in enumerate(chan_groups_names):
        # Plot the encoding accuracy difference
        axs[0].plot(times, np.mean(diff_chan_avg[c], 0), color=colors[c],
            linewidth=2, label=key)
        # Plot the confidence intervals
        axs[0].fill_between(times, ci[c][1], ci[c][0], color=colors[c],
            alpha=.1)
        # Plot the significance
        plt.scatter(times, sig_plot[c], s=50, color=colors[c])
    # x-axis parameters
    axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
    xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[0].set_xlim(left=min(times), right=max(times))
    # y-axis parameters
    axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.1, 0.2, 0.3, 0.4]
    ylabels = [0, 0.1, 0.2, 0.3, 0.4]
    plt.yticks(ticks=yticks, labels=ylabels)
    axs[0].set_ylim(bottom=-.05, top=0.32)
    # Legend
    axs[0].legend(fontsize=10, ncol=3, loc=0, frameon=False)
    # Save the figure
    save_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'encoding_accuracy', 'plots')
    file_name = os.path.join(save_dir,
        f'encoding_accuracy_{comp[0]}_minus_{comp[1]}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()