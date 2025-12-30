"""Plot the RSA scores between in silico EEG responses and behavioral
embeddings.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : str
    EEG channel type(s) retained for the analyses. Possible values are:
    'O' (occipital), 'P' (posterior), 'T' (temporal), 'C' (central),
    'F' (frontal). Alternatively, the list can also contain the names of the
    individual channels used.
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O-P', type=str)
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'dnn_layerwise_modeling', 'stats', 'stats_'+'channels-'+args.channels+
    '_model-'+args.model+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

rsa = results['rsa']
rsa_peak_latency = results['rsa_peak_latency']
rsa_peak_latency_dnn_layer_corr = results['rsa_peak_latency_dnn_layer_corr']
ci_rsa = results['ci_rsa']
ci_rsa_peak_latency = results['ci_rsa_peak_latency']
decoding = results['decoding']
ci_decoding = results['ci_decoding']
times = results['times']


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
colors = [
    (103/255, 78/255, 167/255),
    (166/255, 77/255, 121/255),
    (105/255, 105/255, 105/255),
    (169/255, 169/255, 169/255),
    (100/255, 149/255, 237/255),
    (90/255, 130/255, 200/255),
    (40/255, 65/255, 150/255),
    (0/255, 0/255, 0/255)
    ]


# =============================================================================
# Plot the EEG pairwise decoding results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')

# Plot the RSA subject-average results
axs[0].plot(times, np.mean(decoding, 0), color=colors[0], linewidth=2)

# Plot the peak time point
peak = times[np.argmax(np.mean(decoding, 0))]
max_dec = max(np.mean(decoding, 0))
axs[0].scatter(peak, max_dec, color=colors[0], s=200, marker='o',
    edgecolors='k', linewidths=1, zorder=3)

# Plot the confidence intervals
axs[0].fill_between(times, ci_decoding[1], ci_decoding[0], color=colors[0],
    alpha=.2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=47, top=100)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_channels-'+args.channels+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the RSA results
# =============================================================================
# Get the DNN layers
if args.model == 'alexnet':
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
elif args.model == 'resnet50':
    model_layers = [
        'layer1.2.relu_2',
        'layer2.3.relu_2',
        'layer3.5.relu_2',
        'layer4.2.relu_2',
        'fc'
        ]

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')

# Loop across channel groups
for c, key in enumerate(model_layers):

    # Plot the RSA subject-average results
    axs[0].plot(times, np.mean(rsa[key], 0), color=colors[c], linewidth=2,
        label=key)

    # Plot the peak time point
    peak = rsa_peak_latency[key]
    max_rsa = max(np.mean(rsa[key], 0))
    axs[0].scatter(peak, max_rsa, color=colors[c], s=200, marker='o',
        edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')
    ci_low = peak - ci_rsa_peak_latency[key][0]
    ci_up = ci_rsa_peak_latency[key][1] - peak
    conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
    axs[0].errorbar(peak, max_rsa, xerr=conf_int, fmt="none", ecolor='k',
        elinewidth=1, capsize=3)

    # Plot the confidence intervals
    axs[0].fill_between(times, ci_rsa[key][1], ci_rsa[key][0], color=colors[c],
        alpha=.1)

    # Plot the significance time points
    # sig = np.empty(len(times))
    # sig[:] = np.nan
    # sig[sig_rsa[chan]] = -.015
    # plt.scatter(times, sig, s=100, color=colors[c])

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("RSA (Pearson's $r$)", fontsize=fontsize)
yticks = [0, 0.1, 0.2]
ylabels = [0, 0.1, 0.2]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-.02, top=.2)

# Legend
axs[0].legend(fontsize=15, ncol=1, loc=0, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'rsa_channels-'+args.channels+'_model-'+
    args.model+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')