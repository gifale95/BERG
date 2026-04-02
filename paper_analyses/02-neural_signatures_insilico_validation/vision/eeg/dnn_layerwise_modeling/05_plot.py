"""Plot the RSA scores between in silico EEG responses and behavioral
embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
channels : str
    EEG channel type(s) retained for the analyses. Possible values are:
    'O' (occipital), 'P' (posterior), 'T' (temporal), 'C' (central),
    'F' (frontal). Alternatively, the list can also contain the names of the
    individual channels used.
dnn_model : str
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
from scipy.stats import ttest_1samp


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O-P', type=str)
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'dnn_layerwise_modeling', 'stats', args.encoding_model, 'stats_channels-'+
    args.channels+'_dnn_model-'+args.dnn_model+'.npy')

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
# Get the DNN layers
# =============================================================================
if args.dnn_model == 'alexnet':
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
elif args.dnn_model == 'resnet50':
    model_layers = [
        'layer1.2.relu_2',
        'layer2.3.relu_2',
        'layer3.5.relu_2',
        'layer4.2.relu_2',
        'fc'
        ]


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
# Plot the EEG pairwise decoding results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the RSA subject-average results
axs[0].plot(times, np.mean(decoding, 0), color='k', linewidth=2)

# Plot the peak time point
peak = times[np.argmax(np.mean(decoding, 0))]
max_dec = max(np.mean(decoding, 0))
axs[0].scatter(peak, max_dec, color='k', s=200, marker='o',
    edgecolors='k', linewidths=1, zorder=3)

# Plot the confidence intervals
axs[0].fill_between(times, ci_decoding[1], ci_decoding[0], color='k',
    alpha=.2)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
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
# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(model_layers))

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(10, 7.5))
axs = np.reshape(axs, (-1))

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.25, label='_nolegend_')

# Loop across channel groups
for c, key in enumerate(model_layers):

    # Plot the RSA subject-average results
    axs[0].plot(times, np.mean(rsa[key], 0), color=colors[c], linewidth=2,
        label=key)

    # Plot the confidence intervals
    axs[0].fill_between(times, ci_rsa[key][1], ci_rsa[key][0], color=colors[c],
        alpha=.1)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.05, 0.1, 0.15, 0.2]
ylabels = [0, 0.05, 0.1, 0.15, 0.2]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-.02, top=.2)

# Legend
axs[0].legend(fontsize=15, ncol=1, loc=0, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'rsa_channels-'+args.channels+'_dnn_model-'+
    args.dnn_model+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the layerwise peak latency
# =============================================================================
# Print the subject average peak-latency correlation score and p-value
p_val = ttest_1samp(rsa_peak_latency_dnn_layer_corr, 0,
    alternative='greater')[1]
print((f'RSA peak latency - DNN layer correlation (subject average): '),
    (f'{np.mean(rsa_peak_latency_dnn_layer_corr):.3f} p = {p_val}'))


fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(7.5, 7.5))
axs = np.reshape(axs, (-1))

# Plot the layerwise peak latency (subject average)
layer_nums = np.arange(1, len(rsa_peak_latency)+1)
peak_latency_vals = np.array([rsa_peak_latency[key] for key in model_layers]) * 1000
axs[0].plot(layer_nums, np.mean(peak_latency_vals, 1), color='k', linewidth=2)
axs[0].scatter(layer_nums, np.mean(peak_latency_vals, 1), s=75, color='k')

# Plot the layerwise peak latency (single subjects)
for s in range(len(args.subjects)):
    axs[0].plot(layer_nums, peak_latency_vals[:,s], color='k', linewidth=1,
        alpha=.1, zorder=1)
    for l in range(len(layer_nums)):
        axs[0].scatter(layer_nums[l], peak_latency_vals[l,s], s=25, color='k',
            alpha=.25, zorder=2)

# Plot the confidence intervals
conf_int = np.array([ci_rsa_peak_latency[key] for key in model_layers]) * 1000
conf_int[:,0] = np.mean(peak_latency_vals, 1) - conf_int[:,0]
conf_int[:,1] = conf_int[:,1] - np.mean(peak_latency_vals, 1)
conf_int = np.transpose(conf_int)
axs[0].errorbar(layer_nums, np.mean(peak_latency_vals, 1), yerr=conf_int,
    fmt="none", ecolor='k', elinewidth=1, capsize=3)

# x-axis parameters
axs[0].set_xlabel('DNN layer', fontsize=fontsize)
xticks = layer_nums
xlabels = layer_nums
plt.xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=layer_nums[0]-0.75, right=layer_nums[-1]+0.75)

# y-axis parameters
axs[0].set_ylabel("Peak latency (ms)", fontsize=fontsize)
yticks = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
ylabels = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
plt.yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=0, top=450)

# Save the figure
file_name = os.path.join(save_dir, 'layerwise_peak_latency_channels-'+
    args.channels+'_dnn_model-'+args.dnn_model+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')