"""Plot the RSA results between t-fMRI time point RSMs, and the DNN layerwise
activation RSMs.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
roi : list
    List of used ROIs.
dnn : str
    Name of the used DNN. Possible values are 'dinov2l'.
images : str
    If 'things_eeg_2_vivo', use the in vivo EEG responses for the 200 THINGS
    EEG2 test images.
    If 'things_eeg_2_silico', use the in silico EEG responses for the 200
    THINGS EEG2 test images.
    If 'nsd_515_shared', use the in silico EEG responses for the 515 NSD shared
    images.
normalize: int
    If '1', normalize the RSA results to the range [0, 1]. If '0', do not
    normalize.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import linregress
from berg import BERG

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--rois', default=['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA', 'PPA'], type=list)
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--normalize', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Plot DNN layerwise RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the DNN layerwise RSA results, and average them across fMRI subjects
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'dnn_layerwise_rsa')

# Loop across ROIs
dnn_layerwise_rsa = {}
for r, roi in enumerate(args.rois):

    # Loop across fMRI subjects
    for s, sub in enumerate(args.fmri_subjects):

        # Load the results
        file_name = (f'dnn_layerwise_rsa_sub-{sub:02d}_roi-{roi}_'
            f'images-{args.images}.npy')
        dnn_layerwise_rsa_sub = np.load(os.path.join(data_dir, file_name))

        # Sum the results across fMRI subjects
        if s == 0:
            dnn_layerwise_rsa[roi] = dnn_layerwise_rsa_sub
        else:
            dnn_layerwise_rsa[roi] += dnn_layerwise_rsa_sub
        del dnn_layerwise_rsa_sub

    # Average the results across fMRI subjects
    dnn_layerwise_rsa[roi] /= len(args.fmri_subjects)


# =============================================================================
# Select EEG time points between 60-400ms (where the EEG responses are most
# reliable)
# =============================================================================
# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = np.round(metadata_eeg['eeg']['times'], 3)

# EEG time point selection
idx_times = (times >= 0.06) & (times <= 0.4)
times = times[idx_times]
for key, val in dnn_layerwise_rsa.items():
    dnn_layerwise_rsa[key] = val[:,idx_times]


# =============================================================================
# Normalize the RSA results to the range [0, 1]
# =============================================================================
# Normalize the RSA results to the range [0, 1], independently for each EEG
# time point, across layers, as as to emphasize the DNN layer preference of
# each time point, regardless of the absolute RSA values.

if args.normalize == 1:

    for key, val in dnn_layerwise_rsa.items():
        dnn_layerwise_rsa[key] = (val - np.min(val, 0)) / \
            (np.max(val, 0) - np.min(val, 0))


# =============================================================================
# Create the plot save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'plots', 'dnn_layerwise_rsa')
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
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot the RSA results (2d heatmaps)
# =============================================================================
# Create the plot figure
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(dnn_layerwise_rsa.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the DNN layerwise RSA results
    im = axs[i].imshow(val, aspect='auto', cmap='magma_r', origin='lower')

    # Plot title
    axs[i].set_title(key, fontsize=fontsize)

    # x-axis parameters
    if i in [4, 5, 6]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
        xticks = [8, 28, 48, 68]
        xlabels = [100, 200, 300, 400]
        axs[i].set_xticks(ticks=xticks, labels=xlabels)

    # y-axis parameters
    if i in [0, 3, 6]:
        axs[i].set_ylabel('DNN layer', fontsize=fontsize)
    yticks = [0, 5, 11, 17, 23]
    ylabels = [1, 6, 12, 18, 24]
    axs[i].set_yticks(ticks=yticks, labels=ylabels)

    # Colorbar
    label = "Pearson's $r$"
    fig.colorbar(im, label=label, fraction=0.04, pad=0.04)

# Save the figure
file_name = os.path.join(save_dir, f'dnn_layerwise_rsa_2d_heatmap_'
    f'dnn-{args.dnn}_normalize-{args.normalize}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the RSA results (best DNN layer per t-fMRI time point)
# =============================================================================
# Get the best DNN layer for each t-fMRI time point (based on the average
# of the top-5 DNN layers to get a more robust estimate)
best_dnn_layer = {}
for key, val in dnn_layerwise_rsa.items():
    idx_best = np.mean(np.argsort(val, 0)[-1:], 0) # !!! idx_best = np.mean(np.argsort(val, 0)[-5:], 0)
    best_dnn_layer[key] = idx_best
    del idx_best

# Compute the correlation between best DNN layers and t-fMRI time points
corr_dnn_layer_tfmri_times = {}
for key, val in best_dnn_layer.items():
    corr_dnn_layer_tfmri_times[key] = spearmanr(times, val)[0]

# Fit a regression line between best DNN layers and t-fMRI time points
regression = {}
for key, val in best_dnn_layer.items():
    regression[key] = linregress(times, val)

# Create the plot figure
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
color = (139/255, 0/255, 0/255)
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(best_dnn_layer.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the best DNN layer for each t-fMRI time point
    axs[i].plot(times, val, color=color, linewidth=2, alpha=1,
        label='_nolegend_')

    # Plot the correlation between best DNN layers and t-fMRI time points
    axs[i].text(0.1, 21, f'$ρ$ = {corr_dnn_layer_tfmri_times[key]:.2f}',
        color='k', fontsize=fontsize, ha='left')

    # Plot the regression line between DNN layers and t-fMRI time points
    slope = regression[key].slope
    intercept = regression[key].intercept
    x_fit = np.array([min(times), max(times)])
    y_fit = intercept + slope * x_fit
    axs[i].plot(x_fit, y_fit, color='k', linewidth=2, linestyle='--',
        label=f'$y$ = {slope:.2f}$x$ + {intercept:.2f}', alpha=0.5)

    # Plot title
    axs[i].set_title(key, fontsize=fontsize)

    # x-axis parameters
    if i in [6, 7, 8]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [0.1, 0.2, 0.3, 0.4]
    xlabels = [100, 200, 300, 400]
    axs[i].set_xticks(ticks=xticks, labels=xlabels)
    axs[i].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    if i in [0, 3, 6]:
        axs[i].set_ylabel('DNN layer', fontsize=fontsize)
    yticks = [0, 5, 11, 17, 23]
    ylabels = [1, 6, 12, 18, 24]
    axs[i].set_yticks(ticks=yticks, labels=ylabels)
    axs[i].set_ylim(bottom=min(yticks), top=max(yticks))

    # Legend
    axs[i].legend(frameon=False, loc=4)

# Save the figure
file_name = os.path.join(save_dir, f'dnn_layerwise_rsa_best_dnn_layer_'
    f'dnn-{args.dnn}_normalize-{args.normalize}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the RSA results (best t-fMRI time point per DNN layer)
# =============================================================================
# Get the t-fMRI time point for each DNN layer (based on the average of the
# top-5 t-fMRI time points to get a more robust estimate)
best_tfmri_time = {}
for key, val in dnn_layerwise_rsa.items():
    idx_best = np.mean(np.argsort(np.transpose(val), 0)[-1:], 0).astype(int) # !!! idx_best = np.mean(np.argsort(np.transpose(val), 0)[-5:], 0).astype(int)
    best_tfmri_time[key] = times[idx_best]
    del idx_best

# Compute the correlation between DNN layers and best t-fMRI time points
corr_dnn_layer_tfmri_times = {}
for key, val in best_tfmri_time.items():
    layers = np.arange(len(val)) + 1
    corr_dnn_layer_tfmri_times[key] = spearmanr(layers, val)[0]

# Fit a regression line between DNN layers and best t-fMRI time points
regression = {}
for key, val in best_tfmri_time.items():
    layers = np.arange(len(val))
    regression[key] = linregress(layers, val)

# Create the plot figure
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
color = (139/255, 0/255, 0/255)
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(best_tfmri_time.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the best t-fMRI time point for each DNN layer
    layers = np.arange(len(val)) + 1
    axs[i].plot(layers, val, color=color, linewidth=2, alpha=1,
        label='_nolegend_')

    # Plot the correlation between DNN layers and best t-fMRI time points
    axs[i].text(2, 0.35, f'$ρ$ = {corr_dnn_layer_tfmri_times[key]:.2f}',
        color='k', fontsize=fontsize)

    # Plot the regression line between DNN layers and t-fMRI time points
    slope = regression[key].slope
    intercept = regression[key].intercept
    x_fit = np.array([min(layers), max(layers)])
    y_fit = intercept + slope * x_fit
    axs[i].plot(x_fit, y_fit, color='k', linewidth=2, linestyle='--',
        label=f'$y$ = {slope:.2f}$x$ + {intercept:.2f}', alpha=0.5)

    # Plot title
    axs[i].set_title(key, fontsize=fontsize)

    # x-axis parameters
    if i in [6, 7, 8]:
        axs[i].set_xlabel('DNN layer', fontsize=fontsize)
    xticks = [1, 6, 12, 18, 24]
    xlabels = [1, 6, 12, 18, 24]
    axs[i].set_xticks(ticks=xticks, labels=xlabels)
    axs[i].set_xlim(left=min(layers), right=max(layers))

    # y-axis parameters
    if i in [0, 3, 6]:
        axs[i].set_ylabel('Time (ms)', fontsize=fontsize)
    yticks = [0.1, 0.2, 0.3, 0.4]
    ylabels = [100, 200, 300, 400]
    axs[i].set_yticks(ticks=yticks, labels=ylabels)
    axs[i].set_ylim(bottom=min(times), top=max(times))

    # Legend
    axs[i].legend(frameon=False, loc=4)

# Save the figure
file_name = os.path.join(save_dir, f'dnn_layerwise_rsa_best_tfmri_time_point_'
    f'dnn-{args.dnn}_normalize-{args.normalize}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the RSA results (all t-fMRI time points per DNN layer)
# =============================================================================
# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(times))

# Create the plot figure
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(dnn_layerwise_rsa.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the correlation of each t-fMRI time point as a function of DNN layer
    layers = np.arange(len(val)) + 1
    for t, time in enumerate(times):
        axs[i].plot(layers, val[:,t], color=colors[t], linewidth=2, alpha=.75,
            label='_nolegend_')

    # Plot title
    axs[i].set_title(key, fontsize=fontsize)

    # x-axis parameters
    if i in [6, 7, 8]:
        axs[i].set_xlabel('DNN layer', fontsize=fontsize)
    xticks = [1, 6, 12, 18, 24]
    xlabels = [1, 6, 12, 18, 24]
    axs[i].set_xticks(ticks=xticks, labels=xlabels)
    axs[i].set_xlim(left=min(layers), right=max(layers))

    # y-axis parameters
    if i in [0, 3, 6]:
        axs[i].set_ylabel("Pearson's $r$", fontsize=fontsize)
    # yticks = [0, 0.2, 0.4, 0.595]
    # ylabels = [0, 200, 400, 600]
    # axs[i].set_yticks(ticks=yticks, labels=ylabels)
    # axs[i].set_ylim(bottom=min(times), top=max(times))

# Save the figure
file_name = os.path.join(save_dir,
    f'dnn_layerwise_rsa_tfmri_time_as_function_of_dnn_layer_dnn-{args.dnn}_'
    f'normalize-{args.normalize}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the RSA results (all DNN layers per t-fMRI time point)
# =============================================================================
# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
n_layers = 24
colors = sample_cmap(n_layers)

# Create the plot figure
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(dnn_layerwise_rsa.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the correlation of each DNN layer as a function of t-fMRI time points
    for l in range(n_layers):
        axs[i].plot(times, val[l], color=colors[l], linewidth=2, alpha=.9,
            label='_nolegend_')

    # Plot title
    axs[i].set_title(key, fontsize=fontsize)

    # x-axis parameters
    if i in [6, 7, 8]:
        axs[i].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [0.1, 0.2, 0.3, 0.4]
    xlabels = [100, 200, 300, 400]
    axs[i].set_xticks(ticks=xticks, labels=xlabels)
    axs[i].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    if i in [0, 3, 6]:
        axs[i].set_ylabel("Pearson's $r$", fontsize=fontsize)
    # yticks = [0, 0.2, 0.4, 0.595]
    # ylabels = [0, 200, 400, 600]
    # axs[i].set_yticks(ticks=yticks, labels=ylabels)
    # axs[i].set_ylim(bottom=min(times), top=max(times))

# Save the figure
file_name = os.path.join(save_dir,
    f'dnn_layerwise_rsa_dnn_layer_as_function_of_tfmri_time_dnn-{args.dnn}_'
    f'normalize-{args.normalize}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)