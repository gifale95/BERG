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
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--rois', default=['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA', 'PPA'], type=list)
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Plot <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the DNN layerwise RSA stats
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution', 'stats',
    f'stats_dnn-{args.dnn}_images-{args.images}.npy')

data = np.load(data_dir, allow_pickle=True).item()

times = data['times']
dnn_layerwise_rsa = data['dnn_layerwise_rsa']
best_dnn_layer = data['best_dnn_layer']
corr_dnn_layer_tfmri_times = data['corr_dnn_layer_tfmri_times']
reg_corr_dnn_layer_tfmri_times = data['reg_corr_dnn_layer_tfmri_times']
ci_dnn_layerwise_rsa = data['ci_dnn_layerwise_rsa']
ci_best_dnn_layer = data['ci_best_dnn_layer']


# =============================================================================
# Create the plot save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution', 'plots')
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
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot the RSA scores of all DNN layers as a function of t-fMRI time points
# =============================================================================
# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('cividis')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
n_layers = 24
colors = sample_cmap(n_layers)

# Create the plot figure
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(dnn_layerwise_rsa.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the correlation of each DNN layer as a function of t-fMRI time
    # points
    for l in range(n_layers):
        axs[i].plot(times, np.mean(val[:,l], axis=0), color=colors[l],
            linewidth=2, alpha=.9, label='_nolegend_')

    # Plot the confidence intervals
    for l in range(n_layers):
        axs[i].fill_between(times, ci_dnn_layerwise_rsa[key][0,l],
            ci_dnn_layerwise_rsa[key][1,l], color=colors[l], alpha=.1)

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
    yticks = [0, 0.1, 0.2, 0.3]
    ylabels = [0, 0.1, 0.2, 0.3]
    axs[i].set_yticks(ticks=yticks, labels=ylabels)
    axs[i].set_ylim(bottom=-0.05, top=0.35)

# Save the figure
file_name = os.path.join(save_dir, f'dnn_layer_as_function_of_tfmri_time_'
    f'dnn-{args.dnn}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the best DNN layer as a function of t-fMRI time points
# =============================================================================
# Create the plot figure
color = (139/255, 0/255, 0/255)
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(35, 35))
axs = np.reshape(axs, -1)

for i, (key, val) in enumerate(best_dnn_layer.items()):

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Plot the best DNN layer for each t-fMRI time point
    axs[i].plot(times, np.mean(val, 0), color=color, linewidth=2, alpha=1,
        label='_nolegend_')

    # Plot the confidence intervals
    axs[i].fill_between(times, ci_best_dnn_layer[key][0],
        ci_best_dnn_layer[key][1], color=color, alpha=.1)

    # Plot the correlation between best DNN layers and t-fMRI time points
    axs[i].text(0.1, 21, f'$ρ$ = {corr_dnn_layer_tfmri_times[key][0]:.2f}',
        color='k', fontsize=fontsize, ha='left')

    # Plot the regression line between DNN layers and t-fMRI time points
    slope = reg_best_dnn_layer_tfmri_times[key].slope
    intercept = reg_best_dnn_layer_tfmri_times[key].intercept
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
file_name = os.path.join(save_dir, f'best_dnn_layer_as_function_of_tfmri_time_'
    f'dnn-{args.dnn}_images-{args.images}.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)