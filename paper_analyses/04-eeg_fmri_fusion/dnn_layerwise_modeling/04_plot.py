"""Plot the DNN layerwise assignment of each t-fMRI vertex and time point.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
dnn_model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import cortex
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'plots', 'surfaceplots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA layerwise assignment results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stats', f'stats_dnn_model-{args.dnn_model}.npy')

results = np.load(data_dir, allow_pickle=True).item()

lh_best_layer = results['lh_best_layer']
rh_best_layer = results['rh_best_layer']
best_layer_roi = results['best_layer_roi']
times = results['times']

# Get the model layers
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
# Load the behavioral modeling RSA results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling', 'stats', 'stats.npy')

results = np.load(data_dir, allow_pickle=True).item()

rsa_roi = results['rsa_roi']
ci_rsa_roi = results['ci_rsa_roi']
del results


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 40
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'


# =============================================================================
# Plot the vertex DNN layer assignment
# =============================================================================
# Loop over EEG time points
for t, time in enumerate(tqdm(times)):

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(lh_best_layer[:,:,t], 0),
        np.nanmean(rh_best_layer[:,:,t], 0))

    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='gist_rainbow',
        vmin=1,
        vmax=len(model_layers),
        with_colorbar=True
        )

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        #height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=25,
        curvature_brightness=0.4,
        with_colorbar=True
        )

    # Add title
    title = f'Time (ms): {np.round(time*1000)}'
    plt.title(title, fontsize=fontsize)

    # Save the figure
    plot_file = os.path.join(save_dir,
        f'rsa_layer_assigment_dnn_model-{args.dnn_model}_time-{t:03d}.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()


# =============================================================================
# Plot the ROI-wise DNN layer assignment
# =============================================================================
# Plot parameters
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

# Define the ROIs to plot
rois = ['early', 'intermediate', 'ventral', 'lateral', 'parietal']

# Loop across ROIs
for r, roi in enumerate(rois):

    # Get the plot data
    x = times
    y = np.mean(rsa_roi[roi], 0)
    c = np.mean(best_layer_roi[roi], 0) + 1 # vector controlling color (add 1 so that the layers start from 1)

    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize colors
    norm = Normalize(vmin=1, vmax=len(model_layers))

    # Create LineCollection
    lc = LineCollection(segments, cmap='turbo_r', norm=norm)
    lc.set_array(c)
    lc.set_linewidth(2)

    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Plot the stimulus onset and chance dashed line
    ax.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
        alpha=.5, label='_nolegend_')

    # Plot the behavioral modeling RSA correlation scores, colored by DNN
    # layer assignment
    ax.add_collection(lc)

    # x-axis parameters
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
    xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
    plt.xticks(ticks=xticks, labels=xlabels)
    ax.set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    ax.set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.1, 0.2, 0.3, 0.4]
    ylabels = [0, 0.1, 0.2, 0.3, 0.4]
    plt.yticks(ticks=yticks, labels=ylabels)
    ax.set_ylim(bottom=-.03, top=.25)

    # Colorbar
    if args.dnn_model == 'alexnet':
        model = 'AlexNet'
    elif args.dnn_model == 'resnet50':
        model = 'ResNet-50'
    plt.colorbar(lc, ax=ax, label=f'{model} layers')

    # Save the figure
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'dnn_layerwise_modeling', 'plots')
    file_name = os.path.join(save_dir, f'layer_assignment_roi-{roi}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot all ROIs on the same graph
# =============================================================================
# Define the ROIs to plot
rois = ['early', 'intermediate', 'ventral', 'lateral', 'parietal']

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

# Plot the stimulus onset and chance dashed line
ax.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--', linewidth=2,
    alpha=.5, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Get the plot data
    x = times
    y = np.mean(rsa_roi[roi], 0)
    c = np.mean(best_layer_roi[roi], 0) + 1 # vector controlling color (add 1 so that the layers start from 1)

    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize colors
    norm = Normalize(vmin=1, vmax=len(model_layers))

    # Create LineCollection
    lc = LineCollection(segments, cmap='turbo_r', norm=norm)
    lc.set_array(c)
    lc.set_linewidth(2)

    # Plot the behavioral modeling RSA correlation scores, colored by DNN
    # layer assignment
    ax.add_collection(lc)

# x-axis parameters
ax.set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
ax.set_xlim(left=min(times), right=max(times))

# y-axis parameters
ax.set_ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4]
ylabels = [0, 0.1, 0.2, 0.3, 0.4]
plt.yticks(ticks=yticks, labels=ylabels)
ax.set_ylim(bottom=-.03, top=.25)

# Colorbar
if args.dnn_model == 'alexnet':
    model = 'AlexNet'
elif args.dnn_model == 'resnet50':
    model = 'ResNet-50'
plt.colorbar(lc, ax=ax, label=f'{model} layers')

# Save the figure
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'plots')
file_name = os.path.join(save_dir, f'layer_assignment_roi-all.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()