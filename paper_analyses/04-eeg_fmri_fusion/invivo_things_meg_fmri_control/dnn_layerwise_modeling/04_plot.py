"""Plot the DNN layerwise assignment of each t-fMRI vertex and time point.

Parameters
----------
fmri_subjects : list
    List of THINGS fMRI1 subject identifiers. Valid subject identifiers are
    integers from 1 to 3.
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
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3], type=int)
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'dnn_layerwise_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA layerwise assignment results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'dnn_layerwise_modeling', 'stats',
    f'stats_dnn_model-{args.dnn_model}.npy')

results = np.load(data_dir, allow_pickle=True).item()

best_layer = results['best_layer']
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
# Plot all ROIs on the same graph
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
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'

# Define the ROIs to plot
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'EBA', 'PPA']

# Average the results across subjects
results = []
for roi in rois:
    results.append(np.mean(best_layer[roi], 0) + 1) # add 1 so that the layers start from 1
results = np.array(results)

# Create the figure
fig = plt.figure(figsize=(10, 7.5))

# Plot the results
ax = plt.imshow(results, aspect='auto', cmap='turbo_r', vmin=1,
    vmax=len(model_layers))

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, 20, 40, 60, 80, 100, 120, 139]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)

# y-axis parameters
yticks = np.arange(0, len(rois))
ylabels = rois
plt.yticks(ticks=yticks, labels=ylabels)

# Colorbar
if args.dnn_model == 'alexnet':
    model = 'AlexNet'
elif args.dnn_model == 'resnet50':
    model = 'ResNet-50'
ticks = np.arange(1, len(model_layers)+1)
plt.colorbar(ax, shrink=0.75, ticks=ticks,
    label=f'{model} layers', location='right')

# Save the figure
file_name = os.path.join(save_dir, f'layer_assignment_rois_dnn_model-{args.dnn_model}.svg')
fig.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True,
    format='svg')
plt.close()