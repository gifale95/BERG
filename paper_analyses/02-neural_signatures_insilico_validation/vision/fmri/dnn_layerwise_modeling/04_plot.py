"""Plot the searchlight RSA scores between in silico fMRI responses and DNN
layerwise features.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import cortex
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=0, type=float) # 20
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the stats
# =============================================================================
stats_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'dnn_layerwise_modeling', 'stats', args.encoding_model, 'stats_model-'+
    args.model+'.npy')

stats = np.load(stats_dir, allow_pickle=True).item()

lh_best_layer = []
rh_best_layer = []

# Loop across subjects and hemispheres
for s, sub in enumerate(args.subjects):
    for hemi in ['lh', 'rh']:

        # NCSNR and encoding accuracy vertex selection
        ncsnr = stats['metadata'][s]['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = stats['metadata'][s]['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

        # Store the layer assignment results
        if hemi == 'lh':
            best_layer = stats['lh_best_layer'][s].astype(np.float32)
            best_layer[idx_nan] = np.nan
            lh_best_layer.append(best_layer)
        elif hemi == 'rh':
            best_layer = stats['rh_best_layer'][s].astype(np.float32)
            best_layer[idx_nan] = np.nan
            rh_best_layer.append(best_layer)

# Format the results to numpy arrays
lh_best_layer = np.array(lh_best_layer)
rh_best_layer = np.array(rh_best_layer)


# =============================================================================
# Get the DNN layers
# =============================================================================
# AlexNet
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

# ResNet-50
elif args.model == 'resnet50':
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
# Plot parameters for colorbar
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'


# =============================================================================
# Plot the vertex DNN layer assignment
# =============================================================================
# Average the results across subjects, and append them across left and
# right hemishperes
data = np.append(np.nanmean(lh_best_layer, 0),
    np.nanmean(rh_best_layer, 0)) + 1 # Add 1 so that the layers start from 1

# Create the flat brain surface
vertex_data = cortex.Vertex(
    data,
    subject=subject,
    cmap='turbo_r',
    vmin=1,
    vmax=len(model_layers),
    with_colorbar=True
    )

# Plot the flat brain surface
fig = cortex.quickshow(
    vertex_data,
    height=2000, # Increase resolution of map and ROI contours
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

# Save the figure
file_name = os.path.join(save_dir, 'rsa_layer_assigment_model-'+args.model+
    '.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# - Plot scatter-barplot of best layer of each ROI (averaged across ROI vertices) # !!!
#    - One dot for each subject, and then plot the mean with CIs.
#    - Likely don't use all ROIs, but only a selection of them.
# =============================================================================