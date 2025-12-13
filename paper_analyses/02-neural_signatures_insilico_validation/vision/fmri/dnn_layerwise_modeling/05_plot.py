"""Plot the searchlight RSA scores between in silico fMRI responses and DNN
layerwise features.

Parameters
----------
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) to
    consider a vertex for the tripartite organization analysis.
encoding_threshold : float
    The threshold on the encoding models explained variance to consider a
    vertex for the tripartite organization analysis (in % units).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=20, type=float) # 20
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the stats
# =============================================================================
stats_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'dnn_layerwise_modeling', 'stats', 'stats_model-'+args.model+'.npy')

stats = np.load(stats_dir, allow_pickle=True).item()


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = {}
rh_rsa = {}
lh_best_layer = []
rh_best_layer = []

for s, sub in enumerate(args.subjects):
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation', 'vision', 'fmri',
            'dnn_layerwise_modeling', 'rsa', 'rsa_sub-'+format(sub, '02')+
            '_'+hemi+'_model-'+args.model+'.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        # NCSNR and noise ceiling vertex selection
        ncsnr = results['metadata']['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr > args.ncsnr_threshold
        encoding = results['metadata']['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding > args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_ncsnr)

        # Store the layer assignment results
        if hemi == 'lh':
            best_layer = stats['lh_best_layer'][s].astype(np.float32)
            best_layer[idx_nan] = np.nan
            lh_best_layer.append(best_layer)
        elif hemi == 'rh':
            best_layer = stats['rh_best_layer'][s].astype(np.float32)
            best_layer[idx_nan] = np.nan
            rh_best_layer.append(best_layer)

        # Store the RSA results
        rsa = results['rsa']
        for key, val in rsa.items():
            if s == 0:
                if hemi == 'lh':
                    lh_rsa[key] = []
                elif hemi == 'rh':
                    rh_rsa[key] = []
            val[idx_nan] = np.nan
            if hemi == 'lh':
                lh_rsa[key].append(val)
            elif hemi == 'rh':
                rh_rsa[key].append(val)
        del rsa

# Format the results to numpy arrays
for key in lh_rsa.keys():
    lh_rsa[key] = np.array(lh_rsa[key])
    rh_rsa[key] = np.array(rh_rsa[key])
lh_best_layer = np.array(lh_best_layer)
rh_best_layer = np.array(rh_best_layer)


# =============================================================================
# Threshold the vertices by significance
# =============================================================================
for key in lh_rsa.keys():
    lh_rsa[key][:,~stats['sig_lh_rsa'][key]] = np.nan
    rh_rsa[key][:,~stats['sig_rh_rsa'][key]] = np.nan


# =============================================================================
# Plot parameters
# =============================================================================
# Plot parameters for colorbar
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage'


# =============================================================================
# Plot the RSA results
# =============================================================================
# Loop across model layers
for key in lh_rsa.keys():

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(lh_rsa[key], 0), np.nanmean(rh_rsa[key], 0))

    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='hot',
        vmin=0,
        vmax=0.4,
        with_colorbar=True
        )

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=2,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=15,
        curvature_brightness=0.5,
        with_colorbar=True
        )

    # Save the figure
    file_name = os.path.join(save_dir, 'rsa_model-'+args.model+'_layer-'+key+
        '.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


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
    cmap='gist_rainbow',
    vmin=1,
    vmax=len(lh_rsa.keys()),
    with_colorbar=True
    )

# Plot the flat brain surface
fig = cortex.quickshow(
    vertex_data,
    height=2000, # Increase resolution of map and ROI contours
    with_curvature=True,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    linewidth=2,
    linecolor=(1, 1, 1),
    with_labels=True,
    labelsize=15,
    curvature_brightness=0.5,
    with_colorbar=True
    )

# Save the figure
file_name = os.path.join(save_dir, 'rsa_layer_assigment_model-'+args.model+
    '.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')



# =============================================================================
# - Plot scatter-barplot of best layer of each ROI (averaged across ROI vertices) # !!!
#    - One dot for each subject, and then plot the mean with CIs.
#    - Likely don't use all ROIs, but only a selection of them.
# =============================================================================