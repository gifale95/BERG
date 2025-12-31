"""Plot the searchlight RSA scores between t-fMRI responses and DNN layerwise
features.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
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
from berg import BERG
import cortex
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=20, type=float) # 20
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the stats
# =============================================================================
stats_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stats', f'stats_model-{args.model}.npy')

stats = np.load(stats_dir, allow_pickle=True).item()


# =============================================================================
# Load the RSA results
# =============================================================================
# lh_rsa = {}
# rh_rsa = {}
lh_best_layer = []
rh_best_layer = []

for s, sub in enumerate(args.fmri_subjects):
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'dnn_layerwise_modeling', 'rsa',
            f'rsa_sub-{sub:02d}_{hemi}_model-{args.model}.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        # NCSNR and noise ceiling vertex selection
        ncsnr = results['metadata']['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr > args.ncsnr_threshold
        encoding = results['metadata']['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding > args.encoding_threshold
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

        # # Store the RSA results
        # rsa = results['rsa']
        # for key, val in rsa.items():
        #     if s == 0:
        #         if hemi == 'lh':
        #             lh_rsa[key] = []
        #         elif hemi == 'rh':
        #             rh_rsa[key] = []
        #     val[idx_nan] = np.nan
        #     if hemi == 'lh':
        #         lh_rsa[key].append(val)
        #     elif hemi == 'rh':
        #         rh_rsa[key].append(val)
        # del rsa

# Format the results to numpy arrays
# for key in lh_rsa.keys():
#     lh_rsa[key] = np.array(lh_rsa[key])
#     rh_rsa[key] = np.array(rh_rsa[key])
lh_best_layer = np.array(lh_best_layer)
rh_best_layer = np.array(rh_best_layer)


# =============================================================================
# Only use vertices falling within the NSD visual streams
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=1
    )

lh_idx_v = np.zeros(lh_best_layer.shape[1], dtype=int)
rh_idx_v = np.zeros(rh_best_layer.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    lh_idx_v[metadata['fmri']['lh_fsaverage_rois'][stream]] = 1
    rh_idx_v[metadata['fmri']['rh_fsaverage_rois'][stream]] = 1
lh_idx_v = np.where(lh_idx_v != 1)[0]
rh_idx_v = np.where(rh_idx_v != 1)[0]

lh_best_layer[:,lh_idx_v] = np.nan
rh_best_layer[:,rh_idx_v] = np.nan


# =============================================================================
# Load the EEG time points
# =============================================================================
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)

times = metadata_eeg['eeg']['times']


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
# Get the model layers
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
        linewidth=2,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=15,
        curvature_brightness=0.5,
        with_colorbar=True
        )

    # Add title
    title = f'Time (s): {np.round(time, 3)}'
    plt.title(title, fontsize=fontsize)

    # Save the figure
    plot_file = os.path.join(save_dir,
        f'rsa_layer_assigment_model-{args.model}_time-{t:03d}.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()