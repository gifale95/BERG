"""Assign vertices to the DNN layer leading to highest RSA scores, and
correlate the layer assignment of each vertex with the vertex' position along
the visual hierarchy (early, intermediate, ventral/lateral/dorsal visual
streams).

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=0, type=float)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = {}
rh_rsa = {}
metadata = []
corr_best_layer_hierarchy_score = []

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

for s, sub in enumerate(tqdm(args.subjects)):
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation', 'vision', 'fmri',
            'dnn_layerwise_modeling', 'rsa', args.encoding_model, 'rsa_sub-'+
            format(sub, '02')+'_'+hemi+'_model-'+args.model+'.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        # Load the metadata
        if hemi == 'lh':
            # Get the test image number
            metadata.append(berg.get_model_metadata(
                args.encoding_model,
                subject=sub
            ))

        for key, val in results['rsa'].items():
            if hemi == 'lh':
                if s == 0:
                    lh_rsa[key] = []
                lh_rsa[key].append(val)
            elif hemi == 'rh':
                if s == 0:
                    rh_rsa[key] = []
                rh_rsa[key].append(val)

for key in lh_rsa.keys():
    lh_rsa[key] = np.array(lh_rsa[key])
    rh_rsa[key] = np.array(rh_rsa[key])


# =============================================================================
# Assign vertices to the DNN layer leading to highest RSA scores
# =============================================================================
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

lh_best_layer = []
rh_best_layer = []

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # Append the results across all layers
    lh_rsa_all_layers = []
    rh_rsa_all_layers = []
    for layer in model_layers:
        lh_rsa_all_layers.append(lh_rsa[layer][s])
        rh_rsa_all_layers.append(rh_rsa[layer][s])
    lh_rsa_all_layers = np.array(lh_rsa_all_layers)
    rh_rsa_all_layers = np.array(rh_rsa_all_layers)

    # Get the layer number leading to highest RSA scores
    lh_best_layer.append(np.argsort(lh_rsa_all_layers, axis=0)[-1])
    rh_best_layer.append(np.argsort(rh_rsa_all_layers, axis=0)[-1])

# Format to numpy arrays
lh_best_layer = np.array(lh_best_layer)
rh_best_layer = np.array(rh_best_layer)


# =============================================================================
# Correlate layer assignments with ROI positions
# =============================================================================
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Assign a hierarchical score to each vertex based on its ROI position
    # along the visual hierarchy
    lh_hierarchy_score = np.zeros(lh_best_layer.shape[1], dtype=int)
    rh_hierarchy_score = np.zeros(rh_best_layer.shape[1], dtype=int)
    for i, stream in enumerate(streams):
        if stream in ['midventral', 'midlateral', 'midparietal']:
            hierarchy_level = 2
        elif stream == 'early':
            hierarchy_level = 1
        else:  # ventral, lateral, parietal
            hierarchy_level = 3
        lh_hierarchy_score[metadata[s]['fmri']['lh_fsaverage_rois'][stream]] = \
            hierarchy_level
        rh_hierarchy_score[metadata[s]['fmri']['rh_fsaverage_rois'][stream]] = \
            hierarchy_level

    # Only select vertices from the early, intermediate, and
    # ventral/lateral/dorsal visual streams
    lh_stream_idx = np.zeros(lh_best_layer.shape[1], dtype=int)
    rh_stream_idx = np.zeros(rh_best_layer.shape[1], dtype=int)
    for stream in streams:
        lh_stream_idx[metadata[s]['fmri']['lh_fsaverage_rois'][stream]] = 1
        rh_stream_idx[metadata[s]['fmri']['rh_fsaverage_rois'][stream]] = 1
    lh_stream_idx = lh_stream_idx == 1
    rh_stream_idx = rh_stream_idx == 1

    # Only retain vertices that have above threshold (i) NCSNR AND
    # (ii) encoding prediction accuracy
    lh_idx_ncsnr = metadata[s]['fmri']['lh_ncsnr'] >= \
        args.ncsnr_threshold
    rh_idx_ncsnr = metadata[s]['fmri']['rh_ncsnr'] >= \
        args.ncsnr_threshold
    lh_idx_encoding = \
        metadata[s]['encoding_models']['lh_explained_variance_nsdcore'] >= \
        args.encoding_threshold
    rh_idx_encoding = \
        metadata[s]['encoding_models']['rh_explained_variance_nsdcore'] >= \
        args.encoding_threshold

    # Vertex selection
    lh_idx = np.logical_and.reduce((lh_stream_idx, lh_idx_ncsnr,
        lh_idx_encoding))
    rh_idx = np.logical_and.reduce((rh_stream_idx, rh_idx_ncsnr,
        rh_idx_encoding))
    best_layer = np.append(lh_best_layer[s,lh_idx], rh_best_layer[s,rh_idx])
    hierarchy_score = np.append(lh_hierarchy_score[lh_idx],
        rh_hierarchy_score[rh_idx])

    # Correlate the layer assignment of each vertex with the vertex' position
    # along the visual hierarchy (early, intermediate, and
    # ventral/lateral/dorsal visual streams)
    corr_best_layer_hierarchy_score.append(spearmanr(best_layer,
        hierarchy_score)[0])


# =============================================================================
# Compute the significance
# =============================================================================
p_val_corr_best_layer_hierarchy_score = ttest_1samp(
    corr_best_layer_hierarchy_score, 0, alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'lh_best_layer': lh_best_layer,
    'rh_best_layer': rh_best_layer,
    'corr_best_layer_hierarchy_score': corr_best_layer_hierarchy_score,
    'p_val_corr_best_layer_hierarchy_score': p_val_corr_best_layer_hierarchy_score
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_model-' + args.model + '.npy'

np.save(os.path.join(save_dir, file_name), results)