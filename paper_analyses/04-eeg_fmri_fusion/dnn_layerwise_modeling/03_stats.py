"""Compute the significance of the RSA analysis between t-fMRI responses and
DNN layerwise features.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = {}
rh_rsa = {}

for s, sub in enumerate(args.subjects):
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'dnn_layerwise_modeling', 'rsa', f'rsa_sub-{sub:02d}_{hemi}.npy')
        results = np.load(results_dir, allow_pickle=True).item()

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
# Plot/report the layer assignment averaged across all vertices within all ROIs
# (V1, V2, V3, hV4, ventral) (Guclu & van Gerven, 2015, Fig. 4B).
# Also compute CIs. # !!!
# =============================================================================





# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_best_layer': lh_best_layer,
    'rh_best_layer': rh_best_layer
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_model-{args.model}.npy'

np.save(os.path.join(save_dir, file_name), results)