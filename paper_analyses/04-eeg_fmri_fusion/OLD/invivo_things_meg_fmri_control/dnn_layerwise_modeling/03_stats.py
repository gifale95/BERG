"""Compute the DNN layerwise assignment of each t-fMRI vertex and time point
based on the results of the RSA analysis between t-fMRI responses and DNN
layerwise features.

Parameters
----------
fmri_subjects : list
    List of THINGS fMRI1 subject identifiers. Valid subject identifiers are
    integers from 1 to 3.
dnn_model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
noise_ceiling_threshold : float
    The threshold on the noise ceiling for voxel selection.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3], type=list)
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Get the MEG time points
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the MEG encoding model metadata
metadata_meg = berg.get_model_metadata(
    'meg-things_meg_1-vit_b_32',
    subject=1
)

# Load the MEG time points
tmax = 0.595
times = metadata_meg['meg']['times']
time_idx = np.zeros(len(times), dtype=int)
time_idx[times <= tmax] = 1
time_idx = np.where(time_idx == 1)[0]
times = times[times <= tmax]


# =============================================================================
# Load the RSA results
# =============================================================================
rsa = {}

# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'invivo_things_meg_fmri_control', 'dnn_layerwise_modeling', 'rsa',
        f'rsa_fmri_sub-{fsub:02d}_dnn_model-{args.dnn_model}.npy')
    results = np.load(results_dir, allow_pickle=True).item()

    # Loop across ROIs
    for r, roi in enumerate(tqdm(results.keys())):

        if fs == 0:
            rsa[roi] = {}

        for key, val in results[roi].items():
            if fs == 0:
                rsa[roi][key] = []
            rsa[roi][key].append(val)

    del results

# Format the results to numpy arrays
for roi in rsa.keys():
    for key in rsa[roi].keys():
        rsa[roi][key] = np.array(rsa[roi][key])


# =============================================================================
# Assign vertices to the DNN layer leading to highest RSA scores
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

best_layer = {}

# Loop across ROIs
for r, roi in enumerate(tqdm(rsa.keys())):

    best_layer[roi] = np.zeros((len(args.fmri_subjects), len(times)),
        dtype=np.float32)

    # Loop across subjects
    for s, sub in enumerate(tqdm(args.fmri_subjects)):

        # Append the results across all layers
        rsa_all_layers = []
        for layer in model_layers:
            rsa_all_layers.append(rsa[roi][layer][s])
        rsa_all_layers = np.array(rsa_all_layers)

        # Get the layer number leading to highest RSA scores
        best_layer[roi][s] = np.argsort(rsa_all_layers, axis=0)[-1]

# Format to numpy arrays
for roi in best_layer.keys():
    best_layer[roi] = np.array(best_layer[roi]).astype(np.float32)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'best_layer': best_layer,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'dnn_layerwise_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_dnn_model-{args.dnn_model}.npy'

np.save(os.path.join(save_dir, file_name), results)