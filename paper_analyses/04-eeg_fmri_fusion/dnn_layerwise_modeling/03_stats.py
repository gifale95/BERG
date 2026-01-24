"""Compute the DNN layerwise assignment of each t-fMRI vertex and time point
based on the results of the RSA analysis between t-fMRI responses and DNN
layerwise features.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
dnn_model : str
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
from tqdm import tqdm
from berg import BERG


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=20, type=float) # 20
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Get the EEG time points
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = {}
rh_rsa = {}
metadata_fmri = []

for s, sub in enumerate(tqdm(args.fmri_subjects)):
    for h, hemi in enumerate(args.hemispheres):

        results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'dnn_layerwise_modeling', 'rsa',
            f'rsa_fmri_sub-{sub:02d}_{hemi}_dnn_model-{args.dnn_model}.npy')
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

        if h == 0:
            metadata_fmri.append(results['metadata_fmri'])

        del results

for key in lh_rsa.keys():
    lh_rsa[key] = np.array(lh_rsa[key])
    rh_rsa[key] = np.array(rh_rsa[key])


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

lh_best_layer = []
rh_best_layer = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.fmri_subjects)):

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
lh_best_layer = np.array(lh_best_layer).astype(np.float32)
rh_best_layer = np.array(rh_best_layer).astype(np.float32)


# =============================================================================
# Vertex selection
# =============================================================================
for s, sub in enumerate(tqdm(args.fmri_subjects)):

    # Only use vertices falling within the NSD visual streams
    lh_idx_v = np.zeros(lh_best_layer.shape[1], dtype=int)
    rh_idx_v = np.zeros(rh_best_layer.shape[1], dtype=int)
    streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
        'lateral', 'parietal']
    for stream in streams:
        lh_idx_v[metadata_fmri[s]['fmri']['lh_fsaverage_rois'][stream]] = 1
        rh_idx_v[metadata_fmri[s]['fmri']['rh_fsaverage_rois'][stream]] = 1
    lh_idx_v = np.where(lh_idx_v != 1)[0]
    rh_idx_v = np.where(rh_idx_v != 1)[0]
    lh_best_layer[s,lh_idx_v] = np.nan
    rh_best_layer[s,rh_idx_v] = np.nan

    # NCSNR and encoding accuracy vertex selection
    lh_ncsnr = metadata_fmri[s]['fmri']['lh_ncsnr']
    rh_ncsnr = metadata_fmri[s]['fmri']['rh_ncsnr']
    lh_idx_ncsnr = lh_ncsnr >= args.ncsnr_threshold
    rh_idx_ncsnr = rh_ncsnr >= args.ncsnr_threshold
    lh_encoding = metadata_fmri[s]['encoding_models']\
        ['lh_explained_variance_nsdcore']
    lh_idx_encoding = lh_encoding >= args.encoding_threshold
    lh_idx_nan = ~np.logical_and(lh_idx_ncsnr, lh_idx_encoding)
    rh_encoding = metadata_fmri[s]['encoding_models']\
        ['rh_explained_variance_nsdcore']
    rh_idx_encoding = rh_encoding >= args.encoding_threshold
    rh_idx_nan = ~np.logical_and(rh_idx_ncsnr, rh_idx_encoding)
    lh_best_layer[s,lh_idx_nan] = np.nan
    rh_best_layer[s,rh_idx_nan] = np.nan


# =============================================================================
# Get the ROI-wise DNN layer assignment
# =============================================================================
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'RSC', 'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral',
    'parietal']

# Empty result dictionary
best_layer_roi = {}

# Loop across ROIs
for r, roi in enumerate(rois):

    # Empty ROI layer assignment array of shape:
    # (N fMRI subjects, 140 EEG time points)
    best_layer_roi[roi] = np.zeros((len(args.fmri_subjects), len(times)),
        dtype=np.float32)

    # Loop across subjects and hemispheres
    for fs, fsub in enumerate(args.fmri_subjects):
        best_layer = []
        for h, hemi in enumerate(args.hemispheres):

            # Get the indices of the ROI vertices
            if roi in ['V1', 'V2', 'V3']:
                idx_roi = np.append(
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
                idx_roi.sort()
            elif roi in ['FFA', 'VWFA', 'FBA']:
                idx_roi = np.append(
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
                idx_roi.sort()
            elif roi in ['intermediate']:
                idx_roi = np.append(
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
                idx_roi = np.append(idx_roi,
                    metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
                idx_roi.sort()
            else:
                idx_roi = metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois'][roi]

            # Append the layer assignment vertex scores across hemispheres
            if hemi == 'lh':
                best_layer.append(lh_best_layer[fs,idx_roi])
            elif hemi == 'rh':
                best_layer.append(rh_best_layer[fs,idx_roi])

        # Store the mean layer assignment scores across ROI vertices
        best_layer = np.concatenate(best_layer, 0)
        best_layer_roi[roi][fs] = np.nanmean(best_layer, 0)
        del best_layer


# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_best_layer': lh_best_layer,
    'rh_best_layer': rh_best_layer,
    'best_layer_roi': best_layer_roi,
    'metadata_fmri': metadata_fmri,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'dnn_layerwise_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_dnn_model-{args.dnn_model}.npy'

np.save(os.path.join(save_dir, file_name), results)