"""Compute the stats on the results of the RSA analysis between in silico EEG
responses and DNN layerwise features. The stats consist of bootstrapped 95%
confidence intervals and correlation between RSA peak latency and DNN layer.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
dnn_model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.utils import resample


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--dnn_model', default='alexnet', type=str)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the decoding results
# =============================================================================
decoding = []

for sub in args.subjects:

    # Load the results
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'dnn_layerwise_modeling', 'eeg_rdms', args.encoding_model,
        'eeg_rdms_sub-'+format(sub,'02')+'_channels-'+'-'.join(args.channels)+
        '.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    # Average the decoding results across pairwise comparisons
    idx_tril = np.tril_indices(len(results['eeg_rdm']), -1)
    decoding.append(np.mean(results['eeg_rdm'][idx_tril], 0))

    # EEG metadata
    times = results['metadata']['eeg']['times']

# Convert to numpy arrays
decoding = np.asarray(decoding) * 100


# =============================================================================
# Load the RSA results
# =============================================================================
rsa = {}

for s, sub in enumerate(args.subjects):

    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'dnn_layerwise_modeling', 'rsa', args.encoding_model, 'rsa_sub-'+
        format(sub,'02')+'_channels-'+'-'.join(args.channels)+'_dnn_model-'+
        args.dnn_model+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    # Loop across DNN layers
    for key, val in results['rsa'].items():

        # Get the RSA results
        if s == 0:
            rsa[key] = []
        rsa[key].append(val)

# Convert to numpy arrays
for key, val in rsa.items():
    rsa[key] = np.array(val)


# =============================================================================
# Correlate the RSA layerwise peak latency with the layer number
# =============================================================================
# Get the DNN layers
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

# Compute the peak latency for each layer
rsa_peak_latency = {}
for key in model_layers:
    rsa_peak_latency[key] = times[np.argmax(rsa[key], 1)]
peak_latency_vals = np.array([rsa_peak_latency[key] for key in model_layers])

# Correlate the RSA layerwise peak latency with the layer number
layer_nums = np.arange(1, len(rsa_peak_latency)+1)
rsa_peak_latency_dnn_layer_corr = []
for s in range(len(args.subjects)):
    rsa_peak_latency_dnn_layer_corr.append(spearmanr(layer_nums,
        peak_latency_vals[:,s])[0])
rsa_peak_latency_dnn_layer_corr = np.array(rsa_peak_latency_dnn_layer_corr)


# =============================================================================
# Bootstrap confidence intervals (CIs)
# =============================================================================
# Pairwise decoding CIs
ci_decoding = np.zeros((2, len(times)))
decoding_dist = np.zeros((args.n_iter, len(times)))
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    decoding_dist[i] = np.mean(decoding[idx], 0)
ci_decoding[0] = np.percentile(decoding_dist, 2.5, axis=0)
ci_decoding[1] = np.percentile(decoding_dist, 97.5, axis=0)

# RSA  CIs
ci_rsa = {}
ci_rsa_peak_latency = {}
for key in model_layers:
    ci_rsa[key] = np.zeros((2, len(times)))
    ci_rsa_peak_latency[key] = np.zeros((2))
    rsa_dist = np.zeros((args.n_iter, len(times)))
    peak_lat_dist = np.zeros((args.n_iter))
    for i in tqdm(range(args.n_iter)):
        idx = resample(np.arange(len(args.subjects)))
        rsa_dist[i] = np.mean(rsa[key][idx], 0)
        peak_lat_dist[i] = np.mean(rsa_peak_latency[key][idx])
    ci_rsa[key][0] = np.percentile(rsa_dist, 2.5, axis=0)
    ci_rsa[key][1] = np.percentile(rsa_dist, 97.5, axis=0)
    ci_rsa_peak_latency[key][0] = np.percentile(peak_lat_dist, 2.5, axis=0)
    ci_rsa_peak_latency[key][1] = np.percentile(peak_lat_dist, 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'decoding': decoding,
    'rsa': rsa,
    'rsa_peak_latency': rsa_peak_latency,
    'rsa_peak_latency_dnn_layer_corr': rsa_peak_latency_dnn_layer_corr,
    'ci_decoding': ci_decoding,
    'ci_rsa': ci_rsa,
    'ci_rsa_peak_latency': ci_rsa_peak_latency,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_' + 'channels-' + '-'.join(args.channels) + \
    '_dnn_model-' + args.dnn_model + '.npy'

np.save(os.path.join(save_dir, file_name), results)