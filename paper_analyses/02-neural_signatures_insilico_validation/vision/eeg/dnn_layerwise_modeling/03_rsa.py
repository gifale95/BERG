"""Perform RSA between in silico EEG responses and DNN layerwise features.

Parameters
----------
subject : int
    The subject identifier for the EEG encoding models. Since the used
    encoidng models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
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
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG RDMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'eeg_rdms', 'eeg_rdms_sub-'+
    format(args.subject, '02')+'_channels-'+'-'.join(args.channels)+'.npy')

data = np.load(data_dir, allow_pickle=True).item()
eeg_rdm = data['eeg_rdm']
metadata = data['metadata']


# =============================================================================
# Load the DNN layerwise RDMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'dnn_rdms',
    'dnn_rdms_'+args.model+'.npy')

dnn_rdms = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Perform RSA
# =============================================================================
# Empty RSA results arrays
rsa = {}
for key in dnn_rdms.keys():
    rsa[key] = np.zeros(eeg_rdm.shape[2], dtype=np.float32)

# Take the lower triangle of the DNN amd EEG RDMs
idx_tril = np.tril_indices(len(eeg_rdm), -1)
dnn_rdm_tril = {}
for key, val in dnn_rdms.items():
    dnn_rdm_tril[key] = val[idx_tril]
eeg_rdm_tril = eeg_rdm[idx_tril]

# Loop across EEG time points
for t in tqdm(range(eeg_rdm.shape[2])):

    # Perform RSA with each DNN layer
    for key, val in dnn_rdm_tril.items():
        rsa[key][t] = pearsonr(val, eeg_rdm_tril[:,t])[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'eeg_rdm': eeg_rdm,
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'dnn_layerwise_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_channels-' + \
    '-'.join(args.channels) + '_model-' + args.model + '.npy'

np.save(os.path.join(save_dir, file_name), results)