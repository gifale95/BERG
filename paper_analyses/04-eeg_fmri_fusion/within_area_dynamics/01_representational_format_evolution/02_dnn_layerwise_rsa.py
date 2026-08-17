"""Perform RSA between t-fMRI time point RSMs, and the DNN layerwise
activation RSMs.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
dnn : str
    Name of the used DNN. Possible values are 'dinov2l'.
images : str
    If 'things_eeg_2_vivo', use the in vivo EEG responses for the 200 THINGS
    EEG2 test images.
    If 'things_eeg_2_silico', use the in silico EEG responses for the 200
    THINGS EEG2 test images.
    If 'nsd_515_shared', use the in silico EEG responses for the 515 NSD shared
    images.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> DNN layerwise RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the t-fMRI RSMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'tfmri_rsms')

file_name = (f'tfmri_rsms_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'eeg_reps-all_images-{args.images}.npy')

tfmri_rsms = np.load(os.path.join(data_dir, file_name))


# =============================================================================
# Load the DNN RSMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'dnn_rsms')

if args.images == 'things_eeg_2_vivo' or args.images == 'things_eeg_2_silico':
    file_name = f'dnn_rsms_dnn-{args.dnn}_images-things_eeg_2.npy'
elif args.images == 'nsd_515_shared':
    file_name = f'dnn_rsms_dnn-{args.dnn}_images-nsd_515_shared.npy'

dnn_rsms = np.load(os.path.join(data_dir, file_name))


# =============================================================================
# Perform the DNN layerwise RSA
# =============================================================================
n_times = tfmri_rsms.shape[2]
n_layers = dnn_rsms.shape[2]
n_img = tfmri_rsms.shape[0]
dnn_layerwise_rsa = np.zeros((n_layers, n_times))
idx_tril = np.tril_indices(n_img, k=-1)

for t in range(n_times):
    for l in range(n_layers):
        dnn_layerwise_rsa[l,t] = pearsonr(dnn_rsms[:,:,l][idx_tril],
            tfmri_rsms[:,:,t][idx_tril])[0]


# =============================================================================
# Save the RSA results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'dnn_layerwise_rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'dnn_layerwise_rsa_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'images-{args.images}.npy')

np.save(os.path.join(save_dir, file_name), dnn_layerwise_rsa)