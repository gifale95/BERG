"""Perform RSA between the RDMs of each t-fMRI vertex and time point, and the
DNN layerwise activation RDMs.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
roi : str
    Used ROI.
eeg_reps : str
    String indicating whether to use EEG responses averaged across 'even',
    'odd', or 'all' repeats.
dnn : str
    Name of the used DNN. Possible values are 'dinov2l' and 'alexnet'.
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
parser.add_argument('--eeg_reps', default='all', type=str)
parser.add_argument('--dnn', default='dinov2l', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> DNN layerwise RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the DNN RSMs, and convert them to RDMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution', 'dnn_rsms')

if args.images == 'things_eeg_2_vivo' or args.images == 'things_eeg_2_silico':
    file_name = f'dnn_rsms_dnn-{args.dnn}_images-things_eeg_2.npy'
elif args.images == 'nsd_515_shared':
    file_name = f'dnn_rsms_dnn-{args.dnn}_images-nsd_515_shared.npy'

# Load and convert the RSMs to RDMs (1 - RSM)
dnn_rsms = 1 - np.load(os.path.join(data_dir, file_name))
n_images = dnn_rsms.shape[0]
n_layers = dnn_rsms.shape[2]
idx_tril = np.tril_indices(len(dnn_rsms), k=-1)


# =============================================================================
# Load the t-fMRI responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution',
    'tfmri_responses')

file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'eeg_reps-{args.eeg_reps}_images-{args.images}.npy')

tfmri = np.load(os.path.join(data_dir, file_name))
n_vertices = tfmri.shape[1]
n_times = tfmri.shape[2]


# =============================================================================
# Loop across t-fMRI time points
# =============================================================================
dnn_layerwise_rsa = np.zeros((n_layers, n_vertices, n_times))
for t in tqdm(range(n_times)):


# =============================================================================
# Create the t-fMRI RDMs for each vertex (using MSE)
# =============================================================================
    # Convert the t-fMRI responses to a contiguous array, and create the empty
    # RDMs array 
    data = np.ascontiguousarray(tfmri[:,:,t], dtype=np.float32)
    tfmri_rdms = np.zeros((n_images, n_images, n_vertices), dtype=np.float32)

    # Compute the RDMs for each vertex (using MSE)
    np.subtract(data[:,None,:], data[None,:,:], out=tfmri_rdms)
    np.square(tfmri_rdms, out=tfmri_rdms)


# =============================================================================
# Perform the DNN layerwise RSA
# =============================================================================
    for v in range(n_vertices):
        for l in range(n_layers):
            dnn_layerwise_rsa[l,v,t] = pearsonr(dnn_rsms[:,:,l][idx_tril],
                tfmri_rdms[:,:,v][idx_tril])[0]


# =============================================================================
# Save the RSA results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution',
    'dnn_layerwise_rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'dnn_layerwise_rsa_single_vertices_sub-{args.fmri_subject:02d}_'
    f'roi-{args.roi}_images-{args.images}_dnn-{args.dnn}.npy')

np.save(os.path.join(save_dir, file_name), dnn_layerwise_rsa)