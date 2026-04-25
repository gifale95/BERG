"""Aggregate the t-fMRI univariate responses for the ILSVRC-2012 training
images into h5py files.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
rois : list
    List containing the ROIs used for  which the t-fMRI responses are
    predicted.
tot_img_batches : int
    The total number of batches in which the images are divided.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--rois', default=['V1', 'hV4', 'ventral'], type=list)
parser.add_argument('--tot_img_batches', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Aggregate t-fMRI responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and aggregate the t-fMRI univariate responses across image batches
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'granger_causality',
    'rnc', 'tfmri_responses')

tfmri = {}

for b in tqdm(range(args.tot_img_batches)):

    file_name = f'tfmri_sub-{args.fmri_subject:02d}_batch-{b:02d}.npy'

    tfmri_batch = np.load(os.path.join(data_dir, file_name),
        allow_pickle=True).item()

    # Loop across ROIs
    for roi in args.rois:
        if b == 0:
            tfmri[roi] = tfmri_batch[roi]
        else:
            tfmri[roi] = np.append(tfmri[roi], tfmri_batch[roi], 0)


# =============================================================================
# Save the t-fMRI univariate responses
# =============================================================================
for roi in args.rois:

    file_name = f'tfmri_sub-{args.fmri_subject:02d}_roi-{roi}.npy'

    with h5py.File(os.path.join(data_dir, file_name), 'w') as f:
	    f.create_dataset('tfmri', data=tfmri[roi], dtype=np.float32)