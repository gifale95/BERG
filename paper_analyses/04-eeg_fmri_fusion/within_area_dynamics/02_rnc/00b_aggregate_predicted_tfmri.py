"""Aggregate the t-fMRI univariate responses into h5py files.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
roi : str
    Used ROI.
imageset : str
    The image set to use for the analysis. Possible values are: 'imagenet'
    (ILSVRC-2012 validation split) and 'coco' (MS COCO 2017 test split).
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
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--imageset', default='imagenet', type=str)
parser.add_argument('--tot_img_batches', default=10, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Aggregate t-fMRI responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and aggregate the t-fMRI univariate responses across image batches
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'tfmri_responses')

for b in tqdm(range(args.tot_img_batches)):

    file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
        f'imageset_{args.imageset}_batch-{b:02d}.npy')

    tfmri_batch = np.load(os.path.join(data_dir, file_name)).astype(np.float32)

    if b == 0:
        tfmri = tfmri_batch
    else:
        tfmri = np.append(tfmri, tfmri_batch, 0)
    del tfmri_batch


# =============================================================================
# Save the t-fMRI univariate responses
# =============================================================================
file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'imageset_{args.imageset}.h5')

with h5py.File(os.path.join(data_dir, file_name), 'w') as f:
    f.create_dataset('tfmri', data=tfmri, dtype=np.float32)