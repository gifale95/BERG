"""Prepare the in vivo EEG responses used to train and test the fusion encoding
models.

The preparation includes appending the EEG channels responses across subjects,
and transfomring them with PCA independently for each EEG time point.

Parameters
----------
eeg_subject : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import numpy as np
import h5py
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subjects_all', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Prepare in vivo EEG <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and append the EEG responses across subjects
# =============================================================================
# Loop across subjects
for s, sub in enumerate(tqdm(args.eeg_subjects_all)):

    # Load the EEG responses
    eeg_dir_train = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{sub:02d}_split-train.h5')
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{sub:02d}_split-test.h5')
    eeg_train_sub = h5py.File(eeg_dir_train, 'r')['eeg'][:].astype(np.float32)
    eeg_test_sub = h5py.File(eeg_dir_test, 'r')['eeg'][:].astype(np.float32)

    # Append the EEG channel responses across subjects
    if s == 0:
        eeg_train = eeg_train_sub
        eeg_test = eeg_test_sub
    else:
        eeg_train = np.append(eeg_train, eeg_train_sub, 2)
        eeg_test = np.append(eeg_test, eeg_test_sub, 2)
    del eeg_train_sub, eeg_test_sub


# =============================================================================
# Save the in vivo EEG responses
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge',
    'invivo_eeg_responses')
os.makedirs(save_dir, exist_ok=True)

file_name_train = 'things_eeg_2_train.h5'
file_name_test = 'things_eeg_2_test.h5'

with h5py.File(os.path.join(save_dir, file_name_train), 'w') as f:
    f.create_dataset('eeg', data=eeg_train, dtype=np.float32)
with h5py.File(os.path.join(save_dir, file_name_test), 'w') as f:
    f.create_dataset('eeg', data=eeg_test, dtype=np.float32)