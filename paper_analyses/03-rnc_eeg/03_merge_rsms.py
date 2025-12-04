"""Combine the in silico EEG RSMs across partitions, and merge them across all
subjects.

Parameters
----------
all_subjects : list
    List with the subject identifiers of the 10 THINGS EEG2 subjects.
time : float
    The EEG time point (in seconds) for which to create the RSM.
total_rsm_splits : int
    Number of total RSM splits.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--time', type=float, default=0.1)
parser.add_argument('--total_rsm_splits', type=int, default=5)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Merge RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Combine the in silico EEG RSMs across partitions, and merge them across all
# subjects
# =============================================================================
img_cond = 10000

# Create the empty merged RSMs
rsms_all_subj = np.zeros((int(np.ceil(len(args.all_subjects)/2)), img_cond,
    img_cond), dtype=np.float32)

for s, sub in enumerate(tqdm(args.all_subjects)):

    # Combine the in silico fMRI RSMs across splits
    data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
        'rsms')
    for split in range(args.total_rsm_splits):
        rsm_file = 'rsm_sub-' + format(sub, '02') + '_time-' + \
            str(args.time) + '_split-' + format(split+1, '02') + '.npy'
        if split == 0:
            rsm = np.load(os.path.join(data_dir, rsm_file))
        else:
            rsm = np.append(rsm, np.load(os.path.join(data_dir, rsm_file)), 0)
        # Force garbage collection to free memory
        gc.collect()

    # Merge the RSMs across subjects. Fill the RSM's lower-triangular matrices
    # with even subjects, and the upper-triangular matrices with odd subjects.
    idx = int(np.floor(s / 2))
    if s % 2 == 0:
        rsms_all_subj[idx] += rsm
    else:
        rsms_all_subj[idx] += np.transpose(rsm)
    del rsm
    # Force garbage collection to free memory
    gc.collect()


# =============================================================================
# Save the merged RSMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'rsms')
os.makedirs(save_dir, exist_ok=True)

save_file = os.path.join(save_dir, 'merged_rsms_time-'+str(args.time))

with h5py.File(save_file+'.h5py', 'w') as hf:
    for k,v in {'rsms': rsms_all_subj}.items():
        hf.create_dataset(k,data=v)