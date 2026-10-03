"""Average the in silico EEG RSMs across subjects.

If cross-validation is used, the RSMs are averaged across N-1 (train) subjects.
Multivariate RNC will later be applied on these averaged RSMs, and the
resulting controlling images validated on the RSMs of the left-out (test)
subject.

If cross-validation is not used, the RSMs are averaged across all subjects, and
multivariate RNC will later be applied on the these RSMs to select the
controlling images.

Parameters
----------
all_subjects : list
    List with the subject identifiers of the 10 THINGS EEG2 subjects.
cv : int
    If '1' multivariate RNC leaves the data of one subject out for
    cross-validation, if '0' multivariate RNC uses the data of all subjects.
cv_subject : int
    If cv==1, the left-out subject during cross-validation, out of the 10
    THINGS EEG2 subjects.
time : float
    The EEG time point (in seconds) for which to create the RSM.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--time', type=float, default=0.1)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Average RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the merged RSMs
# =============================================================================
dir_rsm = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'rsms')
rsms = h5py.File(os.path.join(dir_rsm, 'merged_rsms_time-'+str(args.time)+
    '.h5py'), 'r')['rsms']

idx_upper_tr = np.triu_indices(rsms.shape[1], 1)
idx_lower_tr = np.tril_indices(rsms.shape[1], -1)


# =============================================================================
# Average the RSMs across subjects (CV == 0)
# =============================================================================
if args.cv == 0:

    rsm_all = np.zeros((rsms.shape[1], rsms.shape[1]), dtype=np.float32)

    for s, sub in enumerate(args.all_subjects):
        idx_rsm = int(np.floor(s / 2))
        if s % 2 == 0:
            rsm_all[idx_lower_tr] += rsms[idx_rsm][idx_lower_tr]
            rsm_all[idx_upper_tr] += np.transpose(rsms[idx_rsm])[idx_upper_tr]
        else:
            rsm_all[idx_upper_tr] += rsms[idx_rsm][idx_upper_tr]
            rsm_all[idx_lower_tr] += np.transpose(rsms[idx_rsm])[idx_lower_tr]

    rsm_all /= len(args.all_subjects)


# =============================================================================
# Average the RSMs across subjects (CV == 1)
# =============================================================================
elif args.cv == 1:

    rsm_train = np.zeros((rsms.shape[1], rsms.shape[1]), dtype=np.float32)
    rsm_test = np.zeros((rsms.shape[1], rsms.shape[1]), dtype=np.float32)

    for s, sub in enumerate(args.all_subjects):
        idx_rsm = int(np.floor(s / 2))
        if s % 2 == 0:
            if sub == args.cv_subject:
                rsm_test[idx_lower_tr] += rsms[idx_rsm][idx_lower_tr]
                rsm_test[idx_upper_tr] += np.transpose(
                    rsms[idx_rsm])[idx_upper_tr]
            else:
                rsm_train[idx_lower_tr] += rsms[idx_rsm][idx_lower_tr]
                rsm_train[idx_upper_tr] += np.transpose(
                    rsms[idx_rsm])[idx_upper_tr]
        else:
            if sub == args.cv_subject:
                rsm_test[idx_upper_tr] += rsms[idx_rsm][idx_upper_tr]
                rsm_test[idx_lower_tr] += np.transpose(
                    rsms[idx_rsm])[idx_lower_tr]
            else:
                rsm_train[idx_upper_tr] += rsms[idx_rsm][idx_upper_tr]
                rsm_train[idx_lower_tr] += np.transpose(
                    rsms[idx_rsm])[idx_lower_tr]

    rsm_train /= len(args.all_subjects) - 1


# =============================================================================
# Save the averaged RSMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'rsms')
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = 'averaged_rsm_time-' + str(args.time) + '_all_subjects.npy'
    np.save(os.path.join(save_dir, file_name), rsm_all)

elif args.cv == 1:
    file_name_train = 'averaged_rsm_time-' + str(args.time) + \
        '_cv_subject-' + format(args.cv_subject, '02') + '_train.npy'
    file_name_test = 'averaged_rsm_time-' + str(args.time) + \
        '_cv_subject-' + format(args.cv_subject, '02') + '_test.npy'
    np.save(os.path.join(save_dir, file_name_train), rsm_train)
    np.save(os.path.join(save_dir, file_name_test), rsm_test)