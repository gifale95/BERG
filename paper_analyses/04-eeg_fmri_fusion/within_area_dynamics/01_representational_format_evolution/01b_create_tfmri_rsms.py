"""Create RSMs using the t-fMRI responses for the 200 THINGS EEG2 test images.

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

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--eeg_reps', default='all', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Create t-fMRI RSMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the t-fMRI responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'tfmri_responses')

file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'eeg_reps-{args.eeg_reps}_images-{args.images}.npy')

tfmri = np.load(os.path.join(data_dir, file_name))


# =============================================================================
# Create the t-fMRI RSMs
# =============================================================================
Z = np.ascontiguousarray(tfmri.transpose(2, 0, 1), dtype=np.float32)  # (Times, Images, Vertices)
Z -= Z.mean(-1, keepdims=True)
Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
tfmri_rsms = (Z @ Z.transpose(0, 2, 1)).transpose(1, 2, 0)


# =============================================================================
# Save the RSMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'tfmri_rsms')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'tfmri_rsms_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'eeg_reps-{args.eeg_reps}_images-{args.images}.npy')

np.save(os.path.join(save_dir, file_name), tfmri_rsms)