"""Create RSMs using the t-fMRI responses for the RNC baseline and controlling
images.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
roi : str
    Used ROI.
time_window_pair: str
    A string specifying the two time windows of interest used to find the
    baseline and controlling images.
use_time_bins: int
    If '1', average the t-fMRI responses into four time bins (50-100ms,
    100-150ms, 150-200ms, 200-250ms). If '0', do not average the t-fMRI
    responses into time bins.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.05-0.10__0.20-0.25', type=str)
parser.add_argument('--use_time_bins', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Create t-fMRI RSMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the t-fMRI responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'tfmri_responses')

file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}.npy')

tfmri = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()


# =============================================================================
# Average the t-fMRI responses in time windows
# =============================================================================
if args.use_time_bins == 1:

    # Get the EEG time points
    berg = BERG(berg_dir=args.berg_dir)
    metadata_eeg = berg.get_model_metadata(
        'eeg-things_eeg_2-vit_b_32',
        subject=1
    )
    times = np.round(metadata_eeg['eeg']['times'], 3)

    # Get the time window start and end time points
    tw_start = [0.05, 0.1, 0.15, 0.2]
    tw_end = [0.1, 0.15, 0.2, 0.25]
    tw_start_idx = [np.where(times == s)[0][0] for s in tw_start]
    tw_end_idx = [np.where(times == e)[0][0] for e in tw_end]

    # Average the t-fMRI responses in time windows
    tfmri_tw_avg = {}
    for key, val in tfmri.items():
        tfmri_tw_avg[key] = np.zeros((val.shape[0], val.shape[1],
            len(tw_start)))
        for i in range(len(tw_start)):
            tfmri_tw_avg[key][:,:,i] = np.mean(
                val[:,:,tw_start_idx[i]:tw_end_idx[i]], 2)

    # Rename the data variable to the original name and delete the temporary
    # variable
    del tfmri
    tfmri = tfmri_tw_avg
    del tfmri_tw_avg


# =============================================================================
# Create the t-fMRI RSMs
# =============================================================================
tfmri_rsms = {}

for key, val in tqdm(tfmri.items()):

    Z = np.ascontiguousarray(val.transpose(2, 0, 1), dtype=np.float32)  # (Times, Images, Vertices)
    Z -= Z.mean(-1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
    tfmri_rsms[key] = (Z @ Z.transpose(0, 2, 1)).transpose(1, 2, 0)


# =============================================================================
# Save the RSMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'quantitative_dynamics_analysis', 'tfmri_rsms')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'tfmri_rsms_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'image_window_pair-{args.time_window_pair}_'
    f'use_time_bins-{args.use_time_bins}.npy')

np.save(os.path.join(save_dir, file_name), tfmri_rsms)