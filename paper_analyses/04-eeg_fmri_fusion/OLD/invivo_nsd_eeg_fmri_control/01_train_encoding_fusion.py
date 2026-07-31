"""Train the EEG-fMRI encoding fusion models by linearly mapping in vivo EEG
responses (from tNSD) onto in vivo fMRI responses (from NSD) using the training
images.

One regression model is trained for each fMRI voxel and EEG time point,
using the EEG channel responses.

To reduce computational load, the EEG-fMRI fusion encoding models are only
trained, tested, and used for voxels falling within the NSD visual streams.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'all' (all trials), 'even' (even trials), and 'odd' (odd
    trials).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_train_trials', default='all', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the encoding fusion model weight save directory
# =============================================================================
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights')
os.makedirs(save_dir_weights, exist_ok=True)


# =============================================================================
# Load the EEG-fMRI train responses
# =============================================================================
# fMRI
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_data')
file_name = f'fmri_train_sub-{args.subject:02d}_hemi-{args.hemisphere}.npy'
fmri_train = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()['fmri_train'].astype(np.float32)

# EEG
file_name = (f'eeg_train_sub-{args.subject:02d}_'
    f'trial_avg-{args.eeg_train_trials}.npy')
eeg_train = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()['eeg_train'].astype(np.float32)


# =============================================================================
# Train the encoding fusion models
# =============================================================================
# Loop across EEG time points
for t in tqdm(range(eeg_train.shape[2])):

    # Train the encoding fusion models
    alphas = np.logspace(-6, 10, 17)
    reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    reg.fit(eeg_train[:,:,t], fmri_train)

    # Store the encoding fusion model weights
    reg_param = {
        'coef_': reg.coef_.astype(np.float32),
        'intercept_': reg.intercept_.astype(np.float32),
        'alpha_': reg.alpha_,
        'n_features_in_': reg.n_features_in_
    }

    # Save the encoding fusion model weights
    file_name = (f'weights_sub-{args.subject:02d}_'
                f'hemi-{args.hemisphere}_'
                f'eeg_train_trials-{args.eeg_train_trials}_'
                f'eeg_time-{t:03d}.npy')
    np.save(os.path.join(save_dir_weights, file_name), reg_param)
    del reg, reg_param