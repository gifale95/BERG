"""Train the MEG-fMRI encoding fusion models by linearly mapping in vivo MEG
responses (from THINGS MEG1) onto in vivo fMRI responses (from THINGS fMRI1)
using the training images.

One regression model is trained for each fMRI voxel and MEG time point,
using the MEG sensor responses appended across the 4 THINGS MEG1 subjects.

To reduce computational load, the MEG-fMRI fusion encoding models are only
trained, tested, and used for voxels falling within the THINGS fMRI1 visual
ROIs.

The in vivo THINGS MEG1 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_meg_1

The in vivo THINGS fMRI1 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_fmri_1

Parameters
----------
fmri_subject : int
    THINGS fMRI1 subject identifiers. Valid subject identifiers are integers
    from 1 to 3.
meg_subjects : list
    List containing the subject identifiers for the THINGS MEG1 subjects. Valid
    subject identifiers are integers from 1 to 4.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in vivo THINGS fMRI1 train responses
# =============================================================================
# Load the fMRI responses
fmri_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-train.h5')
fmri_train = h5py.File(fmri_dir, 'r')['neural_data']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

# Get the image files names
train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']


# =============================================================================
# Load and append the in vivo THINGS MEG1 train responses across subjects
# =============================================================================
# Loop across subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Time point selection
    tmax = 0.595
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]

    # Load the MEG responses
    meg_train_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-train.h5')
    meg_train_sub = h5py.File(meg_train_dir, 'r')['neural_data']

    # Get the MEG responses for the images shared with the fMRI
    train_stimuli_meg = metadata_meg['encoding_model']['all_training_splits']\
        ['train_stimuli']
    idx_meg = []
    for stim in train_stimuli_fmri:
        idx_meg.append(train_stimuli_meg.index(stim))
    idx_meg = np.array(idx_meg)
    meg_train_sub = meg_train_sub[:,:,time_idx][idx_meg].astype(np.float32)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_train = meg_train_sub
    else:
        meg_train = np.append(meg_train, meg_train_sub, 1)
    del meg_train_sub


# =============================================================================
# Train the encoding fusion models
# =============================================================================
# Loop across ROIs
rois = ['V1', 'V2', 'V3', 'hV4', 'lFFA', 'rFFA', 'lOFA', 'rOFA', 'lEBA',
    'rEBA', 'lPPA', 'rPPA', 'lRSC', 'rRSC', 'lTOS', 'rTOS', 'lLOC', 'rLOC',
    'IT']
reg_param = {}
for r, roi in enumerate(tqdm(rois)):

    # Get the fMRI voxel responses for the current ROI, and sort them based on
    # the image IDs
    roi_idx = metadata_fmri['roi'][roi]
    fmri_train_roi = fmri_train[:,roi_idx]
    fmri_train_roi = fmri_train_roi

    # Empty dictionary to store the encoding fusion model weights for the
    # current ROI
    reg_param[roi] = {}
    reg_param[roi]['coef_'] = []
    reg_param[roi]['intercept_'] = []
    reg_param[roi]['alpha_'] = []
    reg_param[roi]['n_features_in_'] = []

    # Loop across MEG time points
    for t in range(len(times)):

        # Train the encoding fusion models
        alphas = np.logspace(-6, 10, 17)
        reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        reg.fit(meg_train[:,:,t], fmri_train_roi)

        # Store the encoding fusion model weights
        reg_param[roi]['coef_'].append(reg.coef_.astype(np.float32))
        reg_param[roi]['intercept_'].append(reg.intercept_.astype(np.float32))
        reg_param[roi]['alpha_'].append(reg.alpha_.astype(np.float32))
        reg_param[roi]['n_features_in_'].append(reg.n_features_in_)


# =============================================================================
# Save the encoding fusion model weights
# =============================================================================
# Create the encoding fusion model weight save directory
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'encoding_fusion_weights')
os.makedirs(save_dir_weights, exist_ok=True)

# Save the weights
file_name = f'weights_fmri_sub-{args.fmri_subject:02d}.npy'
np.save(os.path.join(save_dir_weights, file_name), reg_param)