"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 100 THINGS MEG1/fMRI1 test images. These t-fMRI
responses are then correlated with the in vivo THINGS fMRI1 test responses,
resulting in one encoding accuracy score for each fMRI voxel and MEG time
point.

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
import gc
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in vivo THINGS fMRI1 test responses
# =============================================================================
# Load the fMRI responses
fmri_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-test.h5')
fmri_test_all = h5py.File(fmri_dir, 'r')['neural_data']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

# Get the image files names
test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
unique_test_stimuli = np.unique(test_stimuli_fmri)

# Average the fMRI responses across repetitions of the same test image
fmri_test = []
for stim in tqdm(unique_test_stimuli):
    idx = np.where(test_stimuli_fmri == stim)[0]
    fmri_test.append(fmri_test_all[idx].mean(0))
fmri_test = np.array(fmri_test)


# =============================================================================
# Load and append the in vivo THINGS MEG1 test responses across subjects
# =============================================================================
# Loop across MEG subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Load the MEG responses
    meg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_dir, 'r')['neural_data']

    # Time point selection
    tmax = 0.595
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]
    meg_test_all = meg_test_all[:,:,time_idx].astype(np.float32)

    # Average the MEG responses across repetitions for the images shared with
    # the fMRI
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_test = meg_test_sub
    else:
        meg_test = np.append(meg_test, meg_test_sub, 1)
    del meg_test_all, meg_test_sub


# =============================================================================
# Test the encoding fusion models
# =============================================================================
# Load the encoding fusion model regression weights
weight_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'encoding_fusion_weights',
    f'weights_fmri_sub-{args.fmri_subject:02d}.npy')
reg_param = np.load(weight_dir, allow_pickle=True).item()

# Loop across ROIs
correlation = {}
rois = ['V1', 'V2', 'V3', 'hV4', 'lFFA', 'rFFA', 'lOFA', 'rOFA', 'lEBA',
    'rEBA', 'lPPA', 'rPPA', 'lRSC', 'rRSC', 'lTOS', 'rTOS', 'lLOC', 'rLOC',
    'IT', 'lSTS', 'rSTS']
for r, roi in enumerate(tqdm(rois)):

    # Empty correlation array of shape:
    # (N ROI voxels, 140 MEG time points)
    n_voxels = len(metadata_fmri['roi'][roi])
    correlation[roi] = np.zeros((n_voxels, len(times)), dtype=np.float32)

    # Get the fMRI voxel responses for the current ROI
    roi_idx = metadata_fmri['roi'][roi]
    fmri_test_roi = fmri_test[:,roi_idx]

    # Center and normalize the test fMRI responses (for later correlation)
    eps = 1e-8
    fmri_test_roi_z = (fmri_test_roi - fmri_test_roi.mean(0)) /  \
        (fmri_test_roi.std(0) + eps)

    # Loop across MEG time points
    for t in tqdm(range(len(times))):

        # Instantiate the fusion regression model
        reg = LinearRegression()
        reg.coef_ = reg_param[roi]['coef_'][t]
        reg.intercept_ = reg_param[roi]['intercept_'][t]
        reg.n_features_in_ = reg_param[roi]['n_features_in_'][t]

        # Generate the t-fMRI responses for the test images with in vivo MEG
        tfmri = reg.predict(meg_test[:,:,t])

        # Center and normalize the t-fMRI responses
        tfmri_z = (tfmri - tfmri.mean(0)) /  (tfmri.std(0) + eps)

        # Correlate the t-fMRI test responses with the fMRI test responses
        correlation[roi][:,t] = \
            np.diag(tfmri_z.T @ fmri_test_roi_z) / len(tfmri_z)

        # Delete unused variables
        del tfmri, tfmri_z, reg
        gc.collect()
    del fmri_test_roi, fmri_test_roi_z
del reg_param, metadata_fmri, metadata_meg, meg_test


# =============================================================================
# Save the results
# =============================================================================
# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'encoding_fusion_accuracy')
os.makedirs(save_dir, exist_ok=True)

# Save the correlation scores
file_name = f'corr_fmri_sub-{args.fmri_subject:02d}.npy'
np.save(os.path.join(save_dir, file_name), correlation)