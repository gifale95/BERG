"""Perfrm MDS on the ROI-wise t-fMRI responses.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import random
from berg import BERG
from tqdm import tqdm
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> MDS <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load and append the in vivo EEG test responses across subjects
# =============================================================================
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
    eeg_test_sub = np.mean(h5py.File(eeg_dir_test, 'r')['eeg'][:],
        1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_test = eeg_test_sub
    else:
        eeg_test = np.append(eeg_test, eeg_test_sub, 1)
    del eeg_test_sub

# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Get the ROI voxel indices
# =============================================================================
# Empty ROI index dictionary
roi_idx = {}
metadata_fmri = []

# ROI list
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral', 'parietal']

# Loop across fMRI subjects
for fs, fsub in enumerate(args.fmri_subjects):

    # Load the fMRI metadata
    metadata = berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=fsub
        )
    metadata_fmri.append(metadata)

    # Loop across ROIs
    for r, roi in enumerate(rois):

        # Loop across subjects and hemispheres
        for h, hemi in enumerate(args.hemispheres):

            # Empty ROI index lists
            if fs == 0:
                roi_idx[f'{hemi}_{roi}'] = []

            # Get the indices of the ROI vertices
            if roi in ['V1', 'V2', 'V3']:
                idx_roi = np.append(
                    metadata['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                    metadata['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
                idx_roi.sort()
            elif roi in ['FFA', 'VWFA', 'FBA']:
                idx_roi = np.append(
                    metadata['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                    metadata['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
                idx_roi.sort()
            elif roi in ['intermediate']:
                idx_roi = np.append(
                    metadata['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
                    metadata['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
                idx_roi = np.append(idx_roi,
                    metadata['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
                idx_roi.sort()
            else:
                idx_roi = metadata['fmri'][f'{hemi}_fsaverage_rois'][roi]

            # NCSNR and encoding accuracy vertex selection
            ncsnr = metadata['fmri'][f'{hemi}_ncsnr'][idx_roi]
            idx_ncsnr = ncsnr >= args.ncsnr_threshold
            encoding = metadata['encoding_models']\
                [f'{hemi}_explained_variance_nsdcore'][idx_roi]
            idx_encoding = encoding >= args.encoding_threshold
            idx_vertex = np.logical_and(idx_ncsnr, idx_encoding)
            idx_roi = idx_roi[idx_vertex]

            # Store the ROI vertex indices
            roi_idx[f'{hemi}_{roi}'].append(idx_roi)
            del idx_roi


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Empty t-fMRI response dictionary
tfmri = {}

# Loop across fMRI subjects
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):

    # Only select vertices falling within the NSD visual streams
    n_vertices = 163842
    streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
        'lateral', 'parietal']
    idx_v = {}
    for hemi in args.hemispheres:
        idx = np.zeros(n_vertices, dtype=int)
        for stream in streams:
            idx[metadata_fmri[fs]['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
        idx_v[hemi] = np.where(idx == 1)[0]
        del idx

    # Loop across EEG time points and fMRI hemispheres
    for t in range(len(times)):
        for h, hemi in enumerate(args.hemispheres):

            # Load the EEG-fMRI encoding fusion models weights
            file_name = (f'weights_fmri_sub-{fsub:02d}_'
                f'hemi-{hemi}_eeg_time-{t:03d}.npy')
            reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
                'encoding_fusion_weights', file_name), allow_pickle=True).item()

            # Instantiate the fusion regression model
            reg = LinearRegression()
            reg.coef_ = reg_param['coef_']
            reg.intercept_ = reg_param['intercept_']
            reg.n_features_in_ = reg_param['n_features_in_']

            # Empty t-fMRI response array of shape:
            # (N Images, 163842 Vertices)
            tfmri_hemi = np.zeros((len(eeg_test), n_vertices), dtype=np.float32)

            # Generate the t-fMRI responses using the in vivo EEG responses
            tfmri_hemi[:,idx_v[hemi]] = reg.predict(eeg_test[:,:,t])
            del reg_param, reg

            # Loop across ROIs
            for r, roi in enumerate(rois):

                # Empty t-fMRI response lists
                if t == 0 and h == 0:
                    tfmri[f's{fsub}_{roi}'] = []

                # Get the t-fMRI responses for the current ROI and hemisphere
                tfmri_roi = tfmri_hemi[:,roi_idx[f'{hemi}_{roi}'][fs]]

                # Remove NaN values
                idx_nan = np.isnan(tfmri_roi).any(axis=0)
                tfmri_roi = tfmri_roi[:,~idx_nan]

                # Store the t-fMRI responses appended across hemispheres
                if h == 0:
                    tfmri[f's{fsub}_{roi}'].append(tfmri_roi)
                else:
                    tfmri[f's{fsub}_{roi}'][t] = np.append(
                        tfmri[f's{fsub}_{roi}'][t], tfmri_roi, 1)
                del tfmri_roi
                
            del tfmri_hemi


# =============================================================================
# Perform MDS (single fMRI subjects)
# =============================================================================
# Empty MDS dictionary
msd_sub_single = {}
n_components = 2

# Loop across fMRI subjects and ROIs
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):
    for r, roi in enumerate(rois):

        # Empty results array of shape (Images, 2 MDS dimensions, Times)
        msd_sub_single[f's{fsub}_{roi}'] = np.zeros(
            (len(eeg_test), n_components, len(times)), dtype=np.float32)

        # Loop across EEG time points
        for t in range(len(times)):

            # Perform MDS
            mds = MDS(n_components=n_components, n_init=10, max_iter=1000,
                random_state=seed)
            msd_sub_single[f's{fsub}_{roi}'][:,:,t] = mds.fit_transform(
                tfmri[f's{fsub}_{roi}'][t])


# =============================================================================
# Perform MDS (all fMRI subjects)
# =============================================================================
# Empty MDS dictionary
msd_sub_all = {}
n_components = 2

# Loop across fMRI ROIs
for r, roi in enumerate(tqdm(rois)):

    # Empty results array of shape (Images, 2 MDS dimensions, Times)
    msd_sub_all[roi] = np.zeros(
        (len(eeg_test), n_components, len(times)), dtype=np.float32)

    # Loop across EEG time points
    for t in range(len(times)):

        # Append the t-fMRI responses across fMRI subjects
        for fs, fsub in enumerate(args.fmri_subjects):
            if fs == 0:
                tfmri_all = tfmri[f's{fsub}_{roi}'][t]
            else:
                tfmri_all = np.append(tfmri_all, tfmri[f's{fsub}_{roi}'][t], 1)

        # Perform MDS
        mds = MDS(n_components=n_components, n_init=10, max_iter=1000,
            random_state=seed)
        msd_sub_all[roi][:,:,t] = mds.fit_transform(tfmri_all)
        del tfmri_all


# =============================================================================
# Save the results
# =============================================================================
results = {
    'msd_sub_single': msd_sub_single,
    'msd_sub_all': msd_sub_all,
    'metadata_fmri': metadata_fmri,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'animacy_mds',
    'mds')
os.makedirs(save_dir, exist_ok=True)

file_name = 'mds.npy'

np.save(os.path.join(save_dir, file_name), results)