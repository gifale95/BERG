"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 200 THINGS EEG2 test images. These t-fMRI responses
are then correlated with the in silico fMRI responses for the same test images,
resulting in one encoding accuracy score for each fMRI vertex and M/EEG time
point.

To reduce computational load, the M/EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

The in vivo THINGS EEG2 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_eeg_2

The in vivo THINGS MEG1 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_meg_1

Parameters
----------
fmri_subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
source_dataset : str
    If 'things_eeg_2', the source dataset is THINGS EEG2. If 'things_meg_1',
    the source dataset  is THINGS MEG1. (The source dataset is the dataset that
    is mapped onto fMRI responses.)
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
meg_subjects : list
    List containing the subject identifiers for the THINGS MEG1 subjects. Valid
    subject identifiers are integers from 1 to 4.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

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
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--source_dataset', default='things_eeg_2', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses', f'imageset-{args.source_dataset}')
file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_split-test.h5'
fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
idx_v = np.zeros(fmri_test.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]
fmri_test = fmri_test[:,idx_v]

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + eps)


# =============================================================================
# Load and append the in vivo EEG test responses across subjects
# =============================================================================
if args.source_dataset == 'things_eeg_2':

    # Loop across subjects
    for es, esub in enumerate(tqdm(args.eeg_subjects)):

        # Load the EEG responses, and average them across repeats
        eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
            'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
        eeg_test_sub = np.mean(h5py.File(eeg_dir_test, 'r')['eeg'][:],
            1).astype(np.float32)

        # Append the EEG channel responses across subjects
        if es == 0:
            source_test = eeg_test_sub
        else:
            source_test = np.append(source_test, eeg_test_sub, 1)
        del eeg_test_sub

    # Load the EEG time points
    berg = BERG(berg_dir=args.berg_dir)
    metadata_eeg = berg.get_model_metadata(
        'eeg-things_eeg_2-vit_b_32',
        subject=1
    )
    times = metadata_eeg['eeg']['times']


# =============================================================================
# Load and append the in vivo MEG test responses across subjects
# =============================================================================
elif args.source_dataset == 'things_meg_1':

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

        # Get the image metadata
        img_ids = metadata_meg['encoding_model']['test_img_ids'].astype(int)
        unique_img_ids = np.unique(img_ids)

        # Load the MEG responses, average them across repeats and sort them
        # according to the image IDs
        meg_test_dir = os.path.join(args.berg_dir, 'model_training_datasets',
            'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
        meg_test_sub = h5py.File(meg_test_dir, 'r')['neural_data']\
            [:,:,time_idx].astype(np.float32)
        meg_test_sub_rep_avg = []
        for id in unique_img_ids:
            idx = np.where(img_ids == id)[0]
            meg_test_sub_rep_avg.append(np.mean(meg_test_sub[idx], 0))
        meg_test_sub_rep_avg = np.array(meg_test_sub_rep_avg)
        del meg_test_sub

        # Append the MEG sensor responses across subjects
        if ms == 0:
            source_test = meg_test_sub_rep_avg
        else:
            source_test = np.append(source_test, meg_test_sub_rep_avg, 1)
        del meg_test_sub_rep_avg


# =============================================================================
# Test the encoding fusion models
# =============================================================================
# Empty correlation array of shape:
# (N fMRI vertices, 140 M/EEG time points)
correlation = np.zeros((fmri_test.shape[1], len(times)), dtype=np.float32)

# Loop across M/EEG time points
for t in tqdm(range(len(times))):

    # Load the M/EEG-fMRI encoding fusion models weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
        f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights', f'source_dataset-{args.source_dataset}',
        file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses for the test images with in vivo M/EEG
    tfmri = reg.predict(source_test[:,:,t])

    # Center and normalize the t-fMRI responses
    tfmri_z = (tfmri - tfmri.mean(0)) /  (tfmri.std(0) + eps)

    # Correlate the t-fMRI test responses with the fMRI test responses
    correlation[:,t] = np.diag(tfmri_z.T @ fmri_test_z) / len(tfmri_z)

    # Delete unused variables
    del tfmri, reg_param, reg
    gc.collect()


# =============================================================================
# Save the results
# =============================================================================
# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_accuracy', f'source_dataset-{args.source_dataset}')
os.makedirs(save_dir, exist_ok=True)

# Save the correlation scores
file_name = (f'corr_fmri_sub-{args.fmri_subject:02d}'
            f'_hemi-{args.hemisphere}.npy')
np.save(os.path.join(save_dir, file_name), correlation)