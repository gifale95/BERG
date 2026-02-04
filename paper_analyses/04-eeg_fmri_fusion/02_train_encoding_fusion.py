"""Train the M/EEG-fMRI encoding fusion models by linearly mapping in vivo
M/EEG responses onto in silico fMRI responses using the training images.

One regression model is trained for each fMRI vertex and M/EEG time point,
using the EEG channel responses appended across the 10 THINGS EEG2 subjects
(or the MEG channel responses appended across the 4 THINGS MEG1 subjects).

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
import random
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--source_dataset', default='things_eeg_2', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
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
# Create the encoding fusion model weight save directory
# =============================================================================
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_weights', f'source_dataset-{args.source_dataset}')
os.makedirs(save_dir_weights, exist_ok=True)


# =============================================================================
# Load the in silico fMRI train responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses', f'imageset-{args.source_dataset}')
file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_split-train.h5'
fmri_train = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
idx_v = np.zeros(fmri_train.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]
fmri_train = fmri_train[:,idx_v]


# =============================================================================
# Load and append the in vivo EEG train responses across subjects
# =============================================================================
if args.source_dataset == 'things_eeg_2':

    # Loop across subjects
    for es, esub in enumerate(tqdm(args.eeg_subjects)):

        # Load the EEG responses, and average them across repeats
        eeg_train_dir = os.path.join(args.berg_dir, 'model_training_datasets',
            'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-train.h5')
        eeg_train_sub = np.mean(h5py.File(eeg_train_dir, 'r')['eeg'][:],
            1).astype(np.float32)

        # Append the EEG channel responses across subjects
        if es == 0:
            source_train = eeg_train_sub
        else:
            source_train = np.append(source_train, eeg_train_sub, 1)
        del eeg_train_sub

    # Load the EEG time points
    metadata_eeg = berg.get_model_metadata(
        'eeg-things_eeg_2-vit_b_32',
        subject=1
    )
    times = metadata_eeg['eeg']['times']


# =============================================================================
# Load and append the in vivo MEG train responses across subjects
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
        img_ids = metadata_meg['encoding_model']['all_training_splits']\
            ['train_img_ids'].astype(int)
        idx_sort = np.argsort(img_ids)

        # Load and sort the MEG responses according to the image IDs
        meg_train_dir = os.path.join(args.berg_dir, 'model_training_datasets',
            'train_dataset-things_meg_1', f'meg_P{msub}_split-train.h5')
        meg_train_sub = h5py.File(meg_train_dir, 'r')['neural_data']\
            [:,:,time_idx].astype(np.float32)
        meg_train_sub = meg_train_sub[idx_sort]

        # Append the MEG sensor responses across subjects
        if ms == 0:
            source_train = meg_train_sub
        else:
            source_train = np.append(source_train, meg_train_sub, 1)
        del meg_train_sub


# =============================================================================
# Train the encoding fusion models
# =============================================================================
# Loop across M/EEG time points
for t in tqdm(range(len(times))):

    # Train the encoding fusion models
    alphas = np.logspace(-6, 10, 17)
    reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    reg.fit(source_train[:,:,t], fmri_train)

    # Store the encoding fusion model weights
    reg_param = {
        'coef_': reg.coef_.astype(np.float32),
        'intercept_': reg.intercept_.astype(np.float32),
        'alpha_': reg.alpha_,
        'n_features_in_': reg.n_features_in_
    }

    # Save the encoding fusion model weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    np.save(os.path.join(save_dir_weights, file_name), reg_param)
    del reg_param