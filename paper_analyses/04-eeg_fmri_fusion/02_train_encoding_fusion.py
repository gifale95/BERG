"""Train the EEG-fMRI encoding fusion models by linearly mapping in vivo
EEG responses onto in silico fMRI responses using the training images.

One regression model is trained for each fMRI vertex and EEG time point,
using the EEG channel responses appended across the 10 THINGS EEG2 subjects.

To reduce computational load, the EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

The in vivo THINGS EEG2 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_eeg_2

Parameters
----------
fmri_subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
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
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_train_trials', default='all', type=str)
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
# Create the encoding fusion model weight save directory
# =============================================================================
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_weights', )
os.makedirs(save_dir_weights, exist_ok=True)


# =============================================================================
# Load the in silico fMRI train responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses', 'imageset-things_eeg_2')
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
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_train_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-train.h5')
    eeg_train_sub = h5py.File(eeg_train_dir, 'r')['eeg'].astype(np.float32)
    if args.eeg_train_trials == 'all':
        eeg_train_sub = np.mean(eeg_train_sub, 1)
    elif args.eeg_train_trials == 'even':
        eeg_train_sub = np.mean(eeg_train_sub[:,::2], 1)
    elif args.eeg_train_trials == 'odd':
        eeg_train_sub = np.mean(eeg_train_sub[:,1::2], 1)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_train = eeg_train_sub
    else:
        eeg_train = np.append(eeg_train, eeg_train_sub, 1)
    del eeg_train_sub

# Load the EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Train the encoding fusion models
# =============================================================================
# Loop across EEG time points
for t in tqdm(range(len(times))):

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
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                f'hemi-{args.hemisphere}_eeg_train_trials-'
                f'{args.eeg_train_trials}_eeg_time-{t:03d}.npy')
    np.save(os.path.join(save_dir_weights, file_name), reg_param)
    del reg_param