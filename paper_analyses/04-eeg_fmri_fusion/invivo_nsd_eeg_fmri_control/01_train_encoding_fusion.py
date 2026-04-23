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
berg_dir : str
    Directory of the BERG.
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/
tnsd_dir : str
    Directory of the Temporal Natural Scenes Dataset.

"""

import argparse
import os
import numpy as np
import h5py
from tnsd_access import TrialHandler
from scipy.stats import zscore
from tqdm import tqdm
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=2, type=int) # '1' '2' '5' '7'
parser.add_argument('--hemisphere', default='lh', type=str) # 'lh' 'rh'
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--tnsd_dir', default='/scratch/giffordale95/datasets/temporal-natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directories
# =============================================================================
# Create the test fMRI/EEG save directories
save_dir_test = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_test_data')
os.makedirs(save_dir_test, exist_ok=True)

# Create the encoding fusion model weight save directory
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights')
os.makedirs(save_dir_weights, exist_ok=True)


# =============================================================================
# Load the fMRI responses
# =============================================================================
# Load the fMRI responses and metadata
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-nsd_fsaverage')
fmri_file_name = f'{args.hemisphere}_betas_subject-{args.subject}.h5'
meta_file_name = f'metadata_subject-{args.subject}.npy'
fmri = h5py.File(os.path.join(data_dir, fmri_file_name), 'r')['betas']
metadata_fmri = np.load(os.path.join(data_dir, meta_file_name),
    allow_pickle=True).item()

# Only select vertices falling within the NSD visual streams
idx_v = np.zeros(fmri.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri[f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]
fmri = fmri[:,idx_v]

# Select the fMRI responses for the training images, and average them across
# repeats
train_img_num = metadata_fmri['train_img_num']
train_img_num.sort()
fmri_train = []
for img_num in train_img_num:
    idx = np.where(metadata_fmri['img_presentation_order'] == img_num)[0]
    fmri_train.append(np.mean(fmri[idx], 0))
fmri_train = np.array(fmri_train)
train_img_num += 1 # since the EEG image numbers are 1 based

# Store the fMRI responses responses for the test images
test_img_num = np.append(metadata_fmri['test_img_num'],
    metadata_fmri['val_img_num'])
test_img_num.sort()
fmri_test = []
for img_num in test_img_num:
    idx = np.where(metadata_fmri['img_presentation_order'] == img_num)[0]
    fmri_test.append(fmri[idx])
fmri_test = np.array(fmri_test)
test_img_num += 1 # since the EEG image numbers are 1 based
fmri_test_dict = {
    'fmri_test': fmri_test,
    'test_img_num': test_img_num
}
np.save(os.path.join(save_dir_test, (f'fmri_test_sub-{args.subject:02d}_'
    f'hemi-{args.hemisphere}.npy')), fmri_test_dict)
del fmri, fmri_test, fmri_test_dict


# =============================================================================
# Load the EEG responses
# =============================================================================
# Initialize tNSD data loader
loader = TrialHandler(args.tnsd_dir)

# Get the time points # !!! Use official time points
n_times = 615
times = np.round(np.linspace(-200, 1000, n_times)).astype(int)

# Account for the 50ms shift in the EEG responses # !!!
shift = -50
times = times + shift

# Only select time points between -100ms and 600ms
t_start = np.where(times == -100)[0][0]
t_end = np.where(times == 600)[0][0]
times = times[t_start:t_end+1]

# Load and z-score the EEG responses at each session
sessions = 36
conditions = []
eeg = []
for ses in tqdm(range(1, sessions+1)):
    trials_sess = loader.lookup_trials(subject=args.subject, session=ses)
    data_sess = loader.get_data(trials_sess)
    conditions.append(np.array(data_sess['metadata']['condition']))
    eeg.append(zscore(data_sess['data'][:,:-4,t_start:t_end+1], 0)) # !!! Select channels using official channels
    del data_sess

# Concatenate the data across sessions
conditions = np.concatenate(conditions)
eeg = np.concatenate(eeg, 0)

# Select the EEG responses for the training images, and average them across
# repeats
eeg_train = []
for img_num in train_img_num:
    idx = np.where(conditions == img_num)[0]
    eeg_train.append(np.mean(eeg[idx], 0))
eeg_train = np.array(eeg_train)

# Store the EEG responses for the test images
eeg_test = []
for img_num in test_img_num:
    idx = np.where(conditions == img_num)[0]
    eeg_test.append(eeg[idx])
eeg_test = np.array(eeg_test)
eeg_test_dict = {
    'eeg_test': eeg_test,
    'test_img_num': test_img_num
}
np.save(os.path.join(save_dir_test, f'eeg_test_sub-{args.subject:02d}_.npy'),
    eeg_test_dict)
del eeg, eeg_test, eeg_test_dict


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
    file_name = (f'weights_sub-{args.subject:02d}_'
                f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    np.save(os.path.join(save_dir_weights, file_name), reg_param)
    del reg, reg_param