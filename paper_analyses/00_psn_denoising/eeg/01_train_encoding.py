"""Train THINGS EEG2 encoding models while optionally denoising the neural
responses with PSN.

PSN GitHub: https://github.com/jacob-prince/PSN

Parameters
----------
subject : int
    Subject identifier for the THINGS EEG2 data. Valid subject identifiers are
    integers from 1 to 10.
psn_invivo_train : int
    If 0, do not apply PSN on the in vivo EEG training responses.
    If 1, apply PSN on the in vivo EEG training responses.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
import copy
import psn
from psn import PSN
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--psn_invivo_train', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the image feature maps
# =============================================================================
# Training feature maps
fmap_dir = os.path.join(args.berg_dir, 'results', 'train_encoding_models',
    'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32',
    'image_features_pca', 'image_features_train.npy')
dnn_train = np.load(fmap_dir)[:,:250]

# Testing feature maps
fmap_dir = os.path.join(args.berg_dir, 'results', 'train_encoding_models',
    'modality-eeg', 'train_dataset-things_eeg_2', 'model-vit_b_32',
    'image_features_pca', 'image_features_test.npy')
dnn_test = np.load(fmap_dir)[:,:250]


# =============================================================================
# Load the EEG training responses
# =============================================================================
# Load the EEG responses
eeg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2', 'eeg_sub-'+format(args.subject, '02')+
    '_split-train.h5')
eeg_train = h5py.File(eeg_dir, 'r')['eeg'][:].astype(np.float32)

# Reshape the EEG responses to (Units, Conditions, Repeats)
n_cond = eeg_train.shape[0]
n_trial = eeg_train.shape[1]
n_chan = eeg_train.shape[2]
n_time = eeg_train.shape[3]
eeg_train = np.reshape(eeg_train, (n_cond, n_trial, -1))
eeg_train = np.swapaxes(np.swapaxes(eeg_train, 0, 2), 1, 2)


# =============================================================================
# Denoise the EEG train responses
# =============================================================================
if args.psn_invivo_train == 1:

    # denoisingtype : int, default=0
    #     Type of denoising to perform:
    #     - 0: Trial-averaged denoising (returns nunits x nconds)
    #     - 1: Single-trial denoising (returns nunits x nconds x ntrials)

    denoiser = PSN(
        basis='signal',
        cv='unit',
        scoring='mse',
        mag_threshold=0.95,
        unit_groups=None,
        truncate=0,
        ranking=None,
        cv_thresholds=None,
        cv_mode=None,
        denoisingtype=1,
        verbose=True,
        wantfig=False,
        gsn_kwargs=None
    )

    denoiser.fit(eeg_train)

    eeg_train = denoiser.transform(eeg_train)


# =============================================================================
# Fit an encoding model at each EEG repeat, time-point and channel
# =============================================================================
reg_param = {}
eeg_test_pred = np.zeros((len(dnn_test), n_trial, n_chan, n_time), dtype=np.float32)

# Loop over the 4 training EEG repeats
for r in range(n_trial): 

    # Reshape the EEG to (Samples x Features)
    eeg = np.swapaxes(eeg_train[:,:,r], 0, 1)

    # Fit the linear regression
    reg = LinearRegression()
    reg.fit(dnn_train, eeg)

    # Store the linear regression weights
    reg_dict = {
        'coef_': reg.coef_,
        'intercept_': reg.intercept_,
        'n_features_in_': reg.n_features_in_
        }
    reg_param['rep-'+str(r+1)] = copy.deepcopy(reg_dict)

    # Use the learned weights to generate in silico EEG responses for the test
    # images
    eeg_test_pred_rep = reg.predict(dnn_test)
    eeg_test_pred[:,r] = np.reshape(eeg_test_pred_rep, (-1, n_chan, n_time))
    del reg_dict

# Save the in silico EEG responses for the test images
save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'insilico_test_responses')
os.makedirs(save_dir, exist_ok=True)
file_name = 'eeg_test_pred_subject-' + format(args.subject, '02') + \
    '_psn_invivo_train-' + str(args.psn_invivo_train) + '.npy'
np.save(os.path.join(save_dir, file_name), eeg_test_pred)

# Save the trained encoding model weights
weights = {
    'reg_param': reg_param
    }
save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'encoding_models_weights')
os.makedirs(save_dir, exist_ok=True)
file_name = 'weights_subject-' + format(args.subject, '02') + \
    '_psn_invivo_train-' + str(args.psn_invivo_train) + '.npy'
np.save(os.path.join(save_dir, file_name), eeg_test_pred)