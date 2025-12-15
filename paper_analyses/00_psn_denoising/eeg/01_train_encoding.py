"""Train THINGS EEG2 encoding models while optionally denoising the neural
responses with PSN.

PSN GitHub: https://github.com/jacob-prince/PSN

Parameters
----------
subject : int
    Subject identifier for the THINGS EEG2 data. Valid subject identifiers are
    integers from 1 to 10.
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
# Load the EEG responses
# =============================================================================
# Load the EEG responses
eeg_dir_train = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2', 'eeg_sub-'+format(args.subject, '02')+
    '_split-train.h5')
eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2', 'eeg_sub-'+format(args.subject, '02')+
    '_split-test.h5')
eeg_train = h5py.File(eeg_dir_train, 'r')['eeg'][:].astype(np.float32)
eeg_test = h5py.File(eeg_dir_test, 'r')['eeg'][:].astype(np.float32)

# Reshape the EEG responses to (Units, Conditions, Repeats)
n_cond_train = eeg_train.shape[0]
n_trial_train = eeg_train.shape[1]
n_chan = eeg_train.shape[2]
n_time = eeg_train.shape[3]
eeg_train = np.reshape(eeg_train, (n_cond_train, n_trial_train, -1))
eeg_train = np.swapaxes(np.swapaxes(eeg_train, 0, 2), 1, 2)
n_cond_test = eeg_test.shape[0]
n_trial_test = eeg_test.shape[1]
eeg_test = np.reshape(eeg_test, (n_cond_test, n_trial_test, -1))
eeg_test = np.swapaxes(np.swapaxes(eeg_test, 0, 2), 1, 2)


# =============================================================================
# Denoise the EEG responses
# =============================================================================

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

eeg_train_denoised = denoiser.transform(eeg_train)
eeg_test_denoised = denoiser.transform(eeg_test)


# =============================================================================
# Fit an encoding model at each EEG repeat, time-point and channel
# =============================================================================
reg_param = {}
eeg_test_pred = np.zeros((n_chan*n_time, n_cond_test, n_trial_train),
    dtype=np.float32)

# Loop over the 4 training EEG repeats
for r in range(n_trial_train): 

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
    eeg_test_pred[:,:,r] = np.swapaxes(reg.predict(dnn_test), 0, 1)
    del reg_dict


# =============================================================================
# Fit an encoding model at each EEG repeat, time-point and channel
# (denoised train responses)
# =============================================================================
reg_param_psn_train = {}
eeg_test_pred_psn_train = np.zeros((n_chan*n_time, n_cond_test, n_trial_train),
    dtype=np.float32)

# Loop over the 4 training EEG repeats
for r in range(n_trial_train): 

    # Reshape the EEG to (Samples x Features)
    eeg = np.swapaxes(eeg_train_denoised[:,:,r], 0, 1)

    # Fit the linear regression
    reg = LinearRegression()
    reg.fit(dnn_train, eeg)

    # Store the linear regression weights
    reg_dict = {
        'coef_': reg.coef_,
        'intercept_': reg.intercept_,
        'n_features_in_': reg.n_features_in_
        }
    reg_param_psn_train['rep-'+str(r+1)] = copy.deepcopy(reg_dict)

    # Use the learned weights to generate in silico EEG responses for the test
    # images
    eeg_test_pred_psn_train[:,:,r] = np.swapaxes(reg.predict(dnn_test), 0, 1)
    del reg_dict


# =============================================================================
# Denoise the in silico EEG responses
# =============================================================================
eeg_test_pred_denoised = denoiser.transform(eeg_test_pred)
eeg_test_pred_psn_train_denoised = denoiser.transform(eeg_test_pred_psn_train)


# =============================================================================
# Save the results
# =============================================================================
# Reshape the in vivo EEG responses to: (n_cond, n_trial, n_chan, n_time)
eeg_test = np.reshape(np.swapaxes(np.swapaxes(eeg_test, 0, 1), 1, 2),
    (n_cond_test, n_trial_test, n_chan, n_time))
eeg_test_denoised = np.reshape(np.swapaxes(np.swapaxes(eeg_test_denoised, 0, 1), 1, 2),
    (n_cond_test, n_trial_test, n_chan, n_time))

# Save the EEG responses for the test images
# vtr: PSN applied to in vivo (v) train data (tr)
# vte: PSN applied to in vivo (v) test data (te)
# ste: PSN applied to in silico (s) test data (te)
eeg = {
    'invivo_eeg_vte-0': eeg_test,
    'invivo_eeg_vte-1': eeg_test_denoised,
    'insilico_eeg_vtr-0_ste-0': eeg_test_pred,
    'insilico_eeg_vtr-1_ste-0': eeg_test_pred_psn_train,
    'insilico_eeg_vtr-0_ste-1': eeg_test_pred_denoised,
    'insilico_eeg_vtr-1_ste-1': eeg_test_pred_psn_train_denoised

}
save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'eeg_test_responses')
os.makedirs(save_dir, exist_ok=True)
file_name = 'eeg_test_subject-' + format(args.subject, '02') + '.npy'
np.save(os.path.join(save_dir, file_name), eeg_test_pred)

# Save the trained encoding model weights
weights = {
    'reg_param_vtr-0': reg_param,
    'reg_param_vtr-1': reg_param_psn_train
    }
save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'encoding_models_weights')
os.makedirs(save_dir, exist_ok=True)
file_name = 'weights_subject-' + format(args.subject, '02') + '.npy'
np.save(os.path.join(save_dir, file_name), eeg_test_pred)