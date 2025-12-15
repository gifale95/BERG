"""Test the THINGS EEG2 encoding models while optionally denoising the neural
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
psn_invivo_test : int
    If 0, do not apply PSN on the in vivo EEG testing responses.
    If 1, apply PSN on the in vivo EEG testing responses.
psn_insilico_test : int
    If 0, do not apply PSN on the in silico EEG testing responses.
    If 1, apply PSN on the in silico EEG testing responses.
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
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--psn_invivo_train', default=0, type=int)
parser.add_argument('--psn_invivo_test', default=0, type=int)
parser.add_argument('--psn_insilico_test', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in vivo EEG test responses
# =============================================================================
# Load the EEG responses
eeg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2', 'eeg_sub-'+format(args.subject, '02')+
    '_split-test.h5')
invivo_eeg_test = h5py.File(eeg_dir, 'r')['eeg'][:].astype(np.float32)

# Reshape the EEG responses to (Units, Conditions, Repeats)
n_cond = invivo_eeg_test.shape[0]
n_trial = invivo_eeg_test.shape[1]
n_chan = invivo_eeg_test.shape[2]
n_time = invivo_eeg_test.shape[3]
invivo_eeg_test = np.reshape(invivo_eeg_test, (n_cond, n_trial, -1))
invivo_eeg_test = np.swapaxes(np.swapaxes(invivo_eeg_test, 0, 2), 1, 2)


# =============================================================================
# Load the in silico EEG test responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'insilico_test_responses', 'eeg_test_pred_subject-'+
    format(args.subject, '02')+'_psn_invivo_train-'+
    str(args.psn_invivo_train)+'.npy')
insilico_eeg_test = np.load(data_dir)

# Reshape the EEG responses to (Units, Conditions, Repeats)
insilico_eeg_test = np.reshape(insilico_eeg_test, (n_cond, n_trial, -1))
insilico_eeg_test = np.swapaxes(np.swapaxes(insilico_eeg_test, 0, 2), 1, 2)


# =============================================================================
# Denoise the EEG test responses
# =============================================================================
if args.psn_invivo_test == 1 or args.psn_insilico_test == 1:

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

    denoiser.fit(invivo_eeg_test)

if args.psn_invivo_test == 1:

    invivo_eeg_test = denoiser.transform(invivo_eeg_test)

if args.psn_insilico_test == 1:

    insilico_eeg_test = denoiser.transform(insilico_eeg_test)


# =============================================================================
# Compute the NCSNR (in vivo EEG test responses)
# =============================================================================
    # Estimate the noise standard deviation (calculate the variance of the
    # responses across the 30 presentations of each test image).
    var = np.nanvar(invivo_eeg_test, axis=2, ddof=1)

    # Average the variance across images and compute the square root of the
    # result
    sigma_noise = np.sqrt(np.nanmean(var, 1))

    # Estimate the signal standard deviation (total variance - noise variance)
    tot_var_data = np.nanvar(np.reshape(invivo_eeg_test,
        (invivo_eeg_test.shape[0], -1)), axis=1, ddof=1)
    sigma_signal = tot_var_data - (sigma_noise ** 2)
    sigma_signal[sigma_signal<0] = 0
    sigma_signal = np.sqrt(sigma_signal)

    # Compute the ncsnr
    ncsnr_invivo = sigma_signal / sigma_noise

    # Convert the ncsnr to noise ceiling (the noise ceiling is in r² explained
    # variance units)
    noise_ceiling_invivo = 100 * (ncsnr_invivo ** 2) / ((ncsnr_invivo ** 2) + (1 / n_trial))

    # Reshape the scores to (n_sensors, n_timepoints)
    ncsnr_invivo = ncsnr_invivo.reshape(n_chan, n_time)
    noise_ceiling_invivo = noise_ceiling_invivo.reshape(n_chan, n_time)


# =============================================================================
# Compute the NCSNR (in silico EEG test responses)
# =============================================================================
    # Estimate the noise standard deviation (calculate the variance of the
    # responses across the 30 presentations of each test image).
    var = np.nanvar(insilico_eeg_test, axis=2, ddof=1)

    # Average the variance across images and compute the square root of the
    # result
    sigma_noise = np.sqrt(np.nanmean(var, 1))

    # Estimate the signal standard deviation (total variance - noise variance)
    tot_var_data = np.nanvar(np.reshape(insilico_eeg_test,
        (insilico_eeg_test.shape[0], -1)), axis=1, ddof=1)
    sigma_signal = tot_var_data - (sigma_noise ** 2)
    sigma_signal[sigma_signal<0] = 0
    sigma_signal = np.sqrt(sigma_signal)

    # Compute the ncsnr
    ncsnr_insilico = sigma_signal / sigma_noise

    # Convert the ncsnr to noise ceiling (the noise ceiling is in r² explained
    # variance units)
    noise_ceiling_insilico = 100 * (ncsnr_insilico ** 2) / ((ncsnr_insilico ** 2) + (1 / n_trial))

    # Reshape the scores to (n_sensors, n_timepoints)
    ncsnr_insilico = ncsnr_insilico.reshape(n_chan, n_time)
    noise_ceiling_insilico = noise_ceiling_insilico.reshape(n_chan, n_time)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
# Average the EEG test responses across repeats
invivo_eeg_test = np.mean(invivo_eeg_test, 2)
insilico_eeg_test = np.mean(insilico_eeg_test, 2)

corr = np.zeros(invivo_eeg_test.shape[0], dtype=np.float32)

for u in tqdm(range(invivo_eeg_test.shape[0])):
    corr[u] = pearsonr(invivo_eeg_test[u], insilico_eeg_test[u])[0]

# Reshape the scores to (n_sensors, n_timepoints)
corr = corr.reshape(n_chan, n_time)

# =============================================================================
# Save the results
# =============================================================================
results = {
    'ncsnr_invivo': ncsnr_invivo,
    'noise_ceiling_invivo': noise_ceiling_invivo,
    'ncsnr_insilico': ncsnr_insilico,
    'noise_ceiling_insilico': noise_ceiling_insilico,
    'corr': corr
    }

save_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg', 'test_encoding')
os.makedirs(save_dir, exist_ok=True)

file_name = 'test_encoding_subject-' + format(args.subject, '02') + \
    '_psn_invivo_train-' + str(args.psn_invivo_train) + \
    '_psn_invivo_test-' + str(args.psn_invivo_test) + \
    '_psn_insilico_test-' + str(args.psn_insilico_test) + '.npy'

np.save(os.path.join(save_dir, file_name), results)