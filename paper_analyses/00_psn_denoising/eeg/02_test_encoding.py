"""Test the THINGS EEG2 encoding models while optionally denoising the neural
responses with PSN.

PSN GitHub: https://github.com/jacob-prince/PSN

Parameters
----------
subjects : list
    List of subject identifier sfor the THINGS EEG2 data. Valid subject
    identifiers are integers from 1 to 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG test responses
# =============================================================================
# Loop across subjects
for s, sub in enumerate(tqdm(range(args.subjects))):

    # Load the EEG responses
    eeg_dir = os.path.join(args.berg_dir, 'psn_denoising', 'eeg',
    'eeg_test_responses', 'eeg_test_subject-'+format(sub, '02')+'.npy')
    eeg = np.load(eeg_dir, allow_pickle=True).item()


eeg = {
    'invivo_eeg_vte-0': np.mean(eeg_test, 2), # Average across repeats
    'invivo_eeg_vte-1': np.mean(eeg_test_denoised, 2), # Average across repeats
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


# Reshape the EEG responses to (Units, Conditions, Repeats)
n_cond = invivo_eeg_test.shape[0]
n_trial = invivo_eeg_test.shape[1]
n_chan = invivo_eeg_test.shape[2]
n_time = invivo_eeg_test.shape[3]
invivo_eeg_test = np.reshape(invivo_eeg_test, (n_cond, n_trial, -1))
invivo_eeg_test = np.swapaxes(np.swapaxes(invivo_eeg_test, 0, 2), 1, 2)


# =============================================================================
# Compute the NCSNR # !!! ALL data types
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
# Compute the encoding accuracy # !!! ALL DATA COMBINATION PAIRS
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
# Confidence intervals # !!! ALL DATA COMBINATION PAIRS
# =============================================================================


# =============================================================================
# Significance # !!! ALL DATA COMBINATION PAIRS
# =============================================================================


# =============================================================================
# Save the results # !!!
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