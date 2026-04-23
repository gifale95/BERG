"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 1000 NSD shared images. These t-fMRI
responses are then correlated with the corresponding in vivo fMRI responses
from NSD, resulting in one encoding accuracy score for each fMRI vertex and
EEG time point.

To reduce computational load, the M/EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import gc
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int) # '1' '2' '5' '7'
parser.add_argument('--hemisphere', default='lh', type=str) # 'lh' 'rh'
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the fMRI and EEG test responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'invivo_test_data')
file_name = f'fmri_test_sub-{args.subject:02d}_hemi-{args.hemisphere}.npy'
fmri_test_dict = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()
fmri_test = np.mean(fmri_test_dict['fmri_test'], 1)
del fmri_test_dict

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + eps)

# Load the EEG responses
file_name = f'eeg_test_sub-{args.subject:02d}_.npy'
eeg_test_dict = np.load(os.path.join(data_dir, file_name),
    allow_pickle=True).item()
eeg_test = np.mean(eeg_test_dict['eeg_test'], 1)
del eeg_test_dict

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


# =============================================================================
# Test the encoding fusion models
# =============================================================================
# Empty correlation array of shape:
# (N fMRI vertices, 140 M/EEG time points)
correlation = np.zeros((fmri_test.shape[1], len(times)), dtype=np.float32)

# Loop across EEG time points
for t in tqdm(range(len(times))):

    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_sub-{args.subject:02d}_hemi-{args.hemisphere}_'
        f'eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'invivo_nsd_eeg_fmri_control', 'encoding_fusion_weights',
        file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses for the test images with in vivo EEG
    tfmri = reg.predict(eeg_test[:,:,t])

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
results = {
    'correlation': correlation,
    'times': times
}

# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_nsd_eeg_fmri_control', 'encoding_fusion_accuracy')
os.makedirs(save_dir, exist_ok=True)

# Save the correlation scores
file_name = f'corr_sub-{args.subject:02d}_hemi-{args.hemisphere}.npy'
np.save(os.path.join(save_dir, file_name), results)