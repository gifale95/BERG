"""Train the EEG-fMRI encoding fusion models by linearly mapping in vivo EEG
responses onto in silico fMRI responses for the 16,540 THINGS EEG2 train
images.

One regression model is trained for each fMRI vertex and EEG time point,
using the EEG channel responses appended across the 10 THINGS EEG2 subjects.

The trained regressions are then used to predict time-resolved fMRI (t-fMRI)
responses for the 200 THINGS EEG2 test images.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico fMRI train responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses')
file_name = f'things_eeg_2_train_sub-{args.fmri_subject:02d}_{args.hemisphere}.h5'

fmri_train = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri'][:]


# =============================================================================
# Load the in vivo EEG responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_eeg_responses')
file_train = 'things_eeg_2_train.h5'
file_test = 'things_eeg_2_test.h5'

eeg_train = h5py.File(os.path.join(data_dir, file_train), 'r')['eeg']
eeg_test = h5py.File(os.path.join(data_dir, file_test), 'r')['eeg']


# =============================================================================
# Train the encoding fusion models, and save the learned weights
# =============================================================================
# Empty t-fMRI response array of shape:
# (200 test images, 163842 fMRI vertices, 140 EEG time points)
tfmri_test = np.zeros((len(eeg_test), fmri_train.shape[1], eeg_test.shape[2]),
    dtype=np.float32)

# Loop across EEG time points
for t in tqdm(range(eeg_train.shape[2])):

    # Train the encoding fusion models
    reg = LinearRegression()
    reg.fit(eeg_train[:,:,t], fmri_train)

    # Save the encoding fusion model weights
    reg_param = {
        'coef_': reg.coef_.astype(np.float32),
        'intercept_': reg.intercept_.astype(np.float32),
        'n_features_in_': reg.n_features_in_
    }
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights')
    os.makedirs(save_dir, exist_ok=True)
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    np.save(os.path.join(save_dir, file_name), reg_param)


# =============================================================================
# Use the trained fusion models to predict t-fMRI responses for the test images
# =============================================================================
    # Predict the t-fMRI responses
    tfmri_test[:,:,t] = reg.predict(eeg_test[:,:,t])

# Save the t-fMRI responses for the test images
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'tfmri_responses',
    'things_eeg_2_test_images')
os.makedirs(save_dir, exist_ok=True)
file_name = f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}.h5'
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('tfmri', data=tfmri_test, dtype=np.float32)