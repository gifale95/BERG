"""Train the EEG-fMRI encoding fusion models by linearly mapping in vivo EEG
responses onto in silico fMRI responses for the 16,540 THINGS EEG2 train
images.

One regression model is trained for each fMRI vertex and EEG time point,
using the EEG channel responses of individual THINGS EEG2 subjects.

The trained regressions are then used to predict time-resolved fMRI (t-fMRI)
responses for the 200 THINGS EEG2 test images.

Parameters
----------
fmri_subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_subject : int
    Subject identifier for the THINGS EEG2 subject. Valid subject identifiers
    are integers from 1 10.
eeg_reps : str
    If 'average' average the EEG responses across repeats. If 'single', use the
    single-trial EEG responses.
regression : str
    If 'linear', apply PCA to the EEG responses. If 'ridge', do not apply PCA.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subject', default=1, type=int)
parser.add_argument('--eeg_reps', default='single', type=str)
parser.add_argument('--regression', default='ridge', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico fMRI train responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses')
file_name = f'things_eeg_2_train_sub-{args.fmri_subject:02d}_{args.hemisphere}.h5'
fmri_train = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
idx_v = np.zeros(fmri_train.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]
fmri_train = fmri_train[:,idx_v]


# =============================================================================
# Load the in vivo EEG responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
    'invivo_eeg_responses')
file_train = (f'things_eeg_2_train_sub-{args.eeg_subject:02d}_'
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}.h5')
file_test = (f'things_eeg_2_test_sub-{args.eeg_subject:02d}_'
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}.h5')

eeg_train = h5py.File(os.path.join(data_dir, file_train), 'r')['eeg']
eeg_test = h5py.File(os.path.join(data_dir, file_test), 'r')['eeg']


# =============================================================================
# Train the encoding fusion models, and save the learned weights
# (average EEG repeats)
# =============================================================================
if args.eeg_reps == 'average':

    # Empty t-fMRI response array of shape:
    # (200 test images, N fMRI vertices, 140 EEG time points)
    tfmri_test = np.zeros((len(eeg_test), fmri_train.shape[1], eeg_test.shape[2]),
        dtype=np.float32)

    # Loop across EEG time points
    for t in tqdm(range(eeg_train.shape[2])):

        if args.regression == 'linear':

            # Train the encoding fusion models
            reg = LinearRegression()
            reg.fit(eeg_train[:,:,t], fmri_train)

            # Store the encoding fusion model weights
            reg_param = {
                'coef_': reg.coef_.astype(np.float32),
                'intercept_': reg.intercept_.astype(np.float32),
                'n_features_in_': reg.n_features_in_
            }

        elif args.regression == 'ridge':

            # Train the encoding fusion models
            alphas = np.logspace(-6, 6, 13)
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
        save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
            'encoding_fusion_weights',
            f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
        os.makedirs(save_dir, exist_ok=True)
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                    f'hemi-{args.hemisphere}_eeg_sub-{args.eeg_subject:02d}'
                    f'_eeg_time-{t:03d}.npy')
        np.save(os.path.join(save_dir, file_name), reg_param)
        del reg_param

        # Predict the t-fMRI responses for the test images
        tfmri_test[:,:,t] = reg.predict(eeg_test[:,:,t])
        del reg

    # Save the t-fMRI responses for the test images
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'tfmri_responses', 'things_eeg_2_test_images',
        f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
    os.makedirs(save_dir, exist_ok=True)
    file_name = (f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}'
        f'_eeg_sub-{args.eeg_subject:02d}.h5')
    with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
        f.create_dataset('tfmri', data=tfmri_test, dtype=np.float32)


# =============================================================================
# Train the encoding fusion models, and save the learned weights
# (single EEG repeats)
# =============================================================================
if args.eeg_reps == 'single':

    # Empty t-fMRI response array of shape:
    # (200 test images, N fMRI vertices, 4 EEG repeats, 140 EEG time points)
    tfmri_test = np.zeros((len(eeg_test), fmri_train.shape[1], eeg_train.shape[1],
        eeg_test.shape[3]), dtype=np.float32)

    # Loop across EEG time points and repeats
    for t in tqdm(range(eeg_train.shape[3])):

        reg_param = []

        for r in tqdm(range(eeg_train.shape[1])):

            if args.regression == 'linear':

                # Train the encoding fusion models
                reg = LinearRegression()
                reg.fit(eeg_train[:,r,:,t], fmri_train)

                # Store the encoding fusion model weights
                reg_param_r = {
                    'coef_': reg.coef_.astype(np.float32),
                    'intercept_': reg.intercept_.astype(np.float32),
                    'n_features_in_': reg.n_features_in_
                }
                reg_param.append(reg_param_r)
                del reg_param_r

            elif args.regression == 'ridge':

                # Train the encoding fusion models
                alphas = np.logspace(-6, 6, 13)
                reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
                reg.fit(eeg_train[:,r,:,t], fmri_train)

                # Store the encoding fusion model weights
                reg_param_r = {
                    'coef_': reg.coef_.astype(np.float32),
                    'intercept_': reg.intercept_.astype(np.float32),
                    'alpha_': reg.alpha_,
                    'n_features_in_': reg.n_features_in_
                }
                reg_param.append(reg_param_r)
                del reg_param_r

            # Predict the t-fMRI responses for the test images
            tfmri_test[:,:,r,t] = reg.predict(np.mean(eeg_test[:,:,:,t], 1))
            del reg

        # Save the encoding fusion model weights
        save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
            'encoding_fusion_weights',
            f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
        os.makedirs(save_dir, exist_ok=True)
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                    f'hemi-{args.hemisphere}_eeg_sub-{args.eeg_subject:02d}'
                    f'_eeg_time-{t:03d}.npy')
        np.save(os.path.join(save_dir, file_name), reg_param)
        del reg_param

    # Save the t-fMRI responses for the test images
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'tfmri_responses', 'things_eeg_2_test_images',
        f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
    os.makedirs(save_dir, exist_ok=True)
    file_name = (f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}'
        f'_eeg_sub-{args.eeg_subject:02d}.h5')
    with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
        f.create_dataset('tfmri', data=tfmri_test, dtype=np.float32)