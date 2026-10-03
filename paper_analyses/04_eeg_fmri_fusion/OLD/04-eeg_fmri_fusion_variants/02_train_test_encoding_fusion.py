"""Train the EEG-fMRI encoding fusion models by linearly mapping in vivo EEG
responses onto in silico fMRI responses for the 16,540 THINGS EEG2 train
images.

One regression model is trained for each fMRI vertex and EEG time point,
using the EEG channel responses of individual THINGS EEG2 subjects.

The trained regressions are then used to predict time-resolved fMRI (t-fMRI)
responses for the 200 THINGS EEG2 test images, and these t-fMRI responses are
correlated with the in silico fMRI responses for the same test images,
resulting in one encoidng accuracy score for each fMRI vertex and EEG time
point.

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
parser.add_argument('--eeg_reps', default='average', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train/test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico fMRI responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses')
file_name = f'things_eeg_2_train_sub-{args.fmri_subject:02d}_{args.hemisphere}.h5'
fmri_train = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']
file_name = f'things_eeg_2_test_sub-{args.fmri_subject:02d}_{args.hemisphere}.h5'
fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

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
fmri_test = fmri_test[:,idx_v]

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + eps)


# =============================================================================
# Load the in vivo EEG responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
    'invivo_eeg_responses')
file_train = (f'things_eeg_2_train_sub-{args.eeg_subject:02d}_'
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}.h5')
file_test = (f'things_eeg_2_test_sub-{args.eeg_subject:02d}_'
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}.h5')

eeg_train = h5py.File(os.path.join(data_dir, file_train), 'r')['eeg'][:]
eeg_test = h5py.File(os.path.join(data_dir, file_test), 'r')['eeg'][:]

# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=args.eeg_subject
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Train and test the encoding fusion models (average EEG repeats)
# =============================================================================
if args.eeg_reps == 'average':

    # Empty correlation array of shape:
    # (N fMRI vertices, 140 EEG time points)
    corr = np.zeros((fmri_train.shape[1], len(times)), dtype=np.float32)

    # Loop across EEG time points
    for t in tqdm(range(len(times))):

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
        tfmri_test = reg.predict(eeg_test[:,:,t])
        del reg

        # Center and normalize the t-fMRI responses
        tfmri_test_z = (tfmri_test - tfmri_test.mean(0)) /  (tfmri_test.std(0) + eps)

        # Correlate the t-fMRI test responses with the fMRI test responses
        corr[:,t] = np.diag(tfmri_test_z.T @ fmri_test_z) / len(tfmri_test_z)
        del tfmri_test, tfmri_test_z

    # Save the correlation scores
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'encoding_fusion_accuracy',
        f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
    os.makedirs(save_dir, exist_ok=True)
    file_name = (f'corr_fmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}'
        f'_eeg_sub-{args.eeg_subject:02d}.npy')
    np.save(os.path.join(save_dir, file_name), corr)


# =============================================================================
# Train and test the encoding fusion models (single EEG repeats)
# =============================================================================
if args.eeg_reps == 'single':

    # Empty correlation array of shape:
    # (N fMRI vertices, 4 EEG repeats, 140 EEG time points)
    corr = np.zeros((fmri_train.shape[1], eeg_train.shape[1], len(times)),
        dtype=np.float32)

    # Loop across EEG time points and repeats
    for t in tqdm(range(len(times))):

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
            tfmri_test = reg.predict(np.mean(eeg_test[:,:,:,t], 1))
            del reg

            # Center and normalize the t-fMRI responses
            tfmri_test_z = (tfmri_test - tfmri_test.mean(0)) /  (tfmri_test.std(0) + eps)

            # Correlate the t-fMRI test responses with the fMRI test responses
            corr[:,r,t] = np.diag(tfmri_test_z.T @ fmri_test_z) / len(tfmri_test_z)
            del tfmri_test, tfmri_test_z

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

    # Save the correlation scores
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'encoding_fusion_accuracy',
        f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
    os.makedirs(save_dir, exist_ok=True)
    file_name = (f'corr_fmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}'
        f'_eeg_sub-{args.eeg_subject:02d}.npy')
    np.save(os.path.join(save_dir, file_name), corr)