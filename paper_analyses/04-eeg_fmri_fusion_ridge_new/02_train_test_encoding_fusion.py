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
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 10.
eeg_test_reps : str
    If 'average' average the in silico test EEG responses across repeats. If
    'single', use the single-trial in silico test EEG responses.
tfmri_test_reps : str
    If 'average' average the t-fMRI responses across repeats. If 'single', use
    the single-trial test t-fMRI responses.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
from PIL import Image
import gc
import torch
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_test_reps', default='average', type=str)
parser.add_argument('--tfmri_test_reps', default='average', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train/test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directories
# =============================================================================
# Encoding fusion model weight save directory
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
    'encoding_fusion_weights', f'eeg_reps-{args.eeg_reps}')
os.makedirs(save_dir_weights, exist_ok=True)

# Encoding fusion model accuracy save directory
save_dir_accuracy = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
    'encoding_fusion_accuracy', f'eeg_reps-{args.eeg_reps}')
os.makedirs(save_dir_accuracy, exist_ok=True)


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
fmri_test = fmri_test[:,idx_v]

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + eps)


# =============================================================================
# Load and append the in vivo EEG responses across subjects
# =============================================================================
# Loop across subjects
for s, sub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, while averagin the train responses across repeats
    eeg_dir_train = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{sub:02d}_split-train.h5')
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{sub:02d}_split-test.h5')
    eeg_train_sub = np.mean(h5py.File(eeg_dir_train, 'r')['eeg'][:],
        1).astype(np.float32)
    eeg_test_sub = h5py.File(eeg_dir_test, 'r')['eeg'][:].astype(np.float32)

    # Append the EEG channel responses across subjects
    if s == 0:
        eeg_train = eeg_train_sub
        eeg_test = eeg_test_sub
    else:
        eeg_train = np.append(eeg_train, eeg_train_sub, 1)
        eeg_test = np.append(eeg_test, eeg_test_sub, 2)
    del eeg_train_sub, eeg_test_sub

# Average the responses across repeats if specified
if args.eeg_test_reps == 'average':
    eeg_test = np.mean(eeg_test, 1)

# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']

# DELETE # !!!
time_range = np.arange(20, 50)
times = times[time_range]
eeg_test = eeg_test[:,:,:,time_range]
eeg_train = eeg_train[:,:,time_range]
# DELETE # !!!


# =============================================================================
# Generate the in silico EEG responses for the test images, and append them
# across subjects
# =============================================================================
# Load the test images
test_img_files = metadata_eeg['encoding_models']['test_img_info']\
    ['test_img_files']
# Loop across test images
images = []
for file in tqdm(test_img_files):
    # Find correct subfolder
    img_path = None
    for root, _, files in os.walk(os.path.join(args.things_dir)):
        if file in files:
            img_path = os.path.join(root, file)
            break
    # Load and transform the image
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
    images.append(img)
# Format the images to a numpy array
images = np.array(images)

# Loop across EEG subjects
for s, esub in enumerate(tqdm(args.eeg_subjects)):
    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=esub
    )
    # Generate and store the in silico EEG responses
    if s == 0:
        eeg_test_insilico = berg.encode(model, images)
    else:
        eeg_test_insilico = np.append(eeg_test_insilico,
            berg.encode(model, images), 2)
    # Delete unused variables
    torch.cuda.empty_cache()
    gc.collect()
    del model

# DELETE # !!!
eeg_test_insilico = eeg_test_insilico[:,:,:,time_range]
# DELETE # !!!

# Average the responses across repeats if specified
if args.eeg_test_reps == 'average':
    eeg_test_insilico = np.mean(eeg_test_insilico, 1)


# =============================================================================
# Train and test the encoding fusion models # !!!
# =============================================================================
if args.eeg_reps == 'average':

    # Empty correlation array of shape:
    # (N fMRI vertices, 140 EEG time points)
    corr = np.zeros((fmri_train.shape[1], len(times)), dtype=np.float32)

    # Loop across EEG time points
    for t in tqdm(range(len(times))):

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
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                    f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
        np.save(os.path.join(save_dir_weights, file_name), reg_param)
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
    file_name = (f'corr_fmri_sub-{args.fmri_subject:02d}'
                f'_hemi-{args.hemisphere}.npy')
    np.save(os.path.join(save_dir_accuracy, file_name), corr)