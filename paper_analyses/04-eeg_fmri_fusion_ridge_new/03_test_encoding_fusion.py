"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 200 THINGS EEG2 test images. These t-fMRI responses
are then correlated with the in silico fMRI responses for the same test images,
resulting in one encoding accuracy score for each fMRI vertex and EEG time
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
from sklearn.linear_model import Ridge

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses')
file_name = f'things_eeg_2_test_sub-{args.fmri_subject:02d}_{args.hemisphere}.h5'
fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
idx_v = np.zeros(fmri_test.shape[1], dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]
fmri_test = fmri_test[:,idx_v]

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + eps)


# =============================================================================
# Load and append the in vivo EEG test responses across subjects
# =============================================================================
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
    eeg_test_sub = np.mean(h5py.File(eeg_dir_test, 'r')['eeg'][:],
        1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_test = eeg_test_sub
    else:
        eeg_test = np.append(eeg_test, eeg_test_sub, 1)
    del eeg_test_sub

# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


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
for es, esub in enumerate(tqdm(args.eeg_subjects)):
    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=esub
    )
    # Generate and store the in silico EEG responses
    if es == 0:
        eeg_test_insilico = berg.encode(model, images)
    else:
        eeg_test_insilico = np.append(eeg_test_insilico,
            berg.encode(model, images), 2)
    # Delete unused variables
    torch.cuda.empty_cache()
    gc.collect()
    del model


# =============================================================================
# Test the encoding fusion models
# =============================================================================
# Empty correlation array of shape:
# (N fMRI vertices, 140 EEG time points)
corr_invivoeeg = np.zeros((fmri_test.shape[1], len(times)), dtype=np.float32)
corr_insilicoeeg_avg_tfmri_avg = np.zeros((fmri_test.shape[1], len(times)),
    dtype=np.float32)
corr_insilicoeeg_sing_tfmri_avg = np.zeros((fmri_test.shape[1], len(times)),
    dtype=np.float32)
corr_insilicoeeg_sing_tfmri_sing = np.zeros((fmri_test.shape[1], len(times)),
    dtype=np.float32)

# Loop across EEG time points
for t in tqdm(range(len(times))):

    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
        f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
        'encoding_fusion_weights', file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = Ridge()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses for the test images with in vivo EEG
    tfmri_invivoeeg = reg.predict(eeg_test[:,:,t])

    # Generate the t-fMRI responses for the test images with average repeat in
    # silico EEG
    tfmri_insilicoeeg_avg = reg.predict(np.mean(eeg_test_insilico[:,:,:,t], 1))

    # Generate the t-fMRI responses for the test images with single repeat in
    # silico EEG
    tfmri_insilicoeeg_sing = np.zeros((eeg_test_insilico.shape[0],
        fmri_test.shape[1], eeg_test_insilico.shape[1]), dtype=np.float32)
    for r in range(eeg_test_insilico.shape[1]):
        tfmri_insilicoeeg_sing[:,:,r] = reg.predict(
            eeg_test_insilico[:,r,:,t])

    # Center and normalize the t-fMRI responses
    tfmri_invivoeeg_z = (tfmri_invivoeeg - tfmri_invivoeeg.mean(0)) /  (tfmri_invivoeeg.std(0) + eps)
    tfmri_insilicoeeg_avg_z = (tfmri_insilicoeeg_avg - tfmri_insilicoeeg_avg.mean(0)) /  (tfmri_insilicoeeg_avg.std(0) + eps)
    tfmri_insilicoeeg_sing_z = (tfmri_insilicoeeg_sing - tfmri_insilicoeeg_sing.mean(0)) /  (tfmri_insilicoeeg_sing.std(0) + eps)

    # Correlate the t-fMRI test responses with the fMRI test responses
    # corr_invivoeeg
    corr_invivoeeg[:,t] = np.diag(tfmri_invivoeeg_z.T @ fmri_test_z) / len(tfmri_invivoeeg_z)
    # corr_insilicoeeg_avg_tfmri_avg
    corr_insilicoeeg_avg_tfmri_avg[:,t] = np.diag(tfmri_insilicoeeg_avg_z.T @ fmri_test_z) / len(tfmri_insilicoeeg_avg_z)
    # corr_insilicoeeg_sing_tfmri_avg
    corr_insilicoeeg_sing_tfmri_avg[:,t] = np.diag(np.mean(tfmri_insilicoeeg_sing_z, 2).T @ fmri_test_z) / len(tfmri_insilicoeeg_sing_z)
    # corr_insilicoeeg_sing_tfmri_sing
    corr = []
    for r in range(eeg_test_insilico.shape[1]):
        corr.append(np.diag(tfmri_insilicoeeg_sing_z[:,:,r].T @ fmri_test_z) / len(tfmri_insilicoeeg_sing_z))
    corr_insilicoeeg_sing_tfmri_sing[:,t] = np.mean(corr, 0)
    del corr

    # Delete unused variables
    del tfmri_invivoeeg, tfmri_insilicoeeg_avg, tfmri_insilicoeeg_sing, \
        tfmri_invivoeeg_z, tfmri_insilicoeeg_avg_z, tfmri_insilicoeeg_sing_z, \
        reg_param, reg
    gc.collect()


# =============================================================================
# Save the results
# =============================================================================
results = {
    'corr_invivoeeg': corr_invivoeeg,
    'corr_insilicoeeg_avg_tfmri_avg': corr_insilicoeeg_avg_tfmri_avg,
    'corr_insilicoeeg_sing_tfmri_avg': corr_insilicoeeg_sing_tfmri_avg,
    'corr_insilicoeeg_sing_tfmri_sing': corr_insilicoeeg_sing_tfmri_sing
}

# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
    'encoding_fusion_accuracy')
os.makedirs(save_dir, exist_ok=True)

# Save the correlation scores
file_name = (f'corr_fmri_sub-{args.fmri_subject:02d}'
            f'_hemi-{args.hemisphere}.npy')
np.save(os.path.join(save_dir, file_name), results)