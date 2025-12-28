"""Prepare the in vivo EEG responses used to train and test the fusion encoding
models.

The preparation includes appending the EEG channels responses across subjects,
and transfomring them with PCA independently for each EEG time point.

Parameters
----------
eeg_subject : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import numpy as np
import h5py
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subjects_all', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Prepare in vivo EEG <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# Edit code such that it generates two instances of EEG responses: # !!!
#     1) Averaged across repeats (as done now)
#     2) Single repeats (PCA applied on single repeats; for the decoding analyses of t-fMRI data)


# =============================================================================
# Load and append the EEG responses across subjects
# =============================================================================
# Loop across subjects
for s, sub in enumerate(tqdm(args.eeg_subjects_all)):

    # Load the EEG responses, and average them across repeats
    eeg_dir_train = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{sub:02d}_split-train.h5')
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{sub:02d}_split-test.h5')
    eeg_train_sub = np.mean(h5py.File(
        eeg_dir_train, 'r')['eeg'], 1).astype(np.float32)
    eeg_test_sub = np.mean(h5py.File(
        eeg_dir_test, 'r')['eeg'], 1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if s == 0:
        eeg_train = eeg_train_sub
        eeg_test = eeg_test_sub
    else:
        eeg_train = np.append(eeg_train, eeg_train_sub, 1)
        eeg_test = np.append(eeg_test, eeg_test_sub, 1)
    del eeg_train_sub, eeg_test_sub


# =============================================================================
# Z-score the EEG responses and transform them with PCA
# =============================================================================
# Empty result variables
scaler_param = []
pca_param = []
eeg_train_pca = np.zeros((eeg_train.shape), dtype=np.float32)
eeg_test_pca = np.zeros((eeg_test.shape), dtype=np.float32)

# Loop across EEG time points
for t in tqdm(range(eeg_train.shape[2])):

    # Z-score the EEG responses, and store the parameters
    scaler = StandardScaler()
    scaler.fit(eeg_train[:,:,t])
    eeg_train_zscore = scaler.transform(eeg_train[:,:,t])
    eeg_test_zscore = scaler.transform(eeg_test[:,:,t])
    scaler_param_t = {
        'scale_': scaler.scale_.astype(np.float32),
        'mean_': scaler.mean_.astype(np.float32),
        'var_': scaler.var_.astype(np.float32),
        'n_features_in_': scaler.n_features_in_,
        'n_samples_seen_': scaler.n_samples_seen_
    }
    scaler_param.append(scaler_param_t)

    # Transform the EEG responses with PCA, and store the parameters
    pca = PCA(n_components=eeg_train_zscore.shape[1], random_state=20200220) 
    pca.fit(eeg_train_zscore)
    eeg_train_pca[:,:,t] = pca.transform(eeg_train_zscore)
    eeg_test_pca[:,:,t] = pca.transform(eeg_test_zscore)
    pca_param_t = {
        'components_': pca.components_.astype(np.float32),
        'explained_variance_': pca.explained_variance_.astype(np.float32),
        'explained_variance_ratio_': pca.explained_variance_ratio_.astype(np.float32),
        'singular_values_': pca.singular_values_.astype(np.float32),
        'mean_': pca.mean_.astype(np.float32),
        'n_components_': pca.n_components_,
        'n_samples_': pca.n_samples_,
        'noise_variance_': pca.noise_variance_ if pca.noise_variance_ is None else np.float32(pca.noise_variance_),
        'n_features_in_': pca.n_features_in_
    }
    pca_param.append(pca_param_t)

    # Delete unused variables
    del scaler_param_t, pca_param_t, eeg_train_zscore, eeg_test_zscore


# =============================================================================
# Save the transformed in vivo EEG responses, and the transformation parameters
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_eeg_responses')
os.makedirs(save_dir, exist_ok=True)

file_name_train = 'things_eeg_2_train.h5'
file_name_test = 'things_eeg_2_test.h5'

with h5py.File(os.path.join(save_dir, file_name_train), 'w') as f:
    f.create_dataset('eeg', data=eeg_train_pca, dtype=np.float32)
with h5py.File(os.path.join(save_dir, file_name_test), 'w') as f:
    f.create_dataset('eeg', data=eeg_test_pca, dtype=np.float32)

np.save(os.path.join(save_dir, 'scaler_param.npy'), scaler_param)
np.save(os.path.join(save_dir, 'pca_param.npy'), pca_param)