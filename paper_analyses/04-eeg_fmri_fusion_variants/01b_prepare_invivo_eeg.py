"""Prepare the in vivo EEG responses used to train and test the fusion encoding
models.

The preparation includes appending the EEG channels responses across subjects,
and transfomring them with PCA independently for each EEG time point.

Parameters
----------
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
import numpy as np
import h5py
import os
from berg import BERG
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subject', default=1, type=int)
parser.add_argument('--eeg_reps', default='single', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Prepare in vivo EEG <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the saving directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
    'invivo_eeg_responses')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the EEG responses
# =============================================================================
# Load the EEG responses
eeg_dir_train = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2',
    f'eeg_sub-{args.eeg_subject:02d}_split-train.h5')
eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2',
    f'eeg_sub-{args.eeg_subject:02d}_split-test.h5')
eeg_train = h5py.File(eeg_dir_train, 'r')['eeg'][:].astype(np.float32)
eeg_test = h5py.File(eeg_dir_test, 'r')['eeg'][:].astype(np.float32)

# Average the EEG responses across repeats
if args.eeg_reps == 'average':
    eeg_train = np.mean(eeg_train, 1)
    eeg_test = np.mean(eeg_test, 1)

# Load the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=args.eeg_subject
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Z-score the EEG responses and transform them with PCA
# =============================================================================
if args.regression == 'linear':

    # Empty result variables
    scaler_param = []
    pca_param = []
    eeg_train_pca = np.zeros((eeg_train.shape), dtype=np.float32)
    eeg_test_pca = np.zeros((eeg_test.shape), dtype=np.float32)

    # Loop across EEG time points
    for t in tqdm(range(len(times))):

        # Trial-average EEG responses
        if args.eeg_reps == 'average':

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
                'noise_variance_': pca.noise_variance_,
                'n_features_in_': pca.n_features_in_
            }
            pca_param.append(pca_param_t)

            # Delete unused variables
            del scaler_param_t, pca_param_t, eeg_train_zscore, eeg_test_zscore

        # Single trial EEG responses
        if args.eeg_reps == 'single':

            scaler_param_rep = []
            pca_param_rep = []

            for r in range(eeg_train.shape[1]):

                # Z-score the EEG responses, and store the parameters
                scaler = StandardScaler()
                scaler.fit(eeg_train[:,r,:,t])
                eeg_train_zscore = scaler.transform(eeg_train[:,r,:,t])
                eeg_test_zscore = scaler.transform(eeg_test[:,r,:,t])
                scaler_param_t = {
                    'scale_': scaler.scale_.astype(np.float32),
                    'mean_': scaler.mean_.astype(np.float32),
                    'var_': scaler.var_.astype(np.float32),
                    'n_features_in_': scaler.n_features_in_,
                    'n_samples_seen_': scaler.n_samples_seen_
                }
                scaler_param_rep.append(scaler_param_t)

                # Transform the EEG responses with PCA, and store the parameters
                pca = PCA(n_components=eeg_train_zscore.shape[1], random_state=20200220) 
                pca.fit(eeg_train_zscore)
                eeg_train_pca[:,r,:,t] = pca.transform(eeg_train_zscore)
                eeg_test_pca[:,r,:,t] = pca.transform(eeg_test_zscore)
                pca_param_t = {
                    'components_': pca.components_.astype(np.float32),
                    'explained_variance_': pca.explained_variance_.astype(np.float32),
                    'explained_variance_ratio_': pca.explained_variance_ratio_.astype(np.float32),
                    'singular_values_': pca.singular_values_.astype(np.float32),
                    'mean_': pca.mean_.astype(np.float32),
                    'n_components_': pca.n_components_,
                    'n_samples_': pca.n_samples_,
                    'noise_variance_': pca.noise_variance_,
                    'n_features_in_': pca.n_features_in_
                }
                pca_param_rep.append(pca_param_t)

                # Delete unused variables
                del scaler_param_t, pca_param_t, eeg_train_zscore, eeg_test_zscore

            # Store the parameters for all repeats
            scaler_param.append(scaler_param_rep)
            pca_param.append(pca_param_rep)

            # Delete unused variables
            del scaler_param_rep, pca_param_rep

    # Save the PCA parameters
    file_name_scaler = (f'scaler_param_sub-{args.eeg_subject:02d}_'
        f'eeg_reps-{args.eeg_reps}.npy')
    file_name_pca = (f'pca_param_sub-{args.eeg_subject:02d}_'
        f'eeg_reps-{args.eeg_reps}.npy')
    np.save(os.path.join(save_dir, file_name_scaler), scaler_param)
    np.save(os.path.join(save_dir, file_name_pca), pca_param)


# =============================================================================
# Save the in vivo EEG responses
# =============================================================================
file_name_train = (f'things_eeg_2_train_sub-{args.eeg_subject:02d}_'
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}.h5')
file_name_test = (f'things_eeg_2_test_sub-{args.eeg_subject:02d}_'
    f'eeg_reps-{args.eeg_reps}_regression-{args.regression}.h5')

if args.regression == 'linear':

    with h5py.File(os.path.join(save_dir, file_name_train), 'w') as f:
        f.create_dataset('eeg', data=eeg_train_pca, dtype=np.float32)
    with h5py.File(os.path.join(save_dir, file_name_test), 'w') as f:
        f.create_dataset('eeg', data=eeg_test_pca, dtype=np.float32)

elif args.regression == 'ridge':

    with h5py.File(os.path.join(save_dir, file_name_train), 'w') as f:
        f.create_dataset('eeg', data=eeg_train, dtype=np.float32)
    with h5py.File(os.path.join(save_dir, file_name_test), 'w') as f:
        f.create_dataset('eeg', data=eeg_test, dtype=np.float32)