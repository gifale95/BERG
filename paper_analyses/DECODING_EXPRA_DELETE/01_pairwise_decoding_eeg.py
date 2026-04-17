"""Pairwise decoding of THINGS EEG2 test data with different amounts of image
conditions, repetitions, and with or without cross-validation.

Parameters
----------
sub : int
    Used subject.
n_conditions : int
    Number of image conditions.
n_repeats : int
    Number of repetitions.
cv : int
    Whether to use cross-validation (1) or not (0).
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--n_conditions', default=200, type=int)
parser.add_argument('--n_repeats', default=80, type=int)
parser.add_argument('--cv', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/decoding_expra', type=str)
parser.add_argument('--things_eeg_2_dir', default='/scratch/giffordale95/datasets/things_eeg_2', type=str)
args, unknown = parser.parse_known_args()

print('>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the EEG data
# =============================================================================
data_dir = os.path.join(args.things_eeg_2_dir, 'preprocessed_data', 'sub-'+
    format(args.sub,'02'), 'preprocessed_eeg_test.npy')
data = np.load(data_dir, allow_pickle=True).item()
eeg = data['preprocessed_eeg_data'].astype(np.float32)
times = data['times']
ch_names = data['ch_names']
del data


# =============================================================================
# Randomly select the specified number of image conditions and repetitions
# =============================================================================
idx_rep = np.arange(0, eeg.shape[1])
idx_rep = resample(idx_rep, replace=False, n_samples=args.n_repeats)
idx_cond = np.arange(0, eeg.shape[0])
idx_cond = resample(idx_cond, replace=False, n_samples=args.n_conditions)
eeg = eeg[idx_cond][:,idx_rep]


# =============================================================================
# Pairwise decoding
# =============================================================================
# Decoding array of shape:
# (Image conditions × Image conditions × EEG time points)
decoding = np.zeros((eeg.shape[0], eeg.shape[0], eeg.shape[3]),
    dtype=np.float32)

# Loop over EEG time points, image-conditions and EEG repetitions
for t in tqdm(range(eeg.shape[3])):
    for i1 in range(len(eeg)):
        for i2 in range(i1):

            # Select the image condition data
            eeg_cond_1 = eeg[i1,:,:,t]
            eeg_cond_2 = eeg[i2,:,:,t]

            # Create pseudo-trials
            n_ptrials_repeats = 2
            n_pseudo_trials = int(
                np.ceil(len(eeg_cond_1) / n_ptrials_repeats))
            pseudo_data_1 = np.zeros((n_pseudo_trials,
                eeg_cond_1.shape[1]))
            pseudo_data_2 = np.zeros((n_pseudo_trials,
                eeg_cond_2.shape[1]))
            for r in range(n_pseudo_trials):
                idx_start = r * n_ptrials_repeats
                idx_end = idx_start + n_ptrials_repeats
                pseudo_data_1[r] = np.mean(eeg_cond_1[idx_start:idx_end],
                    0)
                pseudo_data_2[r] = np.mean(eeg_cond_2[idx_start:idx_end],
                    0)
            eeg_cond_1 = pseudo_data_1
            eeg_cond_2 = pseudo_data_2

            # Train/test the classifier (cross-validation)
            if args.cv == 1:
                # SVM target vectors
                y_train = np.zeros(((len(eeg_cond_1)-1)*2))
                y_train[int(len(y_train)/2):] = 1
                y_test = np.asarray((0, 1))
                scores = np.zeros(len(eeg_cond_1))
                for r in range(len(eeg_cond_1)):
                    # Define the training/test partitions
                    X_train = np.append(np.delete(eeg_cond_1, r, 0),
                        np.delete(eeg_cond_2, r, 0), 0)
                    X_test = np.append(np.expand_dims(eeg_cond_1[r], 0),
                        np.expand_dims(eeg_cond_2[r], 0), 0)
                    # Train the classifier
                    dec_svm = SVC(kernel='linear')
                    dec_svm.fit(X_train, y_train)
                    # Test the classifier
                    y_pred = dec_svm.predict(X_test)
                    scores[r] = sum(y_pred == y_test) / len(y_test)

            # Train/test the classifier (no cross-validation)
            elif args.cv == 0:
                # SVM target vectors
                y_train = np.zeros(((len(eeg_cond_1))*2))
                y_train[int(len(y_train)/2):] = 1
                y_test = y_train
                # Define the training/test partitions
                X_train = np.append(eeg_cond_1, eeg_cond_2, 0)
                X_test = X_train
                # Train the classifier
                dec_svm = SVC(kernel='linear')
                dec_svm.fit(X_train, y_train)
                # Test the classifier
                y_pred = dec_svm.predict(X_test)
                scores = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            decoding[i1,i2,t] = np.mean(scores)
            decoding[i2,i1,t] = decoding[i1,i2,t]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'decoding': decoding,
    'times': times,
    'ch_names': ch_names
}

save_dir = os.path.join(args.project_dir, 'pairwise_decoding')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'decoding_sub-{args.sub:02d}_conditions-{args.n_conditions:03d}_'
            f'repeats-{args.n_repeats:02d}_cv-{args.cv}.npy')

np.save(os.path.join(save_dir, file_name), results)