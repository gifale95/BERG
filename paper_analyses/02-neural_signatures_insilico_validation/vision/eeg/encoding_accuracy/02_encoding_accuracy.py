"""Get the encoding accuracy of BERG's EEG encoding models trained on the
THINGS EEG2 dataset.

This code additionally compares the noise of the in silico EEG responses (i.e.,
the EEG responses generated from encoding models) with the noise of the in vivo
(i.e., target) responses from the THINGS EEG2 experiment, by comparing how much
variance can these two data types explain for a third, independent split of 
THINGS EEG2 responses.

Because the in silico neural responses did not capture all signal variance for
the in vivo THINGS EEG2 responses, the in silico neural responses explaining
more variance than THINGS EEG2's in vivo responses would be indicative of the
former being less affected by noise.

# !!!
The comparison is carried out through three predictions, using the in silico
and in vivo fMRI responses for the 515 test images. Each prediction involves
explaining single NSD in vivo response trials with a different predictor.
The first predictor consists of the two remaining NSD in vivo response trials,
each used independently. The evaluation is repeated until each of the three
trials is used as the target to be explained and the remaining two trials as
separate predictors, and the explained variance scores from the different
evaluations (N = 6 evaluations) are then averaged.
The second predictor consists of the average of the two remaining NSD in vivo
response trials. The evaluation is repeated until each of the three trials is
used as the target to be explained and the average of the remaining two trials
as predictor, and the explained variance scores from the different evaluations
(N = 3 evaluations) are then averaged.
The third predictor consists of the in silico responses from the trained
encoding models. The evaluation is repeated until each of the three trials is
used as the target to be explained by the same in silico responses, and the
explained variance scores from the different evaluations (N = 3 evaluations) is
then averaged.
These comparisons are carried out independently for each vertex and subject.
# !!!

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list of int
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
import h5py
from scipy.stats import pearsonr
import random
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Encoding accuracy and noise analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Loop across subjects
# =============================================================================
# Empty result variables
correlation = []
noise_ceiling = []
metadata_berg = []
corr_iv_iv = []
corr_iv_is = []

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):


# =============================================================================
# Get the encoding accuracy of the in silico EEG responses
# =============================================================================
    # Get the metadata
    metadata = berg.get_model_metadata(
        args.encoding_model,
        subject=sub
    )

    # Store the metadata
    metadata_berg.append(metadata)

    # Extract the encoding accuracy and noise ceiling
    correlation.append(metadata['encoding_models']\
        ['correlation_averaged_repetitions'])
    noise_ceiling.append(metadata['eeg']['noise_ceiling'])


# =============================================================================
# Load the in silico EEG responses
# =============================================================================
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation',
        'vision', 'eeg', 'encoding_accuracy', 'insilico_eeg_responses',
        args.encoding_model, 'insilico_eeg_responses_sub-'+
        format(sub, '02')+'.npy')

    data = np.load(data_dir, allow_pickle=True).item()
    eeg_insilico = data['eeg'].astype(np.float32)

    # Average the EEG responses across repeats
    eeg_insilico = np.mean(eeg_insilico, 1)


# =============================================================================
# Load the in vivo EEG responses
# =============================================================================
# The in vivo fMRI responses were prepared using this code:
# https://github.com/gifale95/BERG/blob/main/berg_creation_code/01_prepare_data/train_dataset-things_eeg_2

    # Data directories
    data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2')
    eeg_dir = os.path.join(data_dir, f'eeg_sub-{sub:02d}_split-test.h5')

    # Access the data
    eeg_invivo = h5py.File(eeg_dir, 'r')['eeg'][:].astype(np.float32)


# =============================================================================
# Correlate in silico and in vivo target EEG for the noise analysis
# =============================================================================
    # Correlate in vivo target EEG pseudo-trials with other in vivo
    # pseudo-trials, and with in silico EEG responses
    # iv => in vivo
    # is => in silico
    iter = 100
    tot_reps = eeg_invivo.shape[1]
    target_reps = 20
    probe_reps = tot_reps - target_reps
    n_chans = eeg_invivo.shape[2]
    n_times = eeg_invivo.shape[3]

    corr_iv_iv_sub = np.zeros((iter, probe_reps, n_chans, n_times))
    corr_iv_is_sub = np.zeros((iter, n_chans, n_times))

    # Loop across analysis iterations
    for i in tqdm(range(iter)): # !!! Vectorize correlations and Remove tqdm # !!!

        # Randomly select the target repetitions and the probe repetitions
        target_idx = np.random.choice(tot_reps, target_reps, replace=False)
        probe_idx = np.setdiff1d(np.arange(tot_reps), target_idx)

        # Correlate in vivo target EEG pseudo-trials with other in vivo
        # pseudo-trials, and with in silico EEG responses
        for c in range(n_chans):
            for t in range(n_times):
                corr_iv_is_sub[i,c,t] = pearsonr(
                    eeg_invivo[:,target_idx,c,t].mean(1),
                    eeg_insilico[:,c,t])[0]
                for p in range(probe_reps):
                    corr_iv_iv_sub[i,p,c,t] = pearsonr(
                        eeg_invivo[:,target_idx,c,t].mean(1),
                        eeg_invivo[:,probe_idx[:p+1],c,t].mean(1))[0]

    # Average the correlations across iterations
    corr_iv_iv_sub = np.mean(corr_iv_iv_sub, 0)
    corr_iv_is_sub = np.mean(corr_iv_is_sub, 0)

    # Store the correlations
    corr_iv_iv.append(corr_iv_iv_sub)
    corr_iv_is.append(corr_iv_is_sub)
    del eeg_insilico, eeg_invivo, corr_iv_is_sub, corr_iv_iv_sub

# Format to numpy arrays
corr_iv_iv = np.array(corr_iv_iv)
corr_iv_is = np.array(corr_iv_is)


# =============================================================================
# Compute the confidence intervals
# =============================================================================
ci_corr_iv_is = np.zeros((2, n_chans, n_times))
ci_corr_iv_iv = np.zeros((2, target_reps, n_chans, n_times))
dist_corr_iv_is = np.zeros((args.n_iter, n_chans, n_times))
dist_corr_iv_iv = np.zeros((args.n_iter, target_reps, n_chans, n_times))

for i in tqdm(range(args.n_iter)):

    idx = resample(np.arange(len(args.subjects)))
    dist_corr_iv_is[i] = np.mean(corr_iv_is[idx], axis=0)
    dist_corr_iv_iv[i] = np.mean(corr_iv_iv[idx], axis=0)

ci_corr_iv_is[0] = np.percentile(dist_corr_iv_is, 2.5, axis=0)
ci_corr_iv_is[1] = np.percentile(dist_corr_iv_is, 97.5, axis=0)
ci_corr_iv_iv[0] = np.percentile(dist_corr_iv_iv, 2.5, axis=0)
ci_corr_iv_iv[1] = np.percentile(dist_corr_iv_iv, 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'correlation': correlation,
    'noise_ceiling': noise_ceiling,
    'metadata': metadata_berg,
    'corr_iv_iv': corr_iv_iv,
    'corr_iv_is': corr_iv_is,
    'ci_corr_iv_iv': ci_corr_iv_iv,
    'ci_corr_iv_is': ci_corr_iv_is
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'encoding_accuracy', 'encoding_accuracy',
    args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)