"""Check for alpha desynchronization of in silico EEG responses after stimulus
onset.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : list
    List containing the EEG channel type(s) retained for the analyses.
    Possible values are: 'O' (occipital), 'P' (posterior), 'T' (temporal),
    'C' (central), 'F' (frontal). Alternatively, the list can also contain the
    names of the individual channels used.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
import mne
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default=['O', 'P'], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Alpha desynchronization - Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in silico EEG responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'alpha_desynchronization', 'insilico_eeg_responses',
    'insilico_eeg_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

eeg = data['insilico_eeg']
metadata = data['metadata']
del data


# =============================================================================
# EEG channel selection
# =============================================================================
# Kept channel indices
ch_names = metadata[0]['eeg']['ch_names']
kept_chan_idx = []
kept_chan_names = []
for c, chan in enumerate(ch_names):
    for ch_select in args.channels:
        if ch_select in chan:
            kept_chan_idx.append(c)
            kept_chan_names.append(chan)
            break

# Channel selection
eeg = eeg[:,:,kept_chan_idx]


# =============================================================================
# Convert the EEG responses into time-frequency space
# =============================================================================
# Define the temporal frequency of EEG responses
sfreq = 200 # https://brain-encoding-response-generator.readthedocs.io/en/latest/models/model_cards/eeg-things_eeg_2-vit_b_32.html
freqs = np.linspace(2, 40, 30)
n_cycles = np.minimum(freqs/2, 10) # !!! np.minimum(freqs / 4, 8)
n_cycles[n_cycles>5] = 5


# Loop across subjects
for s in range(len(args.subjects)):

    # Convert the data to an MNE object
    n_images, n_channels, n_times = eeg[s].shape
    ch_names = kept_chan_names
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types) # type: ignore
    epochs = mne.EpochsArray(eeg[s], info)

    # EEG time-frequency decomposition # !!!
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs,
        n_cycles,
        average=True, # Average the TFRs across epochs
        return_itc=False
        )


    tfr = power.data  # (1000, 64, 30, 500) # !!!





# =============================================================================
# Bootstrap the ERP confidence intervals (CIs)
# =============================================================================
ci_erp_faces = np.zeros((2, erp_faces.shape[1]))
ci_erp_objects = np.zeros((ci_erp_faces.shape))

faces_dist = np.zeros((args.n_iter, erp_faces.shape[1]))
objects_dist = np.zeros((faces_dist.shape))

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    faces_dist[i] = np.mean(erp_faces[idx], 0)
    objects_dist[i] = np.mean(erp_objects[idx], 0)

ci_erp_faces[0] = np.percentile(faces_dist, 2.5, axis=0)
ci_erp_faces[1] = np.percentile(faces_dist, 97.5, axis=0)
ci_erp_objects[0] = np.percentile(objects_dist, 2.5, axis=0)
ci_erp_objects[1] = np.percentile(objects_dist, 97.5, axis=0)


# =============================================================================
# Statistical significance
# =============================================================================
# Compute the difference between face and object absolute ERPs
erp_diff = erp_faces - erp_objects

# Compute the p-value
pval_erp_diff = ttest_rel(erp_faces, erp_objects, axis=0,
    alternative='less')[1]

# Multiple comparison correction
sig_erp_diff, pval_erp_diff_corrected, _, _ = multipletests(pval_erp_diff,
    0.05, 'fdr_bh')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'erp_faces': erp_faces,
    'erp_objects': erp_objects,
    'ci_erp_faces': ci_erp_faces,
    'ci_erp_objects': ci_erp_objects,
    'pval_erp_diff': pval_erp_diff,
    'pval_erp_diff_corrected': pval_erp_diff_corrected,
    'sig_erp_diff': sig_erp_diff,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'n170_faces', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_' + 'channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore