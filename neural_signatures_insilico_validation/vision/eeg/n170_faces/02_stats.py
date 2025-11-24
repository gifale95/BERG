"""Compute stats on the results of the N170 ERP analysis. The stats consist of
bootstrapped 95% confidence intervals for the ERP, and of the calculation of
significant differences between the magnituted of the N170 component for faces
versus objects.

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
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default=['P7', 'P8', 'PO7', 'PO8', 'TP7', 'TP8'], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> EEG N170 - Stats <<<')
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
    'vision', 'eeg', 'n170_faces', 'insilico_eeg_responses',
    'insilico_eeg_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

eeg_faces = data['insilico_eeg']['Faces']
eeg_objects = data['insilico_eeg']['Objects']
metadata = data['metadata']
del data


# =============================================================================
# EEG channel selection
# =============================================================================
# Kept channel indices
ch_names = metadata[0]['eeg']['ch_names']
kept_chan_idx = []
for c, chan in enumerate(ch_names):
    for ch_select in args.channels:
        if ch_select in chan:
            kept_chan_idx.append(c)
            break

# Average the EEG responses across the chosen channels
eeg_faces = np.mean(eeg_faces[:,:,kept_chan_idx], 2)
eeg_objects = np.mean(eeg_objects[:,:,kept_chan_idx], 2)


# =============================================================================
# Compute the ERPs (average across images from the same condition)
# =============================================================================
erp_faces = np.mean(eeg_faces, 1)
erp_objects = np.mean(eeg_objects, 1)


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