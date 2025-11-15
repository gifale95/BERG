"""Compute the stats on the results of the pairwise decoding analysis. The
stats consist of bootstrapped 95% confidence intervals for the pairwise
decoding results, as well as for the difference between exemplar and animacy
pairwise decoding peaks.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : list
    List containing the EEG channel type(s) retained for the analyses.
    Possible values are: 'O' (occipital), 'P' (posterior), 'T' (temporal),
    'C' (central), 'F' (frontal).
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
from scipy.stats import ttest_1samp


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default=['O', 'P'], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Pairwise decoding stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
decoding_exemplars = []
decoding_animacy = []

for sub in args.subjects:

    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'object_exemplar_animacy_categorization', 'pairwise_decoding_results',
        'pairwise_decoding_sub-'+format(sub,'02')+'_channels-'+
        ''.join(args.channels)+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    # Get the exemplars decoding results
    idx_tril = np.tril_indices(len(results['decoding_exemplars']), -1)
    decoding_exemplars.append(np.mean(
        results['decoding_exemplars'][idx_tril], 0))

    # Get the animacy decoding results
    decoding_animacy.append(results['decoding_animacy'])

    # EEG metadata
    times = results['times']
    kept_ch_names = results['kept_ch_names']

# Convert to numpy arrays
decoding_exemplars = np.asarray(decoding_exemplars)
decoding_animacy = np.asarray(decoding_animacy)

# Compute the peak latency difference of each subject
peak_latency_diff = []
for s in range(len(args.subjects)):
    peak_exemplars_latency = times[np.argmax(decoding_exemplars[s])] # type: ignore
    peak_animacy_latency = times[np.argmax(decoding_animacy[s])] # type: ignore
    peak_latency_diff.append(peak_animacy_latency - peak_exemplars_latency)
peak_latency_diff = np.array(peak_latency_diff)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_exemplars = np.zeros((2, len(times))) # type: ignore
ci_animacy = np.zeros((2, len(times))) # type: ignore
ci_peak_latency_diff = np.zeros((2)) # type: ignore

exemplars_dist = np.zeros((args.n_iter, len(times))) # type: ignore
animacy_dist = np.zeros((args.n_iter, len(times))) # type: ignore
latency_dist = np.zeros((args.n_iter)) # type: ignore

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    exemplars_dist[i] = np.mean(decoding_exemplars[idx], 0)
    animacy_dist[i] = np.mean(decoding_animacy[idx], 0)
    latency_dist[i] = np.mean(peak_latency_diff[idx])

ci_exemplars[0] = np.percentile(exemplars_dist, 2.5, axis=0)
ci_exemplars[1] = np.percentile(exemplars_dist, 97.5, axis=0)
ci_animacy[0] = np.percentile(animacy_dist, 2.5, axis=0)
ci_animacy[1] = np.percentile(animacy_dist, 97.5, axis=0)
ci_peak_latency_diff[0] = np.percentile(latency_dist, 2.5, axis=0)
ci_peak_latency_diff[1] = np.percentile(latency_dist, 97.5, axis=0)


# =============================================================================
# Compute the significance of the peak latency difference
# =============================================================================
pval_peak_latency_diff = ttest_1samp(peak_latency_diff, 0,
    alternative='greater')[1]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'decoding_exemplars': decoding_exemplars,
    'decoding_animacy': decoding_animacy,
    'ci_exemplars': ci_exemplars,
    'ci_animacy': ci_animacy,
    'peak_latency_diff': peak_latency_diff,
    'ci_peak_latency_diff': ci_peak_latency_diff,
    'pval_peak_latency_diff': pval_peak_latency_diff,
    'times': times, # type: ignore
    'kept_ch_names': kept_ch_names # type: ignore
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_exemplar_animacy_categorization',
    'pairwise_decoding_results', 'stats_'+'channels-'+
    ''.join(args.channels)+'.npy')

np.save(save_dir, results) # type: ignore