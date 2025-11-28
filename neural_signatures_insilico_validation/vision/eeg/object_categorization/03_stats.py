"""Compute the stats on the results of the pairwise decoding analysis. The
stats consist of bootstrapped 95% confidence intervals for the pairwise
decoding results, as well as for the difference between exemplar, object, and
animacy decoding peaks.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
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
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import MDS


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O', type=lambda s: s.split(','))
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
decoding_objects = []
decoding_animacy = []

for s, sub in enumerate(args.subjects):

    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'object_categorization', 'pairwise_decoding', 'pairwise_decoding_sub-'+
        format(sub,'02')+'_channels-'+'-'.join(args.channels)+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    # Get the in silico EEG responses averaged across repeats, and append them
    # across subject across the channels dimension
    if s == 0:
        eeg = np.mean(results['eeg'], 1)
    else:
        eeg = np.append(eeg, np.mean(results['eeg'], 1), 1)

    # Get the exemplars decoding results
    idx_tril = np.tril_indices(len(results['decoding_exemplars']), -1)
    decoding_exemplars.append(np.mean(
        results['decoding_exemplars'][idx_tril], 0))

    # Get the object category decoding results
    decoding_animacy.append(results['decoding_animacy'])

    # Get the animacy decoding results
    decoding_objects.append(results['decoding_objects'])

    # EEG metadata
    times = results['times']
    kept_ch_names = results['kept_ch_names']

# Convert to numpy arrays
decoding_exemplars = np.asarray(decoding_exemplars) * 100
decoding_objects = np.asarray(decoding_objects) * 100
decoding_animacy = np.asarray(decoding_animacy) * 100

# Compute the decoding peak latency difference of each subject # !!!
peak_latency_diff = []
for s in range(len(args.subjects)):
    peak_exemplars_latency = times[np.argmax(decoding_exemplars[s])] # type: ignore
    peak_animacy_latency = times[np.argmax(decoding_animacy[s])] # type: ignore
    peak_latency_diff.append(peak_animacy_latency - peak_exemplars_latency)
peak_latency_diff = np.array(peak_latency_diff)


# =============================================================================
# Perform MDS on the EEG responses of each time point
# =============================================================================
# Empty results array of shape (Images, 2 MDS dimensions, Times)
n_components = 2
eeg_mds = np.zeros((len(eeg), n_components, len(times)), dtype=np.float32)

# Loop across time point
for t in tqdm(range(len(times))):

    # Perform MDS
    embedding = MDS(n_components=n_components, n_init=10, max_iter=1000,
        random_state=20200220)
    eeg_mds[:,:,t] = embedding.fit_transform(eeg[:,:,t])


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_exemplars = np.zeros((2, len(times))) # type: ignore
ci_objects = np.zeros((2, len(times))) # type: ignore
ci_animacy = np.zeros((2, len(times))) # type: ignore
ci_peak_latency_diff = np.zeros((2)) # type: ignore

exemplars_dist = np.zeros((args.n_iter, len(times))) # type: ignore
object_dist = np.zeros((args.n_iter, len(times))) # type: ignore
animacy_dist = np.zeros((args.n_iter, len(times))) # type: ignore
latency_dist = np.zeros((args.n_iter)) # type: ignore

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    exemplars_dist[i] = np.mean(decoding_exemplars[idx], 0)
    object_dist[i] = np.mean(decoding_objects[idx], 0)
    animacy_dist[i] = np.mean(decoding_animacy[idx], 0)
    latency_dist[i] = np.mean(peak_latency_diff[idx])

ci_exemplars[0] = np.percentile(exemplars_dist, 2.5, axis=0)
ci_exemplars[1] = np.percentile(exemplars_dist, 97.5, axis=0)
ci_objects[0] = np.percentile(object_dist, 2.5, axis=0)
ci_objects[1] = np.percentile(object_dist, 97.5, axis=0)
ci_animacy[0] = np.percentile(animacy_dist, 2.5, axis=0)
ci_animacy[1] = np.percentile(animacy_dist, 97.5, axis=0)
ci_peak_latency_diff[0] = np.percentile(latency_dist, 2.5, axis=0)
ci_peak_latency_diff[1] = np.percentile(latency_dist, 97.5, axis=0)


# =============================================================================
# Statistical significance
# =============================================================================
# Decoding significance
pval_exemplars = ttest_1samp(decoding_exemplars, 50, alternative='greater')[1]
pval_objects = ttest_1samp(decoding_objects, 50, alternative='greater')[1]
pval_animacy = ttest_1samp(decoding_animacy, 50, alternative='greater')[1]
# Multiple comparison correction
sig_exemplars, _, _, _ = multipletests(pval_exemplars, 0.05, 'fdr_bh')
sig_objects, _, _, _ = multipletests(pval_objects, 0.05, 'fdr_bh')
sig_animacy, _, _, _ = multipletests(pval_animacy, 0.05, 'fdr_bh')

# Significance of peak latency differences # !!!
pval_peak_latency_diff = ttest_1samp(peak_latency_diff, 0,
    alternative='greater')[1]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'eeg_mds': eeg_mds,
    'decoding_exemplars': decoding_exemplars,
    'decoding_objects': decoding_objects,
    'decoding_animacy': decoding_animacy,
    'ci_exemplars': ci_exemplars,
    'ci_objects': ci_objects,
    'ci_animacy': ci_animacy,
    'sig_exemplars': sig_exemplars,
    'sig_objects': sig_objects,
    'sig_animacy': sig_animacy,

    'peak_latency_diff': peak_latency_diff, # !!!
    'ci_peak_latency_diff': ci_peak_latency_diff, # !!!
    'pval_peak_latency_diff': pval_peak_latency_diff, # !!!

    'times': times, # type: ignore
    'kept_ch_names': kept_ch_names # type: ignore
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_categorization', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_' + 'channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore