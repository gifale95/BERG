"""Compute the stats on the results of the pairwise decoding analysis. The
stats consist of bootstrapped 95% confidence intervals for the pairwise
decoding results, as well as for the difference between exemplar, object, and
animacy decoding peaks.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
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
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
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
eeg_mds_single_sub = []

for s, sub in enumerate(args.subjects):

    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'object_categorization', 'pairwise_decoding', args.encoding_model,
        'pairwise_decoding_sub-'+format(sub,'02')+'_channels-'+
        '-'.join(args.channels)+'.npy')
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
    idx_tril = np.tril_indices(len(results['decoding_objects']), -1)
    decoding_objects.append(np.mean(
        results['decoding_objects'][idx_tril], 0))

    # Get the animacy decoding results
    decoding_animacy.append(results['decoding_animacy'])

    # Get the single subject MDS results
    eeg_mds_single_sub.append(results['eeg_mds'])

    # EEG metadata
    times = results['times']
    kept_ch_names = results['kept_ch_names']

# Convert to numpy arrays
decoding_exemplars = np.asarray(decoding_exemplars) * 100
decoding_objects = np.asarray(decoding_objects) * 100
decoding_animacy = np.asarray(decoding_animacy) * 100
eeg_mds_single_sub = np.asarray(eeg_mds_single_sub)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_exemplars = np.zeros((2, len(times)))
ci_objects = np.zeros((2, len(times)))
ci_animacy = np.zeros((2, len(times)))
ci_peak_latency_exemplars = np.zeros((2))
ci_peak_latency_objects = np.zeros((2))
ci_peak_latency_animacy = np.zeros((2))

exemplars_dist = np.zeros((args.n_iter, len(times)))
object_dist = np.zeros((args.n_iter, len(times)))
animacy_dist = np.zeros((args.n_iter, len(times)))
peak_latency_exemplars_dist = np.zeros((args.n_iter))
peak_latency_objects_dist = np.zeros((args.n_iter))
peak_latency_animacy_dist = np.zeros((args.n_iter))

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    exemplars_dist[i] = np.mean(decoding_exemplars[idx], 0)
    object_dist[i] = np.mean(decoding_objects[idx], 0)
    animacy_dist[i] = np.mean(decoding_animacy[idx], 0)
    peak_latency_exemplars_dist[i] = times[np.argmax(np.mean(
        decoding_exemplars[idx], 0))]
    peak_latency_objects_dist[i] = times[np.argmax(np.mean(
        decoding_objects[idx], 0))]
    peak_latency_animacy_dist[i] = times[np.argmax(np.mean(
        decoding_animacy[idx], 0))]

ci_exemplars[0] = np.percentile(exemplars_dist, 2.5, axis=0)
ci_exemplars[1] = np.percentile(exemplars_dist, 97.5, axis=0)
ci_objects[0] = np.percentile(object_dist, 2.5, axis=0)
ci_objects[1] = np.percentile(object_dist, 97.5, axis=0)
ci_animacy[0] = np.percentile(animacy_dist, 2.5, axis=0)
ci_animacy[1] = np.percentile(animacy_dist, 97.5, axis=0)
ci_peak_latency_exemplars[0] = np.percentile(peak_latency_exemplars_dist, 2.5)
ci_peak_latency_exemplars[1] = np.percentile(peak_latency_exemplars_dist, 97.5)
ci_peak_latency_objects[0] = np.percentile(peak_latency_objects_dist, 2.5)
ci_peak_latency_objects[1] = np.percentile(peak_latency_objects_dist, 97.5)
ci_peak_latency_animacy[0] = np.percentile(peak_latency_animacy_dist, 2.5)
ci_peak_latency_animacy[1] = np.percentile(peak_latency_animacy_dist, 97.5)


# =============================================================================
# Statistical significance
# =============================================================================
# Decoding significance
pval_exemplars = ttest_1samp(decoding_exemplars, 50, alternative='greater')[1]
pval_objects = ttest_1samp(decoding_objects, 50, alternative='greater')[1]
pval_animacy = ttest_1samp(decoding_animacy, 50, alternative='greater')[1]
# Multiple comparison correction
sig_exemplars = multipletests(pval_exemplars, 0.05, 'fdr_bh')[0]
sig_objects = multipletests(pval_objects, 0.05, 'fdr_bh')[0]
sig_animacy = multipletests(pval_animacy, 0.05, 'fdr_bh')[0]


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
# Save the results
# =============================================================================
results = {
    'eeg_mds': eeg_mds,
    'eeg_mds_single_sub': eeg_mds_single_sub,
    'decoding_exemplars': decoding_exemplars,
    'decoding_objects': decoding_objects,
    'decoding_animacy': decoding_animacy,
    'ci_exemplars': ci_exemplars,
    'ci_objects': ci_objects,
    'ci_animacy': ci_animacy,
    'sig_exemplars': sig_exemplars,
    'sig_objects': sig_objects,
    'sig_animacy': sig_animacy,
    'ci_peak_latency_exemplars': ci_peak_latency_exemplars,
    'ci_peak_latency_objects': ci_peak_latency_objects,
    'ci_peak_latency_animacy': ci_peak_latency_animacy,
    'times': times,
    'kept_ch_names': kept_ch_names
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_categorization', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results)