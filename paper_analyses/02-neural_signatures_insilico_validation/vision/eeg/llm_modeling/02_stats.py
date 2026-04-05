"""Compute the stats on the results of the RSA analysis between in silico EEG
responses and LLM embeddings. The stats consist of bootstrapped 95%
confidence intervals and significance estimates.

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

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the decoding and RSA results
# =============================================================================
decoding = []
rsa = []

for sub in args.subjects:

    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'llm_modeling', 'rsa', args.encoding_model, 'rsa_sub-'+
        format(sub,'02')+'_channels-'+'-'.join(args.channels)+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()

    # Get the decoding results
    idx_tril = np.tril_indices(len(results['eeg_rdm']), -1)
    decoding.append(np.mean(results['eeg_rdm'][idx_tril], 0))

    # Get the RSA results
    rsa.append(results['rsa'])

    # EEG metadata
    times = results['metadata']['eeg']['times']

# Convert to numpy arrays
decoding = np.asarray(decoding) * 100
rsa = np.asarray(rsa)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_decoding = np.zeros((2, len(times)))
ci_rsa = np.zeros((2, len(times)))
ci_peak_latency_ci_decoding = np.zeros((2))
ci_peak_latency_ci_rsa = np.zeros((2))

decoding_dist = np.zeros((args.n_iter, len(times)))
rsa_dist = np.zeros((args.n_iter, len(times)))
peak_lat_dec_dist = np.zeros((args.n_iter))
peak_lat_rsa_dist = np.zeros((args.n_iter))

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    decoding_dist[i] = np.mean(decoding[idx], 0)
    rsa_dist[i] = np.mean(rsa[idx], 0)
    peak_lat_dec_dist[i] = times[np.argmax(np.mean(decoding[idx], 0))]
    peak_lat_rsa_dist[i] = times[np.argmax(np.mean(rsa[idx], 0))]

ci_decoding[0] = np.percentile(decoding_dist, 2.5, axis=0)
ci_decoding[1] = np.percentile(decoding_dist, 97.5, axis=0)
ci_rsa[0] = np.percentile(rsa_dist, 2.5, axis=0)
ci_rsa[1] = np.percentile(rsa_dist, 97.5, axis=0)
ci_peak_latency_ci_decoding[0] = np.percentile(peak_lat_dec_dist, 2.5, axis=0)
ci_peak_latency_ci_decoding[1] = np.percentile(peak_lat_dec_dist, 97.5, axis=0)
ci_peak_latency_ci_rsa[0] = np.percentile(peak_lat_rsa_dist, 2.5, axis=0)
ci_peak_latency_ci_rsa[1] = np.percentile(peak_lat_rsa_dist, 97.5, axis=0)


# =============================================================================
# RSA difference scores between late (200-400ms) and early (50-200ms) time points
# =============================================================================
# Average the MSE across time points
idx_early = np.where((times >= 0.06) & (times <= 0.2))[0]
idx_late = np.where((times > 0.2) & (times <= 0.4))[0]

# Compute the difference scores between late and early time points for each
# subject
diff_rsa_late_early = np.mean(rsa[:,idx_late], 1) - \
    np.mean(rsa[:,idx_early], 1)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'decoding': decoding,
    'rsa': rsa,
    'ci_decoding': ci_decoding,
    'ci_rsa': ci_rsa,
    'ci_peak_latency_ci_decoding': ci_peak_latency_ci_decoding,
    'ci_peak_latency_ci_rsa': ci_peak_latency_ci_rsa,
    'diff_rsa_late_early': diff_rsa_late_early,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'llm_modeling', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results)