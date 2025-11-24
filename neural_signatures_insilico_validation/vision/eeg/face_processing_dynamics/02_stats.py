"""Compute the stats (significance and confidence intervals) on the results of
the EEG face processing dynamics analysis.

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
from scipy.stats import ttest_1samp
import itertools
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

print('>>> Face processing dynamics - Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the results
# =============================================================================
scores = {}

for s, sub in enumerate(args.subjects):

    # Load the results
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'face_processing_dynamics', 'rsa', 'rsa_sub-'+format(sub,'02')+
        '_channels-'+'-'.join(args.channels)+'.npy')
    results = np.load(data_dir, allow_pickle=True).item()
    
    # Get the RSA results
    for key, val in results['rsa'].items():
        if s == 0:
           scores[key] = []
        scores[key].append(results['rsa'][key])
    
    # Get the decoding results
    if s == 0:
        scores['image'] = []
    scores['image'].append(results['eeg_decoding'])
    times = results['times']

# Format the results to numpy arrays
for key, val in scores.items():
    scores[key] = np.array(val)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_scores = {}

for key, val in tqdm(scores.items()):

    ci = np.zeros((2, len(times))) # type: ignore
    dist = np.zeros((args.n_iter, len(times))) # type: ignore

    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.subjects)))
        dist[i] = np.mean(val[idx], 0)

    ci[0] = np.percentile(dist, 2.5, axis=0)
    ci[1] = np.percentile(dist, 97.5, axis=0)

    ci_scores[key] = ci
    del ci


# =============================================================================
# Peak latency analysis
# =============================================================================
# Calculate the peak latencies
peak_latency = {}
for key, val in scores.items():
    peak_latency[key] = times[np.argsort(np.mean(val, 0))[::-1][0]]

# Calculate the peak CIs: bootstrap the RSA scores across subjects, and at each
# iteration store the time point of the peak. Then compute the CIs from this
# distribution.
ci_peak_latency = {}
for key, val in tqdm(scores.items()):
    ci = np.zeros((2)) # type: ignore
    dist = np.zeros((args.n_iter)) # type: ignore
    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.subjects)))
        dist[i] = times[np.argsort(np.mean(val[idx], 0))[::-1][0]]
    ci[0] = np.percentile(dist, 2.5, axis=0)
    ci[1] = np.percentile(dist, 97.5, axis=0)
    ci_peak_latency[key] = ci
    del ci

# Significance between peaks: t-tests between models using the peak latencies
    # of each subject. # !!! Sign permutation tests (FDR correction)


# =============================================================================
# Onset latency analysis # !!! Sign permutation tests (FDR correction)
# =============================================================================
# Calculate the significance (sign permutation tests, FDR corrected)
permutations = np.array(list(itertools.product([-1, 1],
    repeat=len(args.subjects))))
pval_scores = {}
sig_scores = {}
for key, val in scores.items():
    pval = np.ones((len(times)))
    val_perm = permutations[:, :, None] * val[None, :, :]
    for t in range(len(times)):
        count = len(np.where(
            np.mean(val_perm[:,:,t], 1) >= np.mean(val[:,t]))[0])
        pval[t] = count / len(permutations)
    sig, pval_corrected, _, _ = multipletests(pval, 0.05, 'fdr_bh')
    pval_scores[key] = pval_corrected
    sig_scores[key] = sig
    del pval, pval_corrected, sig

# Calculate the onset CIs: bootstrap the RSA scores across subjects, and at
# each iteration compute the significance across all time points, correct it
# with FDR, and store the time point of significance onset. Then compute the
# CIs from this distribution. # !!!
ci_onset_latency = {}
for key, val in tqdm(scores.items()):
    ci = np.zeros((2)) # type: ignore
    dist = np.zeros((args.n_iter)) # type: ignore
    for i in range(args.n_iter):
        idx = resample(np.arange(len(args.subjects)))
        pval = np.ones((len(times)))
        val_perm = permutations[:, :, None] * val[idx][None, :, :]
        for t in range(len(times)):
            count = len(np.where(
                np.mean(val_perm[:,:,t], 1) >= np.mean(val[:,t]))[0])
            pval[t] = count / len(permutations)
        sig = multipletests(pval, 0.05, 'fdr_bh')[0]
        dist[i] = times[np.where(sig == True)[0][0]]
    ci[0] = np.percentile(dist, 2.5, axis=0)
    ci[1] = np.percentile(dist, 97.5, axis=0)
    ci_onset_latency[key] = ci
    del ci, pval, sig

# Significance between onsets: # !!!


# =============================================================================
# Save the results # !!!
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
    'vision', 'eeg', 'face_processing_dynamics', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_' + 'channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore







# =============================================================================
# Plot the results # !!!
# =============================================================================
import matplotlib
from matplotlib import pyplot as plt

fontsize = 30
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(0/255, 255/255, 0/255), (255/255, 0/255, 0/255),
    (0/255, 0/255, 255/255)]

fig= plt.figure(figsize=(13, 7))

# Plot the stimulus onset dashed line
plt.plot([-10, 100], [0, -0], 'k--', linewidth=3,
    alpha=.5, label='_nolegend_')

# Plot the RSA scores
models = ['age', 'gender', 'identity']
for m, model in enumerate(models):
    plt.plot(times, np.mean(scores[model], 0), linewidth=3, color=colors[m],
        label=model)

# Plot the CIs
for m, model in enumerate(models):
    plt.fill_between(times, ci_scores[model][1], ci_scores[model][0],
        color=colors[m], alpha=.2)

# Plot the significance markers
for m, model in enumerate(models):
    sig = np.empty(len(times))
    sig[:] = np.nan
    sig[sig_scores[model]] = -0.005 - m * 0.005
    plt.scatter(times, sig, s=100, color=colors[m])

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels) # type: ignore
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [-1.5, -1, -.5, 0, .5]
ylabels = [-1.5, -1, -.5, 0, .5]
#plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
#plt.ylim(bottom=-1.5, top=.5)

# Legend
plt.legend(ncol=3, fontsize=fontsize, loc=0, frameon=False)