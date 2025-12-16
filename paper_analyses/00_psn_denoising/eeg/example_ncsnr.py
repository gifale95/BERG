import argparse
import os
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the EEG responses
# =============================================================================
eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2', 'eeg_sub-'+format(args.subject, '02')+
    '_split-test.h5')
eeg_test = h5py.File(eeg_dir_test, 'r')['eeg'][:].astype(np.float32)

# Reshape the EEG responses to (Units, Conditions, Repeats)
n_cond_test = eeg_test.shape[0]
n_trial_test = eeg_test.shape[1]
n_chan = eeg_test.shape[2]
n_time = eeg_test.shape[3]
eeg_test = np.reshape(eeg_test, (n_cond_test, n_trial_test, -1))
eeg_test = np.swapaxes(np.swapaxes(eeg_test, 0, 2), 1, 2)
eeg_test = np.nan_to_num(eeg_test)

# Load the EEG channel names and time points
metadata_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-things_eeg_2', 'metadata_subject-1.npy')

metadata = np.load(metadata_dir, allow_pickle=True).item()

ch_names = metadata['ch_names']
times = metadata['times']


# =============================================================================
# Compute the NCSNR and noise ceiling
# =============================================================================
# Estimate the noise standard deviation (calculate the variance of the
# responses across the 30 presentations of each test image).
var = np.nanvar(eeg_test, axis=2, ddof=1)

# Average the variance across images and compute the square root of the
# result
sigma_noise = np.sqrt(np.nanmean(var, 1))

# Estimate the signal standard deviation (total variance - noise variance)
tot_var_data = np.nanvar(np.reshape(eeg_test, (eeg_test.shape[0], -1)), axis=1,
    ddof=1)
sigma_signal = tot_var_data - (sigma_noise ** 2)
sigma_signal[sigma_signal<0] = 0
sigma_signal = np.sqrt(sigma_signal)

# Compute the ncsnr
ncsnr = sigma_signal / sigma_noise

# Convert the ncsnr to noise ceiling (the noise ceiling is in r² explained
# variance units)
noise_ceiling = (ncsnr ** 2) / ((ncsnr ** 2) + (1 / n_trial_test))

# Reshape to (channels, times)
ncsnr = np.reshape(ncsnr, (n_chan, n_time))
noise_ceiling = np.reshape(noise_ceiling, (n_chan, n_time))



eeg_test = np.random.randn(8820, 200, 80).astype(np.float32)
var = np.nanvar(eeg_test, axis=2, ddof=1)

# Average the variance across images and compute the square root of the
# result
sigma_noise = np.sqrt(np.nanmean(var, 1))

# Estimate the signal standard deviation (total variance - noise variance)
tot_var_data = np.nanvar(np.reshape(eeg_test, (eeg_test.shape[0], -1)), axis=1,
    ddof=1)
sigma_signal = tot_var_data - (sigma_noise ** 2)
sigma_signal[sigma_signal<0] = 0
sigma_signal = np.sqrt(sigma_signal)

# Compute the ncsnr
ncsnr = sigma_signal / sigma_noise

# Convert the ncsnr to noise ceiling (the noise ceiling is in r² explained
# variance units)
noise_ceiling = (ncsnr ** 2) / ((ncsnr ** 2) + (1 / n_trial_test))

# Reshape to (channels, times)
ncsnr = np.reshape(ncsnr, (n_chan, n_time))
noise_ceiling = np.reshape(noise_ceiling, (n_chan, n_time))


# =============================================================================
# Compute the split-half correlation
# =============================================================================
idx = n_trial_test // 2
eeg_test_split1 = np.nanmean(eeg_test[:,:,:idx], 2)
eeg_test_split2 = np.nanmean(eeg_test[:,:,idx:], 2)

correlation = np.zeros((n_chan*n_time))
for u in tqdm(range(len(correlation))):
    correlation[u] = pearsonr(eeg_test_split1[u], eeg_test_split2[u])[0]

# Reshape to (channels, times)
correlation = correlation.reshape(n_chan, n_time)


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 15
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


# =============================================================================
# Plot
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False,
    figsize=(20, 7))
axs = np.reshape(axs, (-1)) # type: ignore

# Plot the ncsnr
axs[0].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')
axs[0].plot(times, np.mean(ncsnr, 0), linewidth=2)
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
axs[0].set_xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=min(times), right=max(times))
axs[0].set_ylabel("NCSNR", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
axs[0].set_yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-.05, top=0.3)
title = 'NCSNR'
axs[0].set_title(title)

# Plot the noise ceiling
axs[1].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')
axs[1].plot(times, np.mean(noise_ceiling, 0), linewidth=2)
axs[1].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
axs[1].set_xticks(ticks=xticks, labels=xlabels)
axs[1].set_xlim(left=min(times), right=max(times))
axs[1].set_ylabel("Noise ceiling", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
axs[1].set_yticks(ticks=yticks, labels=ylabels)
axs[1].set_ylim(bottom=-.05, top=0.3)
title = 'Noise ceiling'
axs[1].set_title(title)

# Plot the split-half correlation
axs[2].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=2, alpha=.5, label='_nolegend_')
axs[2].plot(times, np.mean(correlation, 0), linewidth=2)
axs[2].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
axs[2].set_xticks(ticks=xticks, labels=xlabels)
axs[2].set_xlim(left=min(times), right=max(times))
axs[2].set_ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
axs[2].set_yticks(ticks=yticks, labels=ylabels)
axs[2].set_ylim(bottom=-.05, top=0.3)
title = 'Split-half correlation'
axs[2].set_title(title)