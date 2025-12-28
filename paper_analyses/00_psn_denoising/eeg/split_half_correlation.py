import argparse
import os
import numpy as np
from psn import psn
from psn import PSN
from scipy.stats import pearsonr
from matplotlib import pyplot as plt


# =============================================================================
# Load the EEG responses
# =============================================================================
data_dir = '/scratch/giffordale95/eeg_data.npy'
data = np.load(data_dir, allow_pickle=True).item()
eeg = data['eeg']
ch_names = data['ch_names']
times = data['times']


# =============================================================================
# Denoise the EEG responses
# =============================================================================
# Fit the denoiser
denoiser = PSN(mode='conservative')
denoiser.fit(eeg)

# Denoise the EEG responses independently for each repeat
eeg_denoised = np.zeros((eeg.shape), dtype=np.float32)
for r in range(eeg.shape[2]):
    eeg_denoised[:,:,r] = denoiser.transform(np.reshape(eeg[:,:,r],
        (eeg.shape[0], eeg.shape[1], 1)))



psn(eeg,'conservative')
results = psn(eeg,'conservative')

# =============================================================================
# Split-half correlation
# =============================================================================
# Compute the split-half correlation
corr = np.zeros((eeg.shape[0]), dtype=np.float32)
corr_denoised = np.zeros((eeg.shape[0]), dtype=np.float32)
idx = eeg.shape[2] // 2
for u in range(eeg.shape[0]):
    corr[u] = pearsonr(np.mean(eeg[u,:,:idx], 1),
        np.mean(eeg[u,:,idx:], 1))[0]
    corr_denoised[u] = pearsonr(np.mean(eeg_denoised[u,:,:idx], 1),
        np.mean(eeg_denoised[u,:,idx:], 1))[0]
corr = np.reshape(corr, (len(ch_names), len(times)))
corr_denoised = np.reshape(corr_denoised, (len(ch_names), len(times)))

# Plot
plt.figure()
plt.plot(times, np.mean(corr, 0), label='EEG')
plt.plot(times, np.mean(corr_denoised, 0), label='EEG denoised')
plt.ylabel("Pearson's $r$")
plt.xlabel("Time (s)")
plt.legend()