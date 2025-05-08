"""Preprocess the raw EEG data from the THINGS EEG 2 dataset
(https://doi.org/10.1016/j.neuroimage.2022.119754):
	- channel selection,
	- filtering,
	- epoching,
	- current source density transform,
	- frequency downsampling,
	- baseline correction

After preprocessing, the EEG data is reshaped to:
(Image conditions x EEG repetitions x EEG channels x EEG time points).

The data of the test and train EEG partitions is saved independently.

Parameters
----------
subject : int
	Number of the used THINGS EEG2 subject.
n_ses : int
	Number of EEG sessions.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
tmin : float
	Start time of the epochs in seconds, relative to stimulus onset.
tmax : float
	End time of the epochs in seconds, relative to stimulus onset.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
baseline_mode : str
	Whether to apply 'mean' or 'zscore' baseline correction mode.
csd : int
	Whether to transform the data into current source density [1] or not [0].
sfreq : int
	Downsampling frequency.
things_eeg_2_dir : str
	Directory of the THINGS EEG2 dataset.
	https://osf.io/3jk45/
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit (NEST).
	https://github.com/gifale95/NEST

"""

import argparse
from utils import epoching
from utils import save_prepr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--n_ses', default=4, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.03, type=float)
parser.add_argument('--tmin', default=-.1, type=float)
parser.add_argument('--tmax', default=.6, type=float)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--baseline_mode', default='zscore', type=str)
parser.add_argument('--csd', default=1, type=int)
parser.add_argument('--sfreq', default=200, type=int)
parser.add_argument('--things_eeg_2_dir', default='../things_eeg_2', type=str)
parser.add_argument('--nest_dir', default='../neural-encoding-simulation-toolkit', type=str)
args = parser.parse_args()

print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Epoch and sort the data
# =============================================================================
# After preprocessing, the EEG data is reshaped to:
# (Image conditions x EEG repetitions x EEG channels x EEG time points)
# This step is applied independently to the data of each partition and session.
epoched_test, _, ch_names, times = epoching(args, 'test')
epoched_train, img_conditions_train, _, _ = epoching(args, 'training')


# =============================================================================
# Merge and save the preprocessed data
# =============================================================================
# In this step the data of all sessions is merged into the shape:
# (Image conditions x EEG repetitions x EEG channels x EEG time points)
# Then, the preprocessed data of the test and training data partitions is saved.
save_prepr(args, epoched_test, epoched_train, img_conditions_train, ch_names,
	times)
