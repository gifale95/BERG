"""Plot EEG Moments ERPs as a funciton of video repetitions.

Parameters
----------
sub : int
    Used subject.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/decoding_expra', type=str)
parser.add_argument('--data_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'plots_erps')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the preprocessed EEG responses
# =============================================================================
data_dir = os.path.join(args.data_dir, 'dataset', 'preprocessed_data',
    'dataset_02', 'eeg', 'sub-01', 'mvnn-none', 'baseline_correction-01',
    'highpass-0.1_lowpass-40', 'sfreq-0100', 'preprocessed_data.npy')

data = np.load(data_dir, allow_pickle=True).item()

eeg_data = data['eeg_data']
stimuli_presentation_order = data['stimuli_presentation_order']
times = data['times']
ch_names = data['ch_names']
del data


# =============================================================================
# Select the responses for one video condition
# =============================================================================
video_number = 1100

eeg = []

for i in range(len(eeg_data)):
    idx = np.where(stimuli_presentation_order[i] == video_number)[0]
    eeg.append(eeg_data[i][idx])

eeg = np.concatenate(eeg, axis=0)


# =============================================================================
# Time point selection
# =============================================================================
idx_t_max = np.where(times == 1)[0][0]
eeg = eeg[:,:,:idx_t_max]
times = times[:idx_t_max]


# =============================================================================
# Channel selection
# =============================================================================
idx_chan = []
ch_names_new = []

for c, chan in enumerate(ch_names):
    if 'O' in chan:
        idx_chan.append(c)
        ch_names_new.append(chan)
idx_chan = np.array(idx_chan)

eeg = eeg[:,idx_chan]


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 25
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 0
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 0
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

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(eeg.shape[1])


# =============================================================================
# Plot the ERPs as a function of repeats
# =============================================================================
repeats = [1, 12, 24]

# Create the figure
fig, axs = plt.subplots(nrows=1, ncols=len(repeats), sharex=True, sharey=True,
    figsize=(30, 7.5))
axs = np.reshape(axs, (-1))

for r, rep in enumerate(repeats):

    # Plot the stimulus onset dashed lines
    # axs[r].plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.25,
    #     label='_nolegend_')

    # Plot the pairwise decoding
    erp = np.mean(eeg[:rep], 0)
    for c in range(len(erp)):
        axs[r].plot(times, erp[c], color=colors[c], linewidth=2)

    # x-axis parameters
    axs[r].set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-0.2, 0, .2, .4, .6, .8, .99]
    xlabels = [-200, 0, 200, 400, 600, 800, 1000]
    plt.xticks(ticks=xticks, labels=xlabels)
    axs[r].set_xlim(left=-0.2, right=1)

    # y-axis parameters
    if r == 0:
        axs[r].set_ylabel("Voltage (µV)", fontsize=fontsize)
    
    # Title
    axs[r].set_title(f'{rep} repeats', fontsize=fontsize)

# Save the figure
file_name = os.path.join(save_dir, 'eeg_erps_as_function_of_repeats.png')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='png')