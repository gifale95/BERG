"""Plot the encoding models' prediction accuracy for the test stimuli.

Parameters
----------
monkeys : list
	List with all used TVSD monkeys.
electrodes : str
	Used electrode selection strategy ['all', 'best_snr', 'high_snr'].
model : str
	Name of the used encoding model.
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG
 
 

python berg_creation_code/03_test_encoding_models/train_dataset-tvsd_monkey/02_plot.py --monkeys monkeyF --electrodes best_snr --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator'


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
parser.add_argument('--monkeys', nargs='+', default=['monkeyN', 'monkeyF'],
	help='List of monkeys to analyze (e.g., --monkeys monkeyN monkeyF)')
parser.add_argument('--electrodes', type=str, default='all', 
	choices=['all', 'best_snr', 'high_snr'])
parser.add_argument('--model', type=str, default='clip_vit_b_32')
parser.add_argument('--berg_dir', required=True, type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
correlation_results = []
snr_max_all = []
oracle_all = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
	'train_dataset-tvsd_monkey', f'model-{args.model}', 'metadata')

for monkey in args.monkeys:
	file_name = f'metadata_{monkey}.npy'
	metadata = np.load(os.path.join(metadata_dir, file_name),
		allow_pickle=True).item()
	correlation_results.append(
		metadata['encoding_models']['correlation_results'])
	snr_max_all.append(metadata['neural']['SNR_max'])
	oracle_all.append(metadata['neural']['oracle'])
	times = metadata['neural']['times']

correlation_results = np.asarray(correlation_results)
snr_max_all = np.asarray(snr_max_all)
oracle_all = np.asarray(oracle_all)

print(f"Correlation results shape: {correlation_results.shape}")


# =============================================================================
# Electrode selection
# =============================================================================
if args.electrodes == 'all':
	electrode_desc = 'All electrodes'
	# Use all electrodes
	correlation_selected = correlation_results
elif args.electrodes == 'best_snr':
	electrode_desc = 'Best SNR electrodes (top 20%)'
	# Select top 20% of electrodes based on SNR_max
	correlation_selected = []
	for m in range(len(args.monkeys)):
		n_best = int(0.2 * len(snr_max_all[m]))
		best_electrodes = np.argsort(snr_max_all[m])[-n_best:]
		correlation_selected.append(correlation_results[m, best_electrodes, :])
	correlation_selected = np.asarray(correlation_selected)
elif args.electrodes == 'high_snr':
	electrode_desc = 'High SNR electrodes (SNR > median)'
	# Select electrodes with SNR above median
	correlation_selected = []
	for m in range(len(args.monkeys)):
		snr_threshold = np.median(snr_max_all[m])
		high_snr_electrodes = np.where(snr_max_all[m] > snr_threshold)[0]
		correlation_selected.append(correlation_results[m, high_snr_electrodes, :])
	correlation_selected = np.asarray(correlation_selected)

# Average the encoding accuracies across the selected electrodes
correlation_averaged = np.mean(correlation_selected, axis=1)

print(f"Selected correlation shape: {correlation_selected.shape}")
print(f"Averaged correlation shape: {correlation_averaged.shape}")


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
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255)]


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
if len(args.monkeys) == 1:
	fig, ax = plt.subplots(figsize=(8, 6))
	axs = [ax]
else:
	fig, axs = plt.subplots(nrows=1, ncols=len(args.monkeys), figsize=(12, 6), 
		sharex=True, sharey=True)

for m, monkey in enumerate(args.monkeys):

	# Plot the chance and stimulus onset dashed lines
	axs[m].plot([times[0], times[-1]], [0, 0], 'k--', linewidth=2, alpha=.5)
	axs[m].plot([0, 0], [-0.1, 1], 'k--', linewidth=2, alpha=.5)

	# Plot the correlation results
	axs[m].plot(times, correlation_averaged[m], color=colors[m % len(colors)], linewidth=3,
		label=f'{monkey}')

	# x-axis parameters
	axs[m].set_xlabel('Time (ms)', fontsize=fontsize)
	xticks = [-100, -50, 0, 50, 100, 150, 200]
	axs[m].set_xticks(xticks)
	axs[m].set_xlim(left=times[0], right=times[-1])

	# y-axis parameters
	if m == 0:
		axs[m].set_ylabel('Pearson\'s $r$', fontsize=fontsize)
	yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
	axs[m].set_yticks(yticks)
	axs[m].set_ylim(bottom=-0.05, top=1.0)

	# Title
	axs[m].set_title(f'{monkey} - {electrode_desc}', fontsize=fontsize)

	# Add stimulus onset annotation
	axs[m].text(10, 0.9, 'Stimulus\nOnset', fontsize=12, ha='left', va='top',
		bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout and save
plt.tight_layout()

# Save the figure
save_name = f'encoding_accuracy_electrodes-{args.electrodes}_model-{args.model}'
fig.savefig(f'{save_name}.svg', bbox_inches='tight', transparent=True, format='svg')
fig.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', transparent=True, format='png')

print(f"Plot saved as: {save_name}.svg and {save_name}.png")
plt.show()