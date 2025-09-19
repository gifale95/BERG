"""Plot the encoding models' prediction accuracy for the test stimuli.

Parameters
----------
monkeys : list
	List with all used TVSD monkeys.
model : str
	Name of the used encoding model.
berg_dir : str
	Directory of the Brain Encoding Response Generator (BERG).
	https://github.com/gifale95/BERG
 

python berg_creation_code/03_test_encoding_models/train_dataset-tvsd_monkey/02_plot.py \
    --monkey monkeyF \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --only_cls False \
    --regression ridge \
    --model vit_b_32 

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
parser.add_argument('--monkey', nargs='+', default=['monkeyN', 'monkeyF'],
	help='List of monkeys to analyze (e.g., --monkey monkeyN monkeyF)')
parser.add_argument('--model', required=True, choices=["vit_b_32", "clip.vit_b_32"],
                   help="Selecting which model to use")
parser.add_argument('--only_cls', required=True, choices=["True", "False"],
                    help='If we should only use CLS token or all patches')
parser.add_argument('--regression', required=True, choices=["ridge", "linear"],
                   help="Select type of regression")
parser.add_argument('--berg_dir', required=True, type=str)
args = parser.parse_args()

args.only_cls = args.only_cls == "True"

cls_suffix = 'cls' if args.only_cls else 'all'


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
correlation_results = []
oracle_all = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
	'train_dataset-tvsd_monkey', f'model-{args.model}', 'metadata')

for monkey in args.monkey:
	file_name = f'metadata_{args.regression}_{cls_suffix}_{monkey}.npy'
	metadata = np.load(os.path.join(metadata_dir, file_name),
		allow_pickle=True).item()
	correlation_results.append(
		metadata['encoding_models']['correlation_results'])
	oracle_all.append(metadata['neural']['oracle'])
	times = metadata['neural']['times']

correlation_results = np.asarray(correlation_results)
oracle_all = np.asarray(oracle_all)

print(f"Correlation results shape: {correlation_results.shape}")
print(f"Oracle shape: {oracle_all.shape}")


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 12
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
plt.rcParams["text.usetex"] = False


# =============================================================================
# Plot the encoding accuracy results
# =============================================================================
n_monkeys = len(args.monkey)

# Create subplot grid based on number of monkeys
if n_monkeys == 1:
	# Single monkey: 2 rows x 6 cols (row 0: average plot, row 1: 5 individual electrodes)
	fig = plt.figure(figsize=(24, 8))
	n_rows = 2
else:
	# Two monkeys: 3 rows x 6 cols (row 0: both averages, row 1&2: individual electrodes)
	fig = plt.figure(figsize=(24, 12))
	n_rows = 3

# First row: Average plots
for m, monkey in enumerate(args.monkey):
	# Each monkey gets 3 columns (left half or right half)
	ax_main = plt.subplot2grid((n_rows, 6), (0, m*3), colspan=3)
	
	# Average correlation across all electrodes for each timepoint
	correlation_avg = np.mean(correlation_results[m], axis=1)
	
	# Oracle statistics across electrodes
	oracle_mean = np.mean(oracle_all[m])
	oracle_std = np.std(oracle_all[m])
	
	# Plot oracle bounds
	ax_main.axhline(oracle_mean, color='darkgray', linewidth=2, 
		label=f'Oracle (μ={oracle_mean:.3f})')
	ax_main.axhspan(oracle_mean - oracle_std, oracle_mean + oracle_std, 
		color='lightgray', alpha=0.3, label=f'Oracle ±σ')
	
	# Plot average correlation
	ax_main.plot(times, correlation_avg, color='blue', linewidth=3,
		label='Average prediction')
	
	# Plot chance and stimulus onset lines
	ax_main.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
	ax_main.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
	
	# x-axis parameters
	ax_main.set_xlabel('Time (ms)', fontsize=fontsize)
	xticks = [-100, -50, 0, 50, 100, 150, 200]
	ax_main.set_xticks(xticks)
	ax_main.set_xlim(left=times[0], right=times[-1])
	
	# y-axis parameters
	ax_main.set_ylabel('Pearson\'s r', fontsize=fontsize)
	ax_main.set_ylim(bottom=-0.1, top=max(oracle_mean + oracle_std + 0.1, 1.0))
	
	# Title and legend
	ax_main.set_title(f'{monkey} - Average', fontsize=fontsize+2, fontweight='bold')
	ax_main.legend(loc='upper right', fontsize=fontsize-2)

# Individual electrode plots for each monkey
for m, monkey in enumerate(args.monkey):
	if n_monkeys == 1:
		row_start = 1  # Single monkey: electrodes on row 1
		col_offset = 0  # Start from left side
	else:
		row_start = 1 + m  # Two monkeys: monkey 0 on row 1, monkey 1 on row 2
		col_offset = 0  # Start from left side
	
	# Oracle statistics for y-axis scaling
	oracle_mean = np.mean(oracle_all[m])
	oracle_std = np.std(oracle_all[m])
	
	for e in range(min(5, correlation_results.shape[2])):
		ax_ind = plt.subplot2grid((n_rows, 6), (row_start, col_offset + e), colspan=1)
		
		electrode_corr = correlation_results[m, :, e]
		
		# Oracle for this specific electrode
		electrode_oracle = oracle_all[m, e, 0] if oracle_all.shape[2] == 1 else oracle_all[m, e]
		
		# Plot oracle bounds for this electrode
		ax_ind.axhline(electrode_oracle, color='darkgray', linewidth=2)
		
		# Plot electrode correlation
		ax_ind.plot(times, electrode_corr, color='red', linewidth=2)
		
		# Plot chance and stimulus onset lines
		ax_ind.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
		ax_ind.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
		
		# x-axis parameters
		ax_ind.set_xlabel('Time (ms)', fontsize=fontsize)
		ax_ind.set_xticks(xticks)
		ax_ind.set_xlim(left=times[0], right=times[-1])
		
		# y-axis parameters
		ax_ind.set_ylim(bottom=-0.1, top=max(oracle_mean + oracle_std + 0.1, 1.0))
		
		# Title
		if n_monkeys == 1:
			ax_ind.set_title(f'Electrode {e}', fontsize=fontsize)
		else:
			ax_ind.set_title(f'{monkey} - Electrode {e}', fontsize=fontsize)
		
		# Only show y-axis label on leftmost plot
		if e == 0:
			ax_ind.set_ylabel('Pearson\'s r', fontsize=fontsize)

# Adjust layout and save
plt.tight_layout()

# Create save directory
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
	'train_dataset-tvsd_monkey', f'model-{args.model}', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

# Save the figure
save_name = f'encoding_accuracy_model-{args.regression}_{cls_suffix}_{args.model}'
fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight', format='png')
fig.savefig(os.path.join(save_dir, f'{save_name}.jpg'), dpi=300, bbox_inches='tight', format='jpeg')

print(f"Plot saved to: {save_dir}/{save_name}.png and {save_dir}/{save_name}.jpg")
plt.show()