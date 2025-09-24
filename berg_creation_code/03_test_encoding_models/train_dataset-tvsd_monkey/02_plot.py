"""Plot the encoding models' prediction accuracy for the test stimuli by brain region.

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
    --monkey monkeyF monkeyN \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --only_cls True \
    --regression linear \
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
# Load the encoding models' encoding accuracy and metadata
# =============================================================================
correlation_results = []
roi_data = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
    'train_dataset-tvsd_monkey', f'model-{args.model}', 'metadata')

for monkey in args.monkey:
    # Load all data from single metadata file
    file_name = f'metadata_{args.regression}_{cls_suffix}_{monkey}.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name),
        allow_pickle=True).item()
    
    correlation_results.append(metadata['correlation_results'])
    times = metadata['times']
    roi_assignments = metadata['roi_assignments']
    roi_labels = metadata['roi_labels']
    roi_data.append((roi_assignments, roi_labels))

correlation_results = np.asarray(correlation_results)

print(f"Correlation results shape: {correlation_results.shape}")


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


roi_colors = {
    'V1': '#e74c3c',   # Red - vibrant red
    'V4': '#3498db',   # Blue - bright blue  
    'IT': '#2ecc71'    # Green - emerald green
}


# =============================================================================
# Plot the encoding accuracy results by brain region
# =============================================================================
n_monkeys = len(args.monkey)

# Create figure with proper layout
if n_monkeys == 1:
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    monkey_axes = [axes[0]]
    avg_ax = axes[1]
else:
    # Create a figure with proper subplot arrangement
    fig = plt.figure(figsize=(7 * n_monkeys, 12))
    
    # Individual monkey subplots in the top row
    monkey_axes = []
    for i in range(n_monkeys):
        ax = plt.subplot(2, n_monkeys, i + 1)
        monkey_axes.append(ax)
    
    # Average plot spanning the full bottom row
    avg_ax = plt.subplot(2, 1, 2)

# Store data for averaging across monkeys
all_roi_correlations = {roi_label: [] for roi_label in ['V1', 'V4', 'IT']}

# Individual monkey plots
for m, monkey in enumerate(args.monkey):
    ax = monkey_axes[m]
    
    roi_assignments, roi_labels = roi_data[m]
    
    # Calculate region-averaged correlations
    for roi_idx, roi_label in enumerate(roi_labels):
        # Find electrodes belonging to this region
        region_electrodes = np.where(roi_assignments == roi_idx)[0]
        
        if len(region_electrodes) > 0:
            # Average correlation across electrodes in this region
            region_correlations = np.mean(correlation_results[m][:, region_electrodes], axis=1)
            
            # Store for cross-monkey averaging
            all_roi_correlations[roi_label].append(region_correlations)
            
            # Plot region time course
            ax.plot(times, region_correlations, 
                   color=roi_colors[roi_label], 
                   linewidth=3,
                   label=f'{roi_label} (n={len(region_electrodes)})')
    
    # Plot chance and stimulus onset lines
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # x-axis parameters
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-100, -50, 0, 50, 100, 150, 200]
    ax.set_xticks(xticks)
    ax.set_xlim(left=times[0], right=times[-1])
    
    # y-axis parameters
    ax.set_ylabel('Pearson\'s r', fontsize=fontsize)
    ax.set_ylim(bottom=-0.1, top=0.8)
    ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    
    # Title and legend
    ax.set_title(f'{monkey} - Brain Region Comparison', fontsize=fontsize+2, fontweight='bold')
    ax.legend(loc='upper right', fontsize=fontsize)

# Cross-monkey average plot
# Plot averaged correlations across monkeys for each ROI
for roi_label in ['V1', 'V4', 'IT']:
    if all_roi_correlations[roi_label]:  # Check if we have data for this ROI
        roi_data_array = np.array(all_roi_correlations[roi_label])
        
        # Calculate mean and std across monkeys
        roi_mean = np.mean(roi_data_array, axis=0)
        roi_std = np.std(roi_data_array, axis=0)
        
        # Plot mean line
        avg_ax.plot(times, roi_mean, 
                   color=roi_colors[roi_label], 
                   linewidth=3,
                   label=f'{roi_label}')
        
        # Add error bars (standard deviation across monkeys)
        avg_ax.fill_between(times, roi_mean - roi_std, roi_mean + roi_std,
                           color=roi_colors[roi_label], alpha=0.2)

# Plot chance and stimulus onset lines for average plot
avg_ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
avg_ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# x-axis parameters for average plot
avg_ax.set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-100, -50, 0, 50, 100, 150, 200]
avg_ax.set_xticks(xticks)
avg_ax.set_xlim(left=times[0], right=times[-1])

# y-axis parameters for average plot
avg_ax.set_ylabel('Pearson\'s r', fontsize=fontsize)
avg_ax.set_ylim(bottom=-0.1, top=0.8)
avg_ax.set_yticks(np.arange(-0.1, 0.9, 0.1))

# Title and legend for average plot
avg_ax.set_title('Average Across Monkeys - Brain Region Comparison', fontsize=fontsize+2, fontweight='bold')
avg_ax.legend(loc='upper right', fontsize=fontsize)

# Adjust layout and save
plt.tight_layout()

# Create save directory
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-spike',
    'train_dataset-tvsd_monkey', f'model-{args.model}', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Save the figure
save_name = f'encoding_accuracy_roi_model-{args.regression}_{cls_suffix}_{args.model}'
fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight', format='png')
fig.savefig(os.path.join(save_dir, f'{save_name}.jpg'), dpi=300, bbox_inches='tight', format='jpeg')

print(f"Plot saved to: {save_dir}/{save_name}.png and {save_dir}/{save_name}.jpg")
plt.show()