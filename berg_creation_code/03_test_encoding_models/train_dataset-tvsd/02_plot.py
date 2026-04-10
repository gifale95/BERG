"""Plot the encoding models' prediction accuracy for the test stimuli by brain region.

Parameters
----------
monkeys : list
    List with all used TVSD monkeys.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG
train_split : str
    Which training split to plot (default: 'all_training_splits').
plot_noise_ceiling : bool
    Plot noise ceiling for each ROI.
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
parser.add_argument('--plot_noise_ceiling', required=True, choices=["True", "False"],
                   help="Plot noise ceiling for each ROI")
parser.add_argument('--berg_dir', required=True, type=str)
parser.add_argument('--train_split', type=str, default='all_training_splits',
                   choices=['all_training_splits', 'single_training_split_1', 'single_training_split_2', 'single_training_split_3', 'single_training_split_4'],
                   help='Which training split to plot')
args = parser.parse_args()

args.plot_noise_ceiling = args.plot_noise_ceiling == "True"



# =============================================================================
# Load the encoding models' encoding accuracy and metadata
# =============================================================================
correlation_results = []
roi_data = []
noise_ceiling_data = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-utah_array',
    'train_dataset-tvsd', 'model-vit_b_32', 'metadata')

for monkey in args.monkey:
    file_name = f'metadata_{monkey}.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name),
        allow_pickle=True).item()
    
    # Access correlation results from the specific split
    correlation_results.append(metadata['encoding_model'][args.train_split]['correlation_results'])
    times = metadata['utah_array']['times']
    roi_assignments = metadata['roi']['roi_assignments']
    roi_labels = metadata['roi']['roi_labels']
    # Noise ceiling is at the top level of encoding_model (shared across splits)
    noise_ceiling = metadata['encoding_model']['noise_ceiling']
    roi_data.append((roi_assignments, roi_labels))
    noise_ceiling_data.append(noise_ceiling)

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
    'V1': '#e74c3c',
    'V4': '#3498db',
    'IT': '#2ecc71'
}


# =============================================================================
# Plot the encoding accuracy results by brain region
# =============================================================================
n_monkeys = len(args.monkey)

if n_monkeys == 1:
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    monkey_axes = [axes[0]]
    avg_ax = axes[1]
else:
    fig = plt.figure(figsize=(7 * n_monkeys, 12))
    
    monkey_axes = []
    for i in range(n_monkeys):
        ax = plt.subplot(2, n_monkeys, i + 1)
        monkey_axes.append(ax)
    
    avg_ax = plt.subplot(2, 1, 2)

all_roi_correlations = {roi_label: [] for roi_label in ['V1', 'V4', 'IT']}
all_roi_noise_ceilings = {roi_label: [] for roi_label in ['V1', 'V4', 'IT']}

for m, monkey in enumerate(args.monkey):
    ax = monkey_axes[m]
    
    roi_assignments, roi_labels = roi_data[m]
    noise_ceiling = noise_ceiling_data[m]
    
    for roi_idx, roi_label in enumerate(roi_labels):
        region_electrodes = np.where(roi_assignments == roi_idx)[0]
        
        # Filter for electrodes where noise ceiling > 0
        valid_mask = noise_ceiling[region_electrodes, :].max(axis=1) > 0.1
        region_electrodes = region_electrodes[valid_mask]
            
        if len(region_electrodes) > 0:
            # Clip negative correlations to 0 before squaring
            region_correlations = np.clip(correlation_results[m][region_electrodes, :], 0, None)
            region_correlations_r2 = np.mean(region_correlations**2, axis=0)
            
            all_roi_correlations[roi_label].append(region_correlations_r2)
            
            ax.plot(times, region_correlations_r2, 
                   color=roi_colors[roi_label], 
                   linewidth=3,
                   label=f'{roi_label}')
            
            if args.plot_noise_ceiling:
                region_noise_ceiling = np.mean(noise_ceiling[region_electrodes, :], axis=0) / 100
                all_roi_noise_ceilings[roi_label].append(region_noise_ceiling)
                ax.plot(times, region_noise_ceiling, 
                       color=roi_colors[roi_label], 
                       linestyle='--', linewidth=2, alpha=0.5)
    
    if args.plot_noise_ceiling:
        ax.plot([], [], color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Noise Ceiling')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    xticks = [-100, -50, 0, 50, 100, 150, 200]
    ax.set_xticks(xticks)
    ax.set_xlim(left=times[0], right=times[-1])
    
    ax.set_ylabel('r²', fontsize=fontsize)
    ax.set_ylim(bottom=-0.05, top=0.9)
    ax.set_yticks(np.arange(0, 1.0, 0.1))
    
    ax.set_title(f'{monkey} - Brain Region Comparison', fontsize=fontsize+2, fontweight='bold')
    
    # Only show legend on first subplot
    if m == 0:
        ax.legend(loc='upper left', fontsize=fontsize)

for roi_label in ['V1', 'V4', 'IT']:
    if all_roi_correlations[roi_label]:
        roi_data_array = np.array(all_roi_correlations[roi_label])
        roi_mean = np.mean(roi_data_array, axis=0)
        
        avg_ax.plot(times, roi_mean, 
                   color=roi_colors[roi_label], 
                   linewidth=3,
                   label=f'{roi_label}')
        
        if args.plot_noise_ceiling and all_roi_noise_ceilings[roi_label]:
            noise_ceiling_array = np.array(all_roi_noise_ceilings[roi_label])
            noise_ceiling_mean = np.mean(noise_ceiling_array, axis=0)
            avg_ax.plot(times, noise_ceiling_mean,
                       color=roi_colors[roi_label],
                       linestyle='--', linewidth=2, alpha=0.5)

if args.plot_noise_ceiling:
    avg_ax.plot([], [], color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Noise Ceiling')

avg_ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
avg_ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

avg_ax.set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [-100, -50, 0, 50, 100, 150, 200]
avg_ax.set_xticks(xticks)
avg_ax.set_xlim(left=times[0], right=times[-1])

avg_ax.set_ylabel('r²', fontsize=fontsize)
avg_ax.set_ylim(bottom=-0.05, top=0.9)
avg_ax.set_yticks(np.arange(0, 1.0, 0.1))


avg_ax.set_title('Average Across Monkeys - Brain Region Comparison', fontsize=fontsize+2, fontweight='bold')
avg_ax.legend(loc='upper left', fontsize=fontsize)

plt.tight_layout()

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-utah_array',
    'train_dataset-tvsd', 'model-vit_b_32', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_name = f'encoding_accuracy_{args.train_split}'
fig.savefig(os.path.join(save_dir, f'{save_name}.jpg'), dpi=300, bbox_inches='tight', format='jpeg')

print(f"Plot saved to: {save_dir}/{save_name}.jpg")
plt.show()