"""Plot the encoding models' prediction accuracy for the test stimuli by brain region.

Parameters
----------
subject : list
    List with all used subjects (e.g., ['P1', 'P2', 'P3', 'P4']).
model : str
    Name of the used encoding model.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
only_cls : str
    If we should only use CLS token or all patches ('True' or 'False').
regression : str
    Type of regression used ('ridge' or 'linear').

Example usage:
python berg_creation_code/03_test_encoding_models/train_dataset-things_meg_1/02_plot.py \
    --subject P1 P2 P3 P4 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --only_cls True \
    --regression ridge \
    --model clip.vit_b_32
    
    
python berg_creation_code/03_test_encoding_models/train_dataset-things_meg_1/02_plot.py \
    --subject P1 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --only_cls True \
    --regression ridge \
    --model clip.vit_b_32
"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', nargs='+', default=['P1'],
    help='List of subjects to analyze (e.g., --subject P1 P2 P3 P4)')
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
region_data = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', f'model-{args.model}', 'metadata')

for subject in args.subject:
    # Load all data from single metadata file
    file_name = f'metadata_{args.regression}_{cls_suffix}_{subject}.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name),
        allow_pickle=True).item()
    
    correlation_results.append(metadata['correlation_results'])
    times = metadata['times']
    sensor_regions = metadata['sensor_regions']
    region_data.append(sensor_regions)

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


region_colors = {
    'Frontal': '#3498db',    # Blue
    'Central': '#2ecc71',    # Green
    'Parietal': '#f39c12',   # Orange
    'Temporal': '#e74c3c',   # Red
    'Occipital': '#9b59b6'   # Purple
}

region_order = ['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital']


# =============================================================================
# Plot the encoding accuracy results by brain region
# =============================================================================
n_subjects = len(args.subject)

# Create figure with proper layout
if n_subjects == 1:
    # Single subject: keep as is
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    axes = [ax]
    avg_ax = None
elif n_subjects == 4:
    # All 4 subjects: 2x2 grid on top + grand average on bottom
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.67], hspace=0.3)
    
    # Create 2x2 grid for individual subjects
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]
    
    # Create grand average plot spanning bottom
    avg_ax = fig.add_subplot(gs[2, :])
else:
    # 2-3 subjects: just 2x2 grid without average
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    avg_ax = None

# Individual subject plots
for s, subject in enumerate(args.subject):
    ax = axes[s]
    
    sensor_regions = region_data[s]
    
    # Calculate region-averaged correlations
    for region_label in region_order:
        # Find sensors belonging to this region
        region_sensors = np.where(sensor_regions == region_label)[0]
        
        if len(region_sensors) > 0:
            # Average correlation across sensors in this region
            # correlation_results shape: (n_subjects, n_channels, n_timepoints)
            region_correlations = np.mean(correlation_results[s, region_sensors, :], axis=0)
            
            # Plot region time course
            ax.plot(times * 1000, region_correlations,  # Convert times to ms
                   color=region_colors[region_label], 
                   linewidth=3,
                   label=f'{region_label} (n={len(region_sensors)})')
    
    # Plot chance and stimulus onset lines
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # x-axis parameters
    ax.set_xlabel('Time (ms)', fontsize=fontsize)
    ax.set_xlim(left=times[0] * 1000, right=times[-1] * 1000)
    
    # y-axis parameters
    ax.set_ylabel('Pearson\'s r', fontsize=fontsize)
    ax.set_ylim(bottom=-0.1, top=0.8)
    ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    
    # Title and legend
    ax.set_title(f'{subject} - Brain Region Comparison', fontsize=fontsize+2, fontweight='bold')
    ax.legend(loc='upper right', fontsize=fontsize-1)

# Hide unused subplots if we have fewer than 4 subjects
if n_subjects < 4 and avg_ax is None:
    for i in range(n_subjects, 4):
        axes[i].axis('off')

# Grand average plot (only when all 4 subjects are shown)
if avg_ax is not None:
    # Calculate grand average across subjects for each region
    for region_label in region_order:
        region_avg_correlations = []
        total_sensors = 0
        
        for s in range(n_subjects):
            sensor_regions = region_data[s]
            region_sensors = np.where(sensor_regions == region_label)[0]
            
            if len(region_sensors) > 0:
                # Average correlation across sensors in this region for this subject
                region_correlations = np.mean(correlation_results[s, region_sensors, :], axis=0)
                region_avg_correlations.append(region_correlations)
                total_sensors += len(region_sensors)
        
        if len(region_avg_correlations) > 0:
            # Average across subjects
            grand_avg = np.mean(region_avg_correlations, axis=0)
            
            # Plot grand average
            avg_ax.plot(times * 1000, grand_avg,
                       color=region_colors[region_label],
                       linewidth=3,
                       label=f'{region_label} (n={total_sensors})')
    
    # Plot chance and stimulus onset lines
    avg_ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    avg_ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # x-axis parameters
    avg_ax.set_xlabel('Time (ms)', fontsize=fontsize)
    avg_ax.set_xlim(left=times[0] * 1000, right=times[-1] * 1000)
    
    # y-axis parameters
    avg_ax.set_ylabel('Pearson\'s r', fontsize=fontsize)
    avg_ax.set_ylim(bottom=-0.1, top=0.8)
    avg_ax.set_yticks(np.arange(-0.1, 0.9, 0.1))
    
    # Title and legend
    avg_ax.set_title('Grand Average Across All Subjects - Brain Region Comparison', 
                     fontsize=fontsize+2, fontweight='bold')
    avg_ax.legend(loc='upper right', fontsize=fontsize-1)

# Adjust layout
if n_subjects == 1 or avg_ax is None:
    plt.tight_layout()

# Create save directory
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', f'model-{args.model}', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Save the figure
subjects_str = '_'.join(args.subject)
save_name = f'encoding_accuracy_region_model-{args.regression}_{cls_suffix}_{args.model}_{subjects_str}'
fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight', format='png')
fig.savefig(os.path.join(save_dir, f'{save_name}.jpg'), dpi=300, bbox_inches='tight', format='jpeg')

print(f"Plot saved to: {save_dir}/{save_name}.png and {save_dir}/{save_name}.jpg")
plt.show()