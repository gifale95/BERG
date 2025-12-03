"""Plot the encoding models' prediction accuracy for the test stimuli by brain region.

Parameters
----------
subject : list
    List with all used subjects (e.g., ['P1', 'P2', 'P3', 'P4']).
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).

Example usage:
python berg_creation_code/03_test_encoding_models/train_dataset-things_meg_1/02_plot.py \
    --subject P1 P2 P3 P4 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator'
    
python berg_creation_code/03_test_encoding_models/train_dataset-things_meg_1/02_plot.py \
    --subject P1 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator'
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
parser.add_argument('--berg_dir', required=True, type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding models' encoding accuracy and metadata
# =============================================================================
correlation_results = []
noise_ceiling_results = []
region_data = []

metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', 'model-vit_b_32', 'metadata')

for subject in args.subject:
    # Load all data from single metadata file
    file_name = f'metadata_{subject}.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name),
        allow_pickle=True).item()
    
    correlation_results.append(metadata['encoding_model']['correlation_results'])
    noise_ceiling_results.append(metadata['encoding_model']['noise_ceiling'])
    times = metadata['meg']['times']
    sensor_regions = metadata['sensors']['sensor_regions']
    region_data.append(sensor_regions)

correlation_results = np.asarray(correlation_results)
noise_ceiling_results = np.asarray(noise_ceiling_results)


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 12
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.width'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1.5
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
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.67], hspace=0.35, wspace=0.25)
    
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
    
    # Calculate region-averaged correlations and noise ceilings
    for region_label in region_order:
        # Find sensors belonging to this region
        region_sensors = np.where(sensor_regions == region_label)[0]
        
        if len(region_sensors) > 0:
            # Average correlation across sensors in this region
            # correlation_results shape: (n_subjects, n_channels, n_timepoints)
            region_correlations = np.mean(correlation_results[s, region_sensors, :], axis=0)
            
            # Set negative correlations to 0 before squaring
            region_correlations_clipped = np.clip(region_correlations, 0, None)
            
            # Square correlations to get r²
            region_r2 = region_correlations_clipped ** 2
            
            # Average noise ceiling across sensors in this region
            # Convert from r² percentage (0-100) to proportion (0-1)
            region_noise_ceiling_r2 = np.mean(noise_ceiling_results[s, region_sensors, :], axis=0)
            region_noise_ceiling = region_noise_ceiling_r2 / 100
            
            # Plot region time course (r²)
            ax.plot(times * 1000, region_r2,  # Convert times to ms
                   color=region_colors[region_label], 
                   linewidth=2.5,
                   label=f'{region_label} (n={len(region_sensors)})',
                   alpha=0.9)
            
            # Plot noise ceiling as subtle line
            ax.plot(times * 1000, region_noise_ceiling,
                   color=region_colors[region_label],
                   linestyle='--',
                   linewidth=1.2,
                   alpha=0.5)
    
    # Plot chance and stimulus onset lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # x-axis parameters
    ax.set_xlabel('Time (ms)', fontsize=fontsize+1, fontweight='bold')
    ax.set_xlim(left=times[0] * 1000, right=times[-1] * 1000)
    
    # y-axis parameters
    ax.set_ylabel('r² (Explained Variance)', fontsize=fontsize+1, fontweight='bold')
    ax.set_ylim(bottom=-0.05, top=1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Grid for better readability
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Title
    ax.set_title(f'{subject} - Brain Region Encoding Accuracy', 
                fontsize=fontsize+3, fontweight='bold', pad=15)

# Hide unused subplots if we have fewer than 4 subjects
if n_subjects < 4 and avg_ax is None:
    for i in range(n_subjects, 4):
        axes[i].axis('off')

# Grand average plot (only when all 4 subjects are shown)
if avg_ax is not None:
    # Calculate grand average across subjects for each region
    for region_label in region_order:
        region_avg_correlations = []
        region_avg_noise_ceilings = []
        total_sensors = 0
        
        for s in range(n_subjects):
            sensor_regions = region_data[s]
            region_sensors = np.where(sensor_regions == region_label)[0]
            
            if len(region_sensors) > 0:
                # Average correlation across sensors in this region for this subject
                region_correlations = np.mean(correlation_results[s, region_sensors, :], axis=0)
                region_avg_correlations.append(region_correlations)
                
                # Average noise ceiling across sensors in this region for this subject
                region_noise_ceiling_r2 = np.mean(noise_ceiling_results[s, region_sensors, :], axis=0)
                region_avg_noise_ceilings.append(region_noise_ceiling_r2)
                
                total_sensors += len(region_sensors)
        
        if len(region_avg_correlations) > 0:
            # Average across subjects
            grand_avg_corr = np.mean(region_avg_correlations, axis=0)
            grand_avg_nc_r2 = np.mean(region_avg_noise_ceilings, axis=0)
            
            # Set negative correlations to 0 before squaring
            grand_avg_corr_clipped = np.clip(grand_avg_corr, 0, None)
            
            # Square correlations to get r²
            grand_avg_r2 = grand_avg_corr_clipped ** 2
            
            # Convert noise ceiling from percentage to proportion
            grand_avg_nc = grand_avg_nc_r2 / 100
            
            # Plot grand average (r²)
            avg_ax.plot(times * 1000, grand_avg_r2,
                       color=region_colors[region_label],
                       linewidth=3,
                       label=f'{region_label} (n={total_sensors})',
                       alpha=0.9)
            
            # Plot noise ceiling as subtle line
            avg_ax.plot(times * 1000, grand_avg_nc,
                       color=region_colors[region_label],
                       linestyle='--',
                       linewidth=1.2,
                       alpha=0.5)
    
    # Plot chance and stimulus onset lines
    avg_ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    avg_ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # x-axis parameters
    avg_ax.set_xlabel('Time (ms)', fontsize=fontsize+2, fontweight='bold')
    avg_ax.set_xlim(left=times[0] * 1000, right=times[-1] * 1000)
    
    # y-axis parameters
    avg_ax.set_ylabel('r² (Explained Variance)', fontsize=fontsize+2, fontweight='bold')
    avg_ax.set_ylim(bottom=-0.05, top=1.0)
    avg_ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Grid for better readability
    avg_ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    avg_ax.set_axisbelow(True)
    
    # Title
    avg_ax.set_title('Grand Average Across All Subjects - Brain Region Encoding Accuracy', 
                     fontsize=fontsize+3, fontweight='bold', pad=15)

# Add single legend in upper left corner
# Create custom legend entries
from matplotlib.lines import Line2D
legend_elements = []

# Add region lines
for region_label in region_order:
    legend_elements.append(Line2D([0], [0], color=region_colors[region_label], 
                                  linewidth=2.5, label=region_label))

# Add noise ceiling entry
legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', 
                             linewidth=1.2, alpha=0.5, label='Noise Ceiling'))

# Place legend on first subplot (upper left)
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=fontsize, 
              framealpha=0.95, edgecolor='gray', fancybox=True)

# Adjust layout
if n_subjects == 1 or avg_ax is None:
    plt.tight_layout()

# Create save directory
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-meg',
    'train_dataset-things_meg_1', 'model-vit_b_32', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Save the figure
save_name = 'encoding_accuracy'
fig.savefig(os.path.join(save_dir, f'{save_name}.jpg'), dpi=300, bbox_inches='tight', format='jpeg')

plt.show()