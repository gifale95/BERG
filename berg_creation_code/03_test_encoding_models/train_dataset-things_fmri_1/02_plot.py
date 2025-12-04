"""Plot the encoding models' prediction accuracy for the test stimuli by ROI.
Multi-subject version with grid layout.

Parameters
----------
subjects : list of str
    List of subjects to analyze (e.g., 'sub-01 sub-02 sub-03').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=str, nargs='+', required=True,
                   help="List of subject IDs (e.g., 'sub-01 sub-02 sub-03')")
parser.add_argument('--berg_dir', required=True, type=str)
args = parser.parse_args()

n_subjects = len(args.subjects)
print(f"Processing {n_subjects} subjects: {', '.join(args.subjects)}")


# =============================================================================
# Define ROI groups and order
# =============================================================================
roi_groups = {
    'Early Visual': ['V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2'],
    'Lateral Occipital': ['LO1_prf', 'LO2_prf', 'TO1', 'TO2', 'V3a', 'V3b'],
    'Face-selective': ['lFFA', 'rFFA', 'lOFA', 'rOFA'],
    'Body-selective': ['lEBA', 'rEBA'],
    'Scene-selective': ['lPPA', 'rPPA', 'lRSC', 'rRSC', 'lTOS', 'rTOS'],
    'Object-selective': ['lLOC', 'rLOC', 'IT'],
    'Motion/Social': ['lSTS', 'rSTS']
}

# Flatten ROI list in order
all_rois = []
group_boundaries = [0]
for group_name, rois in roi_groups.items():
    all_rois.extend(rois)
    group_boundaries.append(len(all_rois))


# =============================================================================
# Load data for all subjects
# =============================================================================
all_subject_data = {}

for subject in args.subjects:
    # Load encoding model metadata
    metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
        'train_dataset-things_fmri_1', 'model-vit_b_32', 'metadata')
    file_name = f'metadata_{subject}.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name), allow_pickle=True).item()
    
    # Get correlation results (r values) and noise ceiling (R² percentage)
    correlation_results = metadata['encoding_model']['correlation_results']
    noise_ceiling_testset = metadata['encoding_model']['noise_ceiling_testset']
    
    # Convert noise ceiling from percentage to R² (0-1 scale)
    noise_ceiling_r2 = noise_ceiling_testset / 100
    
    # Load ROI indices from preprocessed metadata
    preprocessed_dir = os.path.join(args.berg_dir, 'model_training_datasets',
                                    'train_dataset-things_fmri_1')
    metadata_file = os.path.join(preprocessed_dir, f'fmri_{subject}_metadata.npy')
    preprocessed_metadata = np.load(metadata_file, allow_pickle=True).item()
    
    # ROIs are now directly in the 'roi' dictionary
    roi_indices_dict = preprocessed_metadata['roi']
    
    # Compute ROI-averaged R² and noise ceilings
    roi_r2_values = []
    roi_noise_ceilings = []
    roi_n_voxels = []
    roi_labels = []
    
    for roi_name in all_rois:
        roi_key = roi_name
        
        if roi_key in roi_indices_dict:
            roi_indices = roi_indices_dict[roi_key]
            
            # Filter for voxels where noise ceiling > 0
            roi_indices = roi_indices[noise_ceiling_r2[roi_indices] > 0]
            
            if len(roi_indices) > 0:
                # Clip negative correlations to 0, then square to get R²
                roi_corr_clipped = np.clip(correlation_results[roi_indices], 0, None)
                roi_r2 = np.mean(roi_corr_clipped ** 2)
                
                # Get mean noise ceiling R² for this ROI
                roi_nc = np.mean(noise_ceiling_r2[roi_indices])
                
                roi_r2_values.append(roi_r2)
                roi_noise_ceilings.append(roi_nc)
                roi_n_voxels.append(len(roi_indices))
                roi_labels.append(roi_name)
    
    # Store data for this subject
    all_subject_data[subject] = {
        'roi_r2': np.array(roi_r2_values),
        'roi_noise_ceilings': np.array(roi_noise_ceilings),
        'roi_n_voxels': np.array(roi_n_voxels),
        'roi_labels': roi_labels
    }

print(f"\nLoaded data for {n_subjects} subjects")


# =============================================================================
# Compute average across subjects
# =============================================================================
# Create a matrix for all ROIs across all subjects (using NaN for missing ROIs)
# Shape: (n_subjects, n_rois)
all_r2_matrix = np.full((n_subjects, len(all_rois)), np.nan)
all_ncs_matrix = np.full((n_subjects, len(all_rois)), np.nan)

# Fill in the data for each subject
for subj_idx, subject in enumerate(args.subjects):
    subj_roi_labels = all_subject_data[subject]['roi_labels']
    subj_r2 = all_subject_data[subject]['roi_r2']
    subj_ncs = all_subject_data[subject]['roi_noise_ceilings']
    
    # Map subject's ROIs to the master ROI list
    for roi_idx, roi_name in enumerate(subj_roi_labels):
        master_roi_idx = all_rois.index(roi_name)
        all_r2_matrix[subj_idx, master_roi_idx] = subj_r2[roi_idx]
        all_ncs_matrix[subj_idx, master_roi_idx] = subj_ncs[roi_idx]

# Compute mean across subjects (ignoring NaNs)
mean_r2 = np.nanmean(all_r2_matrix, axis=0)
mean_ncs = np.nanmean(all_ncs_matrix, axis=0)


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

# Colors
bar_color = (170/255, 118/255, 186/255)  # Purple for bars
mean_color = '#2E86AB'  # Blue for mean line
nc_color = '#A23B72'  # Dark rose for noise ceiling


# =============================================================================
# Calculate global y-limit
# =============================================================================
# Find maximum value across all R² data and noise ceilings
max_r2 = np.nanmax(all_r2_matrix)
max_nc = np.nanmax(all_ncs_matrix)
global_max = max(max_r2, max_nc)

# Add 5% padding above the maximum
y_max_global = global_max * 1.05


# =============================================================================
# Create multi-panel grid plot
# =============================================================================
n_rois = len(all_rois)
n_cols = 7
n_rows = int(np.ceil(n_rois / n_cols))

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=True, figsize=(21, n_rows*3))
axs = np.reshape(axs, (-1))

x = np.arange(n_subjects)
width = 0.4

for r, roi_name in enumerate(all_rois):
    # Get R² values for this ROI across subjects
    roi_r2 = all_r2_matrix[:, r]
    roi_nc = all_ncs_matrix[:, r]
    
    # Plot the R² bars for each subject
    axs[r].bar(x, roi_r2, width=width, color=bar_color)
    
    # Plot individual noise ceiling lines above each bar
    for subj_idx in range(n_subjects):
        if not np.isnan(roi_nc[subj_idx]):
            x_pos = x[subj_idx]
            y_nc = roi_nc[subj_idx]
            axs[r].plot([x_pos - width/2, x_pos + width/2], [y_nc, y_nc], 
                       '-', color=nc_color, linewidth=2.5, alpha=0.8)
    
    # Plot the mean R² across subjects (dashed line)
    y_mean = mean_r2[r]
    if not np.isnan(y_mean):
        axs[r].plot([min(x) - 0.5, max(x) + 0.5], [y_mean, y_mean], '--', 
                   color=mean_color, linewidth=2, alpha=0.7)
    
    # y-axis label on leftmost column
    if r % n_cols == 0:
        axs[r].set_ylabel('R² (Explained Variance)', fontsize=fontsize)
    
    # Create reasonable y-ticks based on R² scale (0-1)
    yticks = np.arange(0, 1.0, 0.1) 
    ylabels = [f'{tick:.1f}' for tick in yticks]
    axs[r].set_yticks(yticks)
    axs[r].set_yticklabels(ylabels)
    axs[r].set_ylim(bottom=0, top=y_max_global)
    
    # x-axis label and ticks on bottom of each column
    col_idx = r % n_cols
    subplot_below_idx = r + n_cols
    is_bottom_of_column = subplot_below_idx >= n_rois
    
    if is_bottom_of_column:
        axs[r].set_xlabel('Subjects', fontsize=fontsize)
        xticks = x
        xlabels = [str(i+1) for i in range(n_subjects)]
        axs[r].set_xticks(xticks)
        axs[r].set_xticklabels(xlabels, fontsize=fontsize)
    else:
        # Hide x-tick labels for non-bottom subplots
        axs[r].set_xticks(x)
        axs[r].set_xticklabels([])
    
    # Title
    axs[r].set_title(roi_name, fontsize=fontsize)

# =============================================================================
# Add legend in the empty subplot after the last ROI
# =============================================================================
# Find which subplot should contain the legend (first empty one after last ROI)
legend_subplot_idx = n_rois

if legend_subplot_idx < len(axs):
    # Turn off axis for legend subplot
    axs[legend_subplot_idx].axis('off')
    
    # Create legend elements
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor=bar_color, label='Model R²'),
        Line2D([0], [0], color=mean_color, linewidth=2, linestyle='--', 
               alpha=0.7, label='Mean Model R²'),
        Line2D([0], [0], color=nc_color, linewidth=2.5, linestyle='-', 
               alpha=0.8, label='Noise Ceiling')
    ]
    
    # Place legend in the center of the subplot
    axs[legend_subplot_idx].legend(handles=legend_elements, loc='center', 
                                   fontsize=fontsize, frameon=False)

# Remove remaining empty subplots
for r in range(legend_subplot_idx + 1, len(axs)):
    fig.delaxes(axs[r])

plt.tight_layout()


# =============================================================================
# Save the figure
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri_1', 'model-vit_b_32', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_name = 'encoding_accuracy'
fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight', format='jpg')