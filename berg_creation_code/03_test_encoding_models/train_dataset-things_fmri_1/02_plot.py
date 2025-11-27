"""Plot the encoding models' prediction accuracy for the test stimuli by ROI.
Multi-subject version with grid layout.

Parameters
----------
subjects : list of str
    List of subjects to analyze (e.g., 'sub-01 sub-02 sub-03').
model : str
    Name of the used encoding model.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
only_cls : str
    If we should only use CLS token or all patches ('True' or 'False').
regression : str
    Type of regression used ('ridge' or 'linear').

Example usage:
python berg_creation_code/03_test_encoding_models/train_dataset-things_fmri_1/02_plot.py \
    --subjects sub-01 sub-02 sub-03 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --only_cls True \
    --regression linear \
    --model clip.vit_b_32
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
parser.add_argument('--subjects', type=str, nargs='+', required=True,
                   help="List of subject IDs (e.g., 'sub-01 sub-02 sub-03')")
parser.add_argument('--model', required=True, choices=["vit_b_32", "clip.vit_b_32", "huze"],
                   help="Selecting which model to use")
parser.add_argument('--only_cls', required=True, choices=["True", "False"],
                    help='If we should only use CLS token or all patches')
parser.add_argument('--regression', required=True, choices=["ridge", "linear"],
                   help="Select type of regression")
parser.add_argument('--berg_dir', required=True, type=str)
args = parser.parse_args()

args.only_cls = args.only_cls == "True"
cls_suffix = 'cls' if args.only_cls else 'all'

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
        'train_dataset-things_fmri_1', f'model-{args.model}', 'metadata')
    file_name = f'metadata_{args.regression}_{cls_suffix}_{subject}.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name), allow_pickle=True).item()
    
    correlation_results = metadata['encoding_model']['correlation_results']
    noise_ceiling_testset = metadata['encoding_model']['noise_ceiling_testset']
    noise_ceiling_correlation = np.sqrt(noise_ceiling_testset / 100)
    
    # Load ROI indices from preprocessed metadata
    preprocessed_dir = os.path.join(args.berg_dir, 'model_training_datasets',
                                    'train_dataset-things_fmri_1')
    metadata_file = os.path.join(preprocessed_dir, f'fmri_{subject}_metadata.npy')
    preprocessed_metadata = np.load(metadata_file, allow_pickle=True).item()
    
    # ROIs are now directly in the 'roi' dictionary
    roi_indices_dict = preprocessed_metadata['roi']
    
    # Compute ROI-averaged correlations and noise ceilings
    roi_correlations = []
    roi_noise_ceilings = []
    roi_n_voxels = []
    roi_labels = []
    
    for roi_name in all_rois:
        roi_key = roi_name
        
        if roi_key in roi_indices_dict:
            roi_indices = roi_indices_dict[roi_key]
            
            if len(roi_indices) > 0:
                roi_corr = np.mean(correlation_results[roi_indices])
                roi_nc = np.mean(noise_ceiling_correlation[roi_indices])
                
                roi_correlations.append(roi_corr)
                roi_noise_ceilings.append(roi_nc)
                roi_n_voxels.append(len(roi_indices))
                roi_labels.append(roi_name)
    
    # Store data for this subject
    all_subject_data[subject] = {
        'roi_correlations': np.array(roi_correlations),
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
all_corrs_matrix = np.full((n_subjects, len(all_rois)), np.nan)
all_ncs_matrix = np.full((n_subjects, len(all_rois)), np.nan)

# Fill in the data for each subject
for subj_idx, subject in enumerate(args.subjects):
    subj_roi_labels = all_subject_data[subject]['roi_labels']
    subj_corrs = all_subject_data[subject]['roi_correlations']
    subj_ncs = all_subject_data[subject]['roi_noise_ceilings']
    
    # Map subject's ROIs to the master ROI list
    for roi_idx, roi_name in enumerate(subj_roi_labels):
        master_roi_idx = all_rois.index(roi_name)
        all_corrs_matrix[subj_idx, master_roi_idx] = subj_corrs[roi_idx]
        all_ncs_matrix[subj_idx, master_roi_idx] = subj_ncs[roi_idx]

# Compute mean across subjects (ignoring NaNs)
mean_corrs = np.nanmean(all_corrs_matrix, axis=0)
mean_ncs = np.nanmean(all_ncs_matrix, axis=0)


# =============================================================================
# Prepare data for grid plotting
# =============================================================================
# Convert correlations to R² (explained variance)
all_r2_matrix = all_corrs_matrix ** 2
mean_r2 = mean_corrs ** 2

# Noise ceiling is already in R² scale, square it to match
all_ncs_matrix_r2 = all_ncs_matrix ** 2
mean_ncs_r2 = mean_ncs ** 2


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
colors = [(170/255, 118/255, 186/255)]


# =============================================================================
# Calculate global y-limit
# =============================================================================
# Find maximum value across all R² data and noise ceilings
max_r2 = np.nanmax(all_r2_matrix)
max_nc = np.nanmax(all_ncs_matrix_r2)
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
    roi_nc_r2 = all_ncs_matrix_r2[:, r]
    
    # Plot the R² bars for each subject
    axs[r].bar(x, roi_r2, width=width, color=colors[0])
    
    # Plot the mean R² across subjects (dashed line)
    y_mean = mean_r2[r]
    if not np.isnan(y_mean):
        axs[r].plot([min(x), max(x)], [y_mean, y_mean], '--', color='k', 
                    linewidth=2, alpha=0.4, label='Subjects mean')
    
    # Plot the mean noise ceiling (solid line)
    y_nc = mean_ncs_r2[r]
    if not np.isnan(y_nc):
        axs[r].plot([min(x), max(x)], [y_nc, y_nc], '-', color='k', 
                    linewidth=2, alpha=0.6)
    
    # y-axis label on leftmost column
    if r % n_cols == 0:
        axs[r].set_ylabel('R² (Explained Variance)', fontsize=fontsize)
    yticks = np.arange(0, y_max_global + 0.1, 0.2)
    ylabels = [f'{tick:.1f}' for tick in yticks]
    axs[r].set_yticks(yticks)
    axs[r].set_yticklabels(ylabels)
    axs[r].set_ylim(bottom=0, top=y_max_global)
    
    # x-axis label and ticks on bottom of each column
    # A subplot is at the bottom of its column if there's no subplot below it
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

# Remove empty subplots
for r in range(n_rois, len(axs)):
    fig.delaxes(axs[r])

plt.tight_layout()


# =============================================================================
# Save the figure
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri_1', f'model-{args.model}', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

subject_str = '_'.join(args.subjects)
save_name = f'encoding_accuracy_roi_grid_{args.regression}_{cls_suffix}_{args.model}_{subject_str}'
fig.savefig(os.path.join(save_dir, f'{save_name}.svg'), bbox_inches='tight', transparent=True, format='svg')
fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight', format='png')

print(f"\nPlot saved to: {save_dir}/{save_name}.svg and {save_dir}/{save_name}.png")


# =============================================================================
# Print summary statistics
# =============================================================================
# Overall statistics (excluding NaN values)
valid_mean_corrs = mean_corrs[~np.isnan(mean_corrs)]
valid_mean_ncs = mean_ncs[~np.isnan(mean_ncs)]

print(f"\n{'='*80}")
print(f"SUMMARY STATISTICS - AVERAGE ACROSS SUBJECTS")
print(f"  Mean correlation: {valid_mean_corrs.mean():.4f} ± {valid_mean_corrs.std():.4f}")
print(f"  Mean R²: {(valid_mean_corrs**2).mean():.4f} ± {(valid_mean_corrs**2).std():.4f}")
print(f"  Mean noise ceiling: {valid_mean_ncs.mean():.4f} ± {valid_mean_ncs.std():.4f}")
print(f"  Total ROIs with data: {len(valid_mean_corrs)}/{len(all_rois)}")
print("="*80)