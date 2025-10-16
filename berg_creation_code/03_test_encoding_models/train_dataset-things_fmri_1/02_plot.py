"""Plot the encoding models' prediction accuracy for the test stimuli by ROI.
Multi-subject version with average plot.

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
    print(f"\n{'='*80}")
    print(f"Loading data for {subject}")
    print('='*80)
    
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
    
    roi_indices_dict = {}
    for key in preprocessed_metadata['fmri'].keys():
        if key.startswith('roi_'):
            roi_indices_dict[key] = preprocessed_metadata['fmri'][key]
    
    # Compute ROI-averaged correlations and noise ceilings
    roi_correlations = []
    roi_noise_ceilings = []
    roi_n_voxels = []
    roi_labels = []
    
    print(f"{'ROI':<15} {'N Voxels':<12} {'Model Corr':<15} {'Noise Ceiling':<15} {'Gap':<10}")
    print("-"*80)
    
    for roi_name in all_rois:
        roi_key = f'roi_{roi_name}'
        
        if roi_key in roi_indices_dict:
            roi_indices = roi_indices_dict[roi_key]
            
            if len(roi_indices) > 0:
                roi_corr = np.mean(correlation_results[roi_indices])
                roi_nc = np.mean(noise_ceiling_correlation[roi_indices])
                gap = roi_nc - roi_corr
                
                roi_correlations.append(roi_corr)
                roi_noise_ceilings.append(roi_nc)
                roi_n_voxels.append(len(roi_indices))
                roi_labels.append(roi_name)
                
                print(f"{roi_name:<15} {len(roi_indices):<12} {roi_corr:<15.4f} {roi_nc:<15.4f} {gap:<10.4f}")
    
    # Store data for this subject
    all_subject_data[subject] = {
        'roi_correlations': np.array(roi_correlations),
        'roi_noise_ceilings': np.array(roi_noise_ceilings),
        'roi_n_voxels': np.array(roi_n_voxels),
        'roi_labels': roi_labels
    }


# =============================================================================
# Compute average across subjects
# =============================================================================
print(f"\n{'='*80}")
print("Computing average across subjects")
print('='*80)

# Stack correlations and noise ceilings from all subjects
all_corrs = np.stack([all_subject_data[s]['roi_correlations'] for s in args.subjects])
all_ncs = np.stack([all_subject_data[s]['roi_noise_ceilings'] for s in args.subjects])

# Compute mean and SEM
mean_corrs = np.mean(all_corrs, axis=0)
sem_corrs = np.std(all_corrs, axis=0) / np.sqrt(n_subjects)

mean_ncs = np.mean(all_ncs, axis=0)
sem_ncs = np.std(all_ncs, axis=0) / np.sqrt(n_subjects)

# Use the first subject's ROI labels (they should all be the same)
roi_labels = all_subject_data[args.subjects[0]]['roi_labels']

print(f"Mean correlation across subjects: {mean_corrs.mean():.4f}")
print(f"Mean noise ceiling across subjects: {mean_ncs.mean():.4f}")


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 12
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize-1)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = False
plt.rcParams["text.usetex"] = False


# =============================================================================
# Create the multi-panel plot
# =============================================================================
n_rows = n_subjects + 1  # Individual subjects + average
fig, axes = plt.subplots(n_rows, 1, figsize=(18, 7 * n_rows))

# Ensure axes is always a list
if n_rows == 1:
    axes = [axes]

x_positions = np.arange(len(roi_labels))


# Plot individual subjects
for idx, subject in enumerate(args.subjects):
    ax = axes[idx]
    data = all_subject_data[subject]
    
    # Plot bars for model correlations
    bars = ax.bar(x_positions, data['roi_correlations'], color='#3498db', alpha=0.8,
                  label='Model Correlation', width=0.7)
    
    # Plot noise ceiling as horizontal lines per ROI
    for i, (x, nc) in enumerate(zip(x_positions, data['roi_noise_ceilings'])):
        ax.plot([x-0.35, x+0.35], [nc, nc], color='#e74c3c', linewidth=3, alpha=0.8)
    
    # Add one legend entry for noise ceiling
    ax.plot([], [], color='#e74c3c', linewidth=3, alpha=0.8, label='Noise Ceiling (testset)')
    
    # Add vertical lines to separate groups
    current_pos = 0
    for i, group_name in enumerate(roi_groups.keys()):
        group_size = len(roi_groups[group_name])
        
        # Add group label
        group_center = current_pos + group_size / 2 - 0.5
        ax.text(group_center, -0.15, group_name, ha='center', va='top',
                fontsize=fontsize, fontweight='bold', transform=ax.get_xaxis_transform())
        
        # Add separator line (except after last group)
        if i < len(roi_groups) - 1:
            separator_pos = current_pos + group_size - 0.5
            ax.axvline(separator_pos, color='black', linewidth=1.5, alpha=0.3, linestyle='--')
        
        current_pos += group_size
    
    # Axis labels and formatting
    ax.set_ylabel("Pearson's r", fontsize=fontsize+2, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(roi_labels, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(roi_labels) - 0.5)
    ax.set_title(f'{subject} - Encoding Accuracy by ROI',
                 fontsize=fontsize+4, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=fontsize)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)


# Plot average across subjects
ax = axes[-1]

# Plot bars with error bars for model correlations
bars = ax.bar(x_positions, mean_corrs, yerr=sem_corrs, color='#3498db', alpha=0.8,
              label='Model Correlation (mean ± SEM)', width=0.7, capsize=3, 
              error_kw={'linewidth': 1.5, 'alpha': 0.7})

# Plot noise ceiling with error bars as error regions
for i, (x, nc, sem) in enumerate(zip(x_positions, mean_ncs, sem_ncs)):
    # Central line
    ax.plot([x-0.35, x+0.35], [nc, nc], color='#e74c3c', linewidth=3, alpha=0.8)
    # Error region
    ax.fill_between([x-0.35, x+0.35], nc-sem, nc+sem, color='#e74c3c', alpha=0.2)

# Add legend entry for noise ceiling
ax.plot([], [], color='#e74c3c', linewidth=3, alpha=0.8, label='Noise Ceiling (mean ± SEM)')

# Add vertical lines to separate groups
current_pos = 0
for i, group_name in enumerate(roi_groups.keys()):
    group_size = len(roi_groups[group_name])
    
    # Add group label
    group_center = current_pos + group_size / 2 - 0.5
    ax.text(group_center, -0.15, group_name, ha='center', va='top',
            fontsize=fontsize, fontweight='bold', transform=ax.get_xaxis_transform())
    
    # Add separator line (except after last group)
    if i < len(roi_groups) - 1:
        separator_pos = current_pos + group_size - 0.5
        ax.axvline(separator_pos, color='black', linewidth=1.5, alpha=0.3, linestyle='--')
    
    current_pos += group_size

# Axis labels and formatting
ax.set_xlabel('ROI', fontsize=fontsize+2, fontweight='bold', labelpad=40)
ax.set_ylabel("Pearson's r", fontsize=fontsize+2, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(roi_labels, rotation=45, ha='right')
ax.set_xlim(-0.5, len(roi_labels) - 0.5)
ax.set_title(f'Average Across Subjects (N={n_subjects}) - Encoding Accuracy by ROI',
             fontsize=fontsize+4, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=fontsize)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

plt.tight_layout()


# =============================================================================
# Save the figure
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-things_fmri_1', f'model-{args.model}', 'encoding_models_accuracy')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

subject_str = '_'.join(args.subjects)
save_name = f'encoding_accuracy_roi_multisubject_{args.regression}_{cls_suffix}_{args.model}_{subject_str}'
fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight', format='png')
fig.savefig(os.path.join(save_dir, f'{save_name}.jpg'), dpi=300, bbox_inches='tight', format='jpeg')

print(f"\nPlot saved to: {save_dir}/{save_name}.png and {save_dir}/{save_name}.jpg")


# =============================================================================
# Print summary statistics
# =============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS - AVERAGE ACROSS SUBJECTS")
print("="*80)

for i, group_name in enumerate(roi_groups.keys()):
    group_start = group_boundaries[i]
    group_end = group_boundaries[i+1]
    
    group_corrs = mean_corrs[group_start:group_end]
    group_ncs = mean_ncs[group_start:group_end]
    
    print(f"\n{group_name}:")
    print(f"  Mean correlation: {group_corrs.mean():.4f} ± {group_corrs.std():.4f}")
    print(f"  Mean noise ceiling: {group_ncs.mean():.4f} ± {group_ncs.std():.4f}")
    print(f"  Gap to ceiling: {(group_ncs.mean() - group_corrs.mean()):.4f}")

print(f"\n{'='*80}")
print(f"Overall:")
print(f"  Mean correlation: {mean_corrs.mean():.4f} ± {mean_corrs.std():.4f}")
print(f"  Mean noise ceiling: {mean_ncs.mean():.4f} ± {mean_ncs.std():.4f}")
print(f"  Gap to ceiling: {(mean_ncs.mean() - mean_corrs.mean()):.4f}")
print("="*80)