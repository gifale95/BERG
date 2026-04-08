"""Plot the OPT-1.3B encoding models' prediction accuracy by brain region,
with noise ceiling reference.

Generates two figures:
1. Histogram of voxelwise correlations (CCabs) with noise ceiling (CCmax)
   distribution overlaid — one panel per subject.
2. Per-ROI bar chart showing mean encoding accuracy with noise ceiling
   reference bars — averaged across subjects.

Parameters
----------
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
subjects : list of str
    Subject identifiers. Default: UTS01, UTS02, UTS03.
"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(
    description='Plot OPT-1.3B fMRI encoding model results.')

parser.add_argument('--berg_dir', required=True, type=str,
    help='Path to the BERG data directory.')
parser.add_argument('--subjects', nargs='+', type=str,
    default=['UTS01', 'UTS02', 'UTS03'],
    help='Subject identifiers.  Default: UTS01 UTS02 UTS03.')

args = parser.parse_args()

print('>>> Plot OPT-1.3B fMRI encoding model results <<<')


# ============================================================================
# Plot parameters
# ============================================================================
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
matplotlib.rcParams['axes.grid'] = False
plt.rcParams['text.usetex'] = False


# ============================================================================
# Load metadata for all subjects
# ============================================================================
metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-lebel2023', 'model-opt_1_3b_ridge', 'metadata')

all_corrs = []
all_cc_norm = []
all_noise_ceiling = []
all_roi_dicts = []

for subject in args.subjects:
    meta_path = os.path.join(metadata_dir, f'metadata_{subject}.npy')
    metadata = np.load(meta_path, allow_pickle=True).item()

    all_corrs.append(metadata['encoding_model']['correlation'])
    all_cc_norm.append(metadata['encoding_model']['cc_norm'])
    all_noise_ceiling.append(metadata['encoding_model']['noise_ceiling'])
    all_roi_dicts.append(metadata['roi'])

print(f'Loaded metadata for {len(args.subjects)} subjects.')


# ============================================================================
# Figure 1: Histogram of voxelwise correlations per subject
# ============================================================================
n_subjects = len(args.subjects)
fig1, axes = plt.subplots(1, n_subjects, figsize=(5 * n_subjects, 4),
                           squeeze=False)

for s, subject in enumerate(args.subjects):
    ax = axes[0, s]
    corrs = all_corrs[s]
    nc = all_noise_ceiling[s]

    # Only include voxels with CCmax > 0.35 (paper convention)
    good = nc > 0.35
    corrs_good = corrs[good]
    nc_good = nc[good]

    bins = np.linspace(-0.4, 1.0, 60)

    ax.hist(corrs_good, bins=bins, color='#3498db', alpha=0.7,
            edgecolor='white', linewidth=0.3, label='$CC_{abs}$', density=True)
    ax.hist(nc_good, bins=bins, color='#e74c3c', alpha=0.4,
            edgecolor='white', linewidth=0.3, label='$CC_{max}$', density=True)

    ax.axvline(np.mean(corrs_good), color='#2c3e50', linestyle='--',
               linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Correlation (r)', fontsize=fontsize)
    if s == 0:
        ax.set_ylabel('Density', fontsize=fontsize)
    ax.set_xlim(-0.4, 1.0)
    ax.set_title(f'{subject}\n'
                 f'mean r = {np.mean(corrs_good):.3f}, '
                 f'n = {np.sum(good)} voxels',
                 fontsize=fontsize)
    ax.legend(fontsize=fontsize - 2, loc='upper left')

fig1.suptitle('Voxelwise Encoding Performance ($CC_{max}$ > 0.35)',
              fontsize=fontsize + 2, fontweight='bold', y=1.02)
fig1.tight_layout()


# ============================================================================
# Figure 2: Per-ROI encoding accuracy (averaged across subjects)
# ============================================================================
# Collect all ROI names present in any subject
all_roi_names = set()
for rd in all_roi_dicts:
    all_roi_names.update(rd.keys())

# For each ROI, compute mean correlation and noise ceiling across subjects
# (only considering voxels with CCmax > 0.35)
roi_mean_r = {}
roi_mean_nc = {}
roi_n_voxels = {}

for roi_name in sorted(all_roi_names):
    rs = []
    ncs = []
    ns = []
    for s in range(n_subjects):
        if roi_name not in all_roi_dicts[s]:
            continue
        roi_mask = all_roi_dicts[s][roi_name]
        good_mask = all_noise_ceiling[s] > 0.35
        mask = roi_mask & good_mask
        n = np.sum(mask)
        if n > 0:
            rs.append(np.mean(all_corrs[s][mask]))
            ncs.append(np.mean(all_noise_ceiling[s][mask]))
            ns.append(n)
    if rs:
        roi_mean_r[roi_name] = np.mean(rs)
        roi_mean_nc[roi_name] = np.mean(ncs)
        roi_n_voxels[roi_name] = int(np.mean(ns))

# Sort ROIs by mean correlation (descending)
sorted_rois = sorted(roi_mean_r.keys(), key=lambda x: roi_mean_r[x],
                     reverse=True)

# Select top ROIs (skip ROIs with very few voxels)
min_voxels = 20
sorted_rois = [r for r in sorted_rois if roi_n_voxels[r] >= min_voxels]

# Limit to top 20 for readability
if len(sorted_rois) > 20:
    sorted_rois = sorted_rois[:20]

# Assign colours: highlight key language regions
highlight_rois = {'AC', 'Broca', 'Brocas', 'AG', 'PFC', 'PrCu', 'sPMv',
                  'FFA', 'EBA', 'PPA', 'RSC', 'OPA', 'OFA'}

fig2, ax = plt.subplots(figsize=(max(8, len(sorted_rois) * 0.6), 5))

x = np.arange(len(sorted_rois))
bar_width = 0.35

# Noise ceiling bars (behind)
nc_vals = [roi_mean_nc[r] for r in sorted_rois]
ax.bar(x, nc_vals, width=bar_width * 2, color='#ecf0f1', edgecolor='#bdc3c7',
       linewidth=1, label='Noise ceiling ($CC_{max}$)', zorder=1)

# Encoding accuracy bars (in front)
r_vals = [roi_mean_r[r] for r in sorted_rois]
colors = ['#2980b9' if r in highlight_rois else '#7f8c8d' for r in sorted_rois]
ax.bar(x, r_vals, width=bar_width, color=colors, edgecolor='white',
       linewidth=0.5, label='Encoding model ($CC_{abs}$)', zorder=2)

# Labels
ax.set_xticks(x)
labels = [f'{r}\n({roi_n_voxels[r]})' for r in sorted_rois]
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize - 2)
ax.set_ylabel('Correlation (r)', fontsize=fontsize)
ax.set_ylim(0, max(nc_vals) * 1.15)
ax.legend(fontsize=fontsize - 1, loc='upper right')

title_parts = ', '.join(args.subjects)
ax.set_title(f'Per-ROI Encoding Accuracy — OPT-1.3B Layer 18\n'
             f'Average across {n_subjects} subjects ({title_parts})',
             fontsize=fontsize + 1, fontweight='bold')

fig2.tight_layout()


# ============================================================================
# Save figures
# ============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-lebel2023', 'model-opt_1_3b_ridge', 'encoding_models_accuracy')
os.makedirs(save_dir, exist_ok=True)

fig1_path = os.path.join(save_dir, 'encoding_accuracy_histogram.jpg')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight', format='jpeg')
print(f'Saved histogram to: {fig1_path}')

fig2_path = os.path.join(save_dir, 'encoding_accuracy_per_roi.jpg')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight', format='jpeg')
print(f'Saved ROI plot to:  {fig2_path}')

plt.show()


"""
Example usage
=============

python 02_plot.py \\
    --berg_dir /path/to/brain-encoding-response-generator
"""
