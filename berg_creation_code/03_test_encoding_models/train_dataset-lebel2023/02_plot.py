"""Plot the OPT-1.3B encoding models' prediction accuracy:
  1. Per-subject cortical flatmap of noise-ceiling-normalised encoding
     accuracy (CCnorm = CCabs / CCmax), following Antonello et al. (2023)
     Figure 3c.  Only voxels with CCmax > threshold are shown.
  2. Per-subject bar chart of mean encoding accuracy per ROI, with noise
     ceiling reference markers.

Parameters
----------
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
subjects : list of str
    Subject identifiers. Default: UTS01, UTS02, UTS03.
ccmax_threshold : float
    Only include voxels with CCmax above this value. Default: 0.35
    (matching Antonello et al. Section 2.5).
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
    default=['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05',
             'UTS06', 'UTS07', 'UTS08'],
    help='Subject identifiers.  Default: all 8 LeBel et al. subjects.')
parser.add_argument('--ccmax_threshold', type=float, default=0.35,
    help='Only show voxels / ROIs with CCmax above this.  Default: 0.35.')

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

all_metadata = {}
for subject in args.subjects:
    meta_path = os.path.join(metadata_dir, f'metadata_{subject}.npy')
    all_metadata[subject] = np.load(meta_path, allow_pickle=True).item()

print(f'Loaded metadata for {len(args.subjects)} subjects.')

# Save directory
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-lebel2023', 'model-opt_1_3b_ridge',
    'encoding_models_accuracy')
os.makedirs(save_dir, exist_ok=True)


# ############################################################################
#  FIGURE 1 (per subject): Cortical flatmap of CCnorm
# ############################################################################
# CCnorm = CCabs / CCmax, only for voxels with CCmax > threshold.
# This is the standard noise-ceiling-normalised view used in
# Antonello et al. (2023), Figure 3c.

try:
    import cortex
    has_pycortex = True
except ImportError:
    has_pycortex = False
    print('\n  WARNING: pycortex not installed. Skipping cortical flatmaps.')

if has_pycortex:
    print('\nGenerating cortical flatmaps (CCnorm) ...')

    for subject in args.subjects:
        meta = all_metadata[subject]
        corrs = meta['encoding_model']['correlation']
        nc = meta['encoding_model']['noise_ceiling']

        # CCnorm = CCabs / CCmax (only for suprathreshold voxels)
        cc_norm = corrs / nc
        cc_norm[nc <= args.ccmax_threshold] = np.nan

        # Find pycortex transform name
        db_path = cortex.db.filestore
        xfm_dir = os.path.join(db_path, subject, 'transforms')
        xfm_names = [x for x in os.listdir(xfm_dir)
                     if not x.startswith('.')]
        xfmname = xfm_names[0]

        # Plot
        fig = plt.figure(figsize=(16, 10))
        vol = cortex.Volume(cc_norm, subject, xfmname,
                            vmin=0, vmax=1.0, cmap='inferno')
        cortex.quickshow(vol, fig=fig, with_colorbar=True,
                         with_curvature=True, with_rois=True,
                         linewidth=2)

        # Reposition axes after quickshow to avoid overlap:
        # Nudge brain down slightly, center the colorbar — but preserve
        # pycortex's per-subject sizing so no flatmap gets squished.
        all_axes = fig.get_axes()
        brain_ax = all_axes[0]
        bp = brain_ax.get_position()
        # Shift brain down a bit to make room for title, keep size
        brain_ax.set_position([bp.x0, bp.y0 - 0.03, bp.width, bp.height])

        # Find and center the colorbar axis
        for ax in all_axes[1:]:
            pos = ax.get_position()
            if pos.width > pos.height:
                cb_width = 0.4
                cb_x = 0.5 - cb_width / 2
                ax.set_position([cb_x, 0.03, cb_width, pos.height])
                break

        fig.suptitle(
            f'{subject} — Normalised Encoding Performance '
            f'($CC_{{norm}}$, $CC_{{max}}$ > {args.ccmax_threshold})',
            fontsize=fontsize + 1, fontweight='bold', y=0.96)

        fig_path = os.path.join(save_dir, f'flatmap_ccnorm_{subject}.png')
        fig.savefig(fig_path, dpi=200, facecolor='white')
        print(f'  Saved: {fig_path}')
        plt.close(fig)


# ############################################################################
#  FIGURE 2 (per subject): Per-ROI encoding accuracy bar chart
# ############################################################################

print('\nGenerating per-ROI bar charts ...')

min_voxels = 20

for subject in args.subjects:
    meta = all_metadata[subject]
    corrs = meta['encoding_model']['correlation']
    nc = meta['encoding_model']['noise_ceiling']
    roi_dict = meta['roi']

    # Compute per-ROI stats
    roi_stats = {}
    for roi_name in sorted(roi_dict.keys()):
        roi_mask = roi_dict[roi_name]
        good = nc > args.ccmax_threshold
        mask = roi_mask & good
        n = np.sum(mask)
        if n >= min_voxels:
            roi_stats[roi_name] = {
                'r_mean': np.mean(corrs[mask]),
                'nc_mean': np.mean(nc[mask]),
                'n_voxels': n,
            }

    if not roi_stats:
        print(f'  {subject}: no ROIs with enough voxels, skipping.')
        continue

    # Sort by encoding accuracy (descending)
    sorted_rois = sorted(roi_stats.keys(),
                         key=lambda x: roi_stats[x]['r_mean'], reverse=True)

    # Plot
    n_rois = len(sorted_rois)
    fig, ax = plt.subplots(figsize=(max(10, n_rois * 0.55), 5))

    x = np.arange(n_rois)
    bar_width = 0.6

    # Bars: encoding accuracy (CCabs)
    r_vals = [roi_stats[r]['r_mean'] for r in sorted_rois]
    ax.bar(x, r_vals, width=bar_width, color='#2980b9', edgecolor='white',
           linewidth=0.5, zorder=2)

    # Noise ceiling markers (CCmax)
    nc_vals = [roi_stats[r]['nc_mean'] for r in sorted_rois]
    ax.plot(x, nc_vals, color='#e74c3c', marker='_', markersize=14,
            markeredgewidth=2.5, linestyle='none', zorder=4)

    # Labels
    labels = [f'{r}\n({roi_stats[r]["n_voxels"]})' for r in sorted_rois]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor',
                       fontsize=fontsize - 2)
    ax.set_ylabel('Correlation (r)', fontsize=fontsize)
    ax.set_ylim(bottom=0, top=max(nc_vals) * 1.15)
    ax.axhline(0, color='black', linewidth=0.5)

    # Legend
    handles = [
        matplotlib.patches.Patch(facecolor='#2980b9',
            label='Encoding model ($CC_{abs}$)'),
        matplotlib.lines.Line2D([], [], color='#e74c3c', marker='_',
            markersize=12, markeredgewidth=2.5, linestyle='none',
            label='Noise ceiling ($CC_{max}$)'),
    ]
    ax.legend(handles=handles, fontsize=fontsize - 1, loc='upper right')

    n_voxels = meta['fmri']['n_voxels']
    n_good = np.sum(nc > args.ccmax_threshold)
    ax.set_title(
        f'{subject} — Per-ROI Encoding Accuracy (OPT-1.3B Layer 18)\n'
        f'{n_good:,} / {n_voxels:,} voxels with '
        f'$CC_{{max}}$ > {args.ccmax_threshold}',
        fontsize=fontsize + 1, fontweight='bold')

    fig.tight_layout()

    fig_path = os.path.join(save_dir, f'encoding_accuracy_per_roi_{subject}.jpg')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', format='jpeg')
    print(f'  Saved: {fig_path}')
    plt.close(fig)


# ============================================================================
# Summary
# ============================================================================
print(f'\n{"="*60}')
print(f'All plots saved to: {save_dir}')
print(f'{"="*60}')