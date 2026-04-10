"""Plot the encoding models' prediction accuracy for the Podcast ECoG dataset.

Two plots per subject:
1. Temporal profile: mean correlation across electrodes vs lag time
2. Spatial map: max-over-lags correlation per electrode on brain

And one summary plot averaging across subjects.

Parameters
----------
subjects : list
    List of subject identifiers.
berg_dir : str
    Directory of the BERG framework.
"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

try:
    from nilearn.plotting import plot_markers
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    print("Warning: nilearn not installed, skipping brain plots.")


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', nargs='+',
                    default=[f'sub-{i:02d}' for i in range(1, 10)],
                    help='List of subjects (default: sub-01 to sub-09)')
parser.add_argument('--berg_dir', required=True, type=str)
parser.add_argument('--top_percent', type=float, default=10.0,
                    help='Percentage of top electrodes to use for selected-electrodes plot (default: 10)')
args = parser.parse_args()

print('>>> Plot Podcast ECoG Encoding Accuracy <<<')


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 12
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = False
plt.rcParams["text.usetex"] = False

# Subject colors — a perceptually distinct palette for up to 9 subjects
subject_colors = [
    '#e74c3c',  # red
    '#3498db',  # blue
    '#2ecc71',  # green
    '#9b59b6',  # purple
    '#e67e22',  # orange
    '#1abc9c',  # teal
    '#e84393',  # pink
    '#f39c12',  # amber
    '#6c5ce7',  # indigo
]

LINE_COLOR = '#2c3e50'      # dark slate for individual temporal profiles
FILL_COLOR = '#3498db'      # blue for SEM shading
AVG_LINE_COLOR = '#e74c3c'  # red for average across subjects
AVG_FILL_COLOR = '#e74c3c'


# =============================================================================
# Load metadata
# =============================================================================
metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-ecog',
                            'train_dataset-zada2025', 'model-gpt2_xl',
                            'metadata')

all_correlations = []
all_times = []
all_coords = []
all_ch_names = []
loaded_subjects = []

for subject in args.subjects:
    file_name = f'metadata_{subject}.npy'
    file_path = os.path.join(metadata_dir, file_name)

    if not os.path.exists(file_path):
        print(f"  Warning: {file_path} not found, skipping.")
        continue

    metadata = np.load(file_path, allow_pickle=True).item()

    corr = metadata['encoding_model']['correlation_results']
    times = metadata['ecog']['times']
    coords = metadata['ecog']['ch_coords']
    ch_names = metadata['ecog']['ch_names']

    all_correlations.append(corr)
    all_times.append(times)
    all_coords.append(coords)
    all_ch_names.append(ch_names)
    loaded_subjects.append(subject)

n_subjects = len(all_correlations)
if n_subjects == 0:
    raise RuntimeError("No metadata files found.")


# =============================================================================
# Print detailed statistics
# =============================================================================
print("\n" + "=" * 70)
print("ENCODING ACCURACY STATISTICS")
print("=" * 70)

for i, subject in enumerate(loaded_subjects):
    corr = all_correlations[i]
    times = all_times[i]
    n_el = corr.shape[0]

    # Mean correlation across all electrodes and lags
    mean_all = corr.mean()

    # Max-over-lags per electrode
    max_per_el = corr.max(axis=1)

    # Best lag per electrode
    best_lag_idx = corr.argmax(axis=1)
    best_lag_times = times[best_lag_idx]

    # Mean temporal profile (averaged across electrodes)
    mean_temporal = corr.mean(axis=0)
    peak_lag_idx = mean_temporal.argmax()
    peak_lag_time = times[peak_lag_idx]
    peak_corr = mean_temporal[peak_lag_idx]

    # Fraction of electrodes with meaningful encoding
    n_positive = (max_per_el > 0).sum()
    n_above_01 = (max_per_el > 0.1).sum()
    n_above_02 = (max_per_el > 0.2).sum()

    print(f"\n{subject} ({n_el} el):")
    print(f"  Mean r (all el x lags):        {mean_all:.4f}")
    print(f"  Max-over-lags per electrode:")
    print(f"    min={max_per_el.min():.4f}, "
          f"median={np.median(max_per_el):.4f}, "
          f"mean={max_per_el.mean():.4f}, "
          f"max={max_per_el.max():.4f}")
    print(f"  Mean temporal profile peak:     r={peak_corr:.4f} at lag={peak_lag_time:.3f}s")
    print(f"  Electrodes with max r > 0:      {n_positive}/{n_el} ({100*n_positive/n_el:.1f}%)")
    print(f"  Electrodes with max r > 0.1:    {n_above_01}/{n_el} ({100*n_above_01/n_el:.1f}%)")
    print(f"  Electrodes with max r > 0.2:    {n_above_02}/{n_el} ({100*n_above_02/n_el:.1f}%)")

# Grand average
print(f"\n{'─' * 70}")
print(f"Grand average across {n_subjects} subjects:")
grand_mean_temporal = np.mean([c.mean(axis=0) for c in all_correlations], axis=0)
grand_peak_idx = grand_mean_temporal.argmax()
grand_peak_time = all_times[0][grand_peak_idx]
grand_peak_corr = grand_mean_temporal[grand_peak_idx]
grand_mean_all = np.mean([c.mean() for c in all_correlations])
print(f"  Mean r:                         {grand_mean_all:.4f}")
print(f"  Temporal profile peak:          r={grand_peak_corr:.4f} at lag={grand_peak_time:.3f}s")
print("=" * 70)


# =============================================================================
# Determine shared y-axis range
# =============================================================================
# Compute the range from mean temporal profiles (what's actually plotted)
all_mean_profiles = [c.mean(axis=0) for c in all_correlations]
all_sem_profiles = [c.std(axis=0) / np.sqrt(c.shape[0]) for c in all_correlations]

y_max_data = max((m + s).max() for m, s in zip(all_mean_profiles, all_sem_profiles))
y_min_data = min((m - s).min() for m, s in zip(all_mean_profiles, all_sem_profiles))

# Add 10% padding
y_range = y_max_data - y_min_data
y_lim = (y_min_data - 0.1 * y_range, y_max_data + 0.1 * y_range)


# =============================================================================
# Plot 1: Temporal profiles per subject + average
# =============================================================================
n_cols = min(3, n_subjects)
n_rows = (n_subjects + n_cols - 1) // n_cols + 1  # +1 for average row

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                         squeeze=False)

for i, subject in enumerate(loaded_subjects):
    ax = axes[i // n_cols, i % n_cols]
    corr = all_correlations[i]
    times = all_times[i]

    mean_corr = corr.mean(axis=0)
    sem_corr = corr.std(axis=0) / np.sqrt(corr.shape[0])

    ax.plot(times, mean_corr, color=LINE_COLOR, linewidth=2)
    ax.fill_between(times, mean_corr - sem_corr, mean_corr + sem_corr,
                    alpha=0.2, color=FILL_COLOR)
    ax.axhline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Encoding r (± SEM)')
    ax.set_title(f'{subject} ({corr.shape[0]} elec)', fontweight='bold')
    ax.set_ylim(y_lim)

# Hide unused subplots in subject rows
for i in range(n_subjects, (n_rows - 1) * n_cols):
    axes[i // n_cols, i % n_cols].set_visible(False)

# Average across subjects in last row
ax_avg = axes[-1, 0]
avg_corr = np.mean(all_mean_profiles, axis=0)
avg_sem = np.std(all_mean_profiles, axis=0) / np.sqrt(n_subjects)
times = all_times[0]

ax_avg.plot(times, avg_corr, color=AVG_LINE_COLOR, linewidth=2.5)
ax_avg.fill_between(times, avg_corr - avg_sem, avg_corr + avg_sem,
                    alpha=0.2, color=AVG_FILL_COLOR)
ax_avg.axhline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
ax_avg.axvline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
ax_avg.set_xlabel('Lag (s)')
ax_avg.set_ylabel('Encoding r (± SEM)')
ax_avg.set_title(f'Average across {n_subjects} subjects', fontweight='bold')
ax_avg.set_ylim(y_lim)

for j in range(1, n_cols):
    axes[-1, j].set_visible(False)

plt.tight_layout()

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-ecog',
                        'train_dataset-zada2025', 'model-gpt2_xl',
                        'encoding_models_accuracy')
os.makedirs(save_dir, exist_ok=True)

fig.savefig(os.path.join(save_dir, 'encoding_accuracy_temporal.jpg'),
            dpi=300, bbox_inches='tight')
print(f"\nTemporal plot saved.")


# =============================================================================
# Plot 2: Temporal profiles using top-k% electrodes per subject
# =============================================================================
# For each subject, select the top top_percent% of electrodes ranked by their
# max-over-lags correlation. This approximates the paper's approach of showing
# only electrodes with significant encoding performance.
top_pct = args.top_percent
print(f"\nComputing temporal profiles for top {top_pct:.0f}% electrodes...")

n_cols_top = min(3, n_subjects)
n_rows_top = (n_subjects + n_cols_top - 1) // n_cols_top + 1

fig_top, axes_top = plt.subplots(n_rows_top, n_cols_top,
                                 figsize=(5 * n_cols_top, 4 * n_rows_top),
                                 squeeze=False)

all_topk_profiles = []

for i, subject in enumerate(loaded_subjects):
    ax = axes_top[i // n_cols_top, i % n_cols_top]
    corr = all_correlations[i]
    times = all_times[i]
    n_el = corr.shape[0]

    # Select top-k% electrodes by max-over-lags correlation
    max_per_el = corr.max(axis=1)
    k = max(1, int(np.ceil(n_el * top_pct / 100)))
    top_indices = np.argsort(max_per_el)[-k:]

    mean_corr = corr[top_indices].mean(axis=0)
    sem_corr = corr[top_indices].std(axis=0) / np.sqrt(k)

    all_topk_profiles.append(mean_corr)

    ax.plot(times, mean_corr, color=LINE_COLOR, linewidth=2)
    ax.fill_between(times, mean_corr - sem_corr, mean_corr + sem_corr,
                    alpha=0.2, color=FILL_COLOR)
    ax.axhline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Encoding r (± SEM)')
    ax.set_title(f'{subject} (top {k}/{n_el} elec)', fontweight='bold')

    print(f"  {subject}: top {k}/{n_el} elec, "
          f"peak r={mean_corr.max():.4f} at lag={times[mean_corr.argmax()]:.3f}s")

# Shared y-axis for top-k plots
all_topk_sem = [all_correlations[i][
    np.argsort(all_correlations[i].max(axis=1))[
        -max(1, int(np.ceil(all_correlations[i].shape[0] * top_pct / 100))):
    ]
].std(axis=0) / np.sqrt(
    max(1, int(np.ceil(all_correlations[i].shape[0] * top_pct / 100)))
) for i in range(n_subjects)]

y_max_top = max((m + s).max() for m, s in zip(all_topk_profiles, all_topk_sem))
y_min_top = min((m - s).min() for m, s in zip(all_topk_profiles, all_topk_sem))
y_range_top = y_max_top - y_min_top
y_lim_top = (y_min_top - 0.1 * y_range_top, y_max_top + 0.1 * y_range_top)

for i in range(n_subjects):
    axes_top[i // n_cols_top, i % n_cols_top].set_ylim(y_lim_top)

# Hide unused subplots
for i in range(n_subjects, (n_rows_top - 1) * n_cols_top):
    axes_top[i // n_cols_top, i % n_cols_top].set_visible(False)

# Average across subjects in last row
ax_avg_top = axes_top[-1, 0]
avg_topk = np.mean(all_topk_profiles, axis=0)
avg_topk_sem = np.std(all_topk_profiles, axis=0) / np.sqrt(n_subjects)

ax_avg_top.plot(all_times[0], avg_topk, color=AVG_LINE_COLOR, linewidth=2.5)
ax_avg_top.fill_between(all_times[0], avg_topk - avg_topk_sem,
                        avg_topk + avg_topk_sem,
                        alpha=0.2, color=AVG_FILL_COLOR)
ax_avg_top.axhline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
ax_avg_top.axvline(0, color='#bdc3c7', linestyle='--', linewidth=0.8)
ax_avg_top.set_xlabel('Lag (s)')
ax_avg_top.set_ylabel('Encoding r (± SEM)')
ax_avg_top.set_title(f'Average across {n_subjects} subjects '
                     f'(top {top_pct:.0f}% elec)', fontweight='bold')
ax_avg_top.set_ylim(y_lim_top)

for j in range(1, n_cols_top):
    axes_top[-1, j].set_visible(False)

plt.tight_layout()
fig_top.savefig(os.path.join(save_dir,
                f'encoding_accuracy_temporal_top{int(top_pct)}pct.jpg'),
                dpi=300, bbox_inches='tight')
print(f"Top {top_pct:.0f}% temporal plot saved.")




# =============================================================================
# Plot 3: Spatial maps (max-over-lags correlation on brain)
# =============================================================================
if HAS_NILEARN:
    for i, subject in enumerate(loaded_subjects):
        corr = all_correlations[i]
        coords = all_coords[i]

        values = corr.max(axis=1)

        # nilearn expects coordinates in mm
        coords_mm = coords.copy()
        if np.abs(coords_mm).max() < 1:
            coords_mm *= 1000

        order = values.argsort()

        fig_brain, ax_brain = plt.subplots(1, 1, figsize=(10, 4))
        plot_markers(
            values[order], coords_mm[order],
            node_size=30, display_mode='lzr',
            node_vmin=0, node_cmap='inferno_r',
            colorbar=True, axes=ax_brain,
        )
        fig_brain.suptitle(f'{subject} - Max encoding r per electrode',
                           fontsize=fontsize + 2)

        fig_brain.savefig(
            os.path.join(save_dir, f'encoding_accuracy_spatial_{subject}.jpg'),
            dpi=300, bbox_inches='tight')
        plt.close(fig_brain)

    print("Spatial plots saved.")

plt.show()