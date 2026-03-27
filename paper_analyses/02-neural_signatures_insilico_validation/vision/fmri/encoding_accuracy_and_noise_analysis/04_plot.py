"""Plot the encoding accuracy and noise analysis results fro BERG's fMRI
encoding models trained on NSD.

Parameters
----------
threshold : int
    If 1, only plot encoding accuracies for significant vertices on brain
    surfaces. If 0, plot encoding accuracies for all vertices.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import cortex
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the encoding accuracy results, and create the plot saving directory
# =============================================================================
# Load the results
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation',
'vision', 'fmri', 'encoding_accuracy', 'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
correlation_nsdcore = results['correlation_nsdcore']
correlation_nsdsynthetic = results['correlation_nsdsynthetic']
sig_correlation_nsdcore = results['sig_correlation_nsdcore']
sig_correlation_nsdsynthetic = results['sig_correlation_nsdsynthetic']
diff_correlation_nsdcore = results['diff_correlation_nsdcore']
diff_correlation_nsdsynthetic = results['diff_correlation_nsdsynthetic']
sig_diff_correlation_nsdcore = results['sig_diff_correlation_nsdcore']
sig_diff_correlation_nsdsynthetic = results['sig_diff_correlation_nsdsynthetic']                                                         
corr_iv1tr_is_avg = results['corr_iv1tr_is_avg']
corr_iv1tr_iv2tr_avg = results['corr_iv1tr_iv2tr_avg']
corr_iv1tr_iv1tr_avg = results['corr_iv1tr_iv1tr_avg']
ci_corr_iv1tr_is = results['ci_corr_iv1tr_is']
ci_corr_iv1tr_iv2tr = results['ci_corr_iv1tr_iv2tr']
ci_corr_iv1tr_iv1tr = results['ci_corr_iv1tr_iv1tr']
p_val_1 = results['p_val_1']
p_val_2 = results['p_val_2']

# Plot save directory
save_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'encoding_accuracy', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the encoding accuracy results on brain surfaces
# =============================================================================
# Plot parameters
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'

# Loop across models
for model in correlation_nsdcore.keys():

    # NSD-core
    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(correlation_nsdcore[model]['lh'], 0),
        np.nanmean(correlation_nsdcore[model]['rh'], 0))
    # Only retain significant vertices
    if args.threshold == 1:
        sig = np.append(sig_correlation_nsdcore[model]['lh'],
            sig_correlation_nsdcore[model]['rh'])
        data[~sig] = np.nan
    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='afmhot',
        vmin=0,
        vmax=0.8,
        with_colorbar=True
        )
    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=False,
        labelsize=35,
        curvature_brightness=0.4,
        with_colorbar=True
        )
    # Save the figure
    if args.threshold == 0:
        file_name = os.path.join(save_dir,
            f'encoding_accuracy_nsdcore_model-{model}.svg')
    if args.threshold == 1:
        file_name = os.path.join(save_dir,
            f'encoding_accuracy_thresholded_nsdcore_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()

    # NSD-synthetic
    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(
        np.nanmean(correlation_nsdsynthetic[model]['lh'], 0),
        np.nanmean(correlation_nsdsynthetic[model]['rh'], 0))
    # Only retain significant vertices
    if args.threshold == 1:
        sig = np.append(sig_correlation_nsdsynthetic[model]['lh'],
            sig_correlation_nsdsynthetic[model]['rh'])
        data[~sig] = np.nan
    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='afmhot',
        vmin=0,
        vmax=0.8,
        with_colorbar=True
        )
    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=False,
        labelsize=35,
        curvature_brightness=0.4,
        with_colorbar=True
        )
    # Save the figure
    if args.threshold == 0:
        file_name = os.path.join(save_dir,
            f'encoding_accuracy_nsdsynthetic_model-{model}.svg')
    elif args.threshold == 1:
        file_name = os.path.join(save_dir,
            f'encoding_accuracy_thresholded_nsdsynthetic_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the encoding accuracy differences on brain surfaces
# =============================================================================
# Plot parameters
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'

# Loop across models
for model in diff_correlation_nsdcore.keys():

    # NSD-core
    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(diff_correlation_nsdcore[model]['lh'], 0),
        np.nanmean(diff_correlation_nsdcore[model]['rh'], 0))
    # Only retain significant vertices
    if args.threshold == 1:
        sig = np.append(sig_diff_correlation_nsdcore[model]['lh'],
            sig_diff_correlation_nsdcore[model]['rh'])
        data[~sig] = np.nan
    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='RdGy_r',
        vmin=-0.4,
        vmax=0.4,
        with_colorbar=True
        )
    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=False,
        labelsize=35,
        curvature_brightness=0.4,
        with_colorbar=True
        )
    # Save the figure
    if args.threshold == 0:
        file_name = os.path.join(save_dir,
            f'diff_encoding_accuracy_nsdcore_model-{model}.svg')
    if args.threshold == 1:
        file_name = os.path.join(save_dir,
            f'diff_encoding_accuracy_thresholded_nsdcore_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()

    # NSD-synthetic
    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(
        np.nanmean(diff_correlation_nsdsynthetic[model]['lh'], 0),
        np.nanmean(diff_correlation_nsdsynthetic[model]['rh'], 0))
    # Only retain significant vertices
    if args.threshold == 1:
        sig = np.append(sig_diff_correlation_nsdsynthetic[model]['lh'],
            sig_diff_correlation_nsdsynthetic[model]['rh'])
        data[~sig] = np.nan
    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='RdGy_r',
        vmin=-0.4,
        vmax=0.4,
        with_colorbar=True
        )
    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=False,
        labelsize=35,
        curvature_brightness=0.4,
        with_colorbar=True
        )
    # Save the figure
    if args.threshold == 0:
        file_name = os.path.join(save_dir,
            f'diff_encoding_accuracy_nsdsynthetic_model-{model}.svg')
    if args.threshold == 1:
        file_name = os.path.join(save_dir,
            f'diff_encoding_accuracy_thresholded_nsdsynthetic_model-{model}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close()


# =============================================================================
# Plot the noise analysis results
# =============================================================================
# Plot parameters
fontsize = 25
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 0
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 0
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

for model in corr_iv1tr_is_avg.keys():

    # Format the results for plotting
    corr_avg = []
    corr_avg.append(corr_iv1tr_iv1tr_avg[model])
    corr_avg.append(corr_iv1tr_iv2tr_avg[model])
    corr_avg.append(corr_iv1tr_is_avg[model])
    corr_ci = []
    corr_ci.append(ci_corr_iv1tr_iv1tr[model])
    corr_ci.append(ci_corr_iv1tr_iv2tr[model])
    corr_ci.append(ci_corr_iv1tr_is[model])
    labels = ['In vivo\n(single trials)', 'In vivo\n(trial average)',
        'In silico']

    # Plot parameters
    n_sub = len(args.subjects)
    x_coord = np.arange(len(corr_avg))
    alpha = 0.2
    marker = 'o'
    s_single = 500
    s_mean = 750
    color = 'k'
    dist = 0.15
    fontsize_sig = 20
    sig_offset = 0.02
    sig_bar_length = 0.005
    linewidth_sig_bar = 1
    sig_star_offset_top = 0.005

    # Create the figure
    fig = plt.figure(figsize=(10, 10))

    # Loop across comparisons
    for i in range(len(corr_avg)):

        # Plot lines connecting subjects from different comparisons
        for s in range(n_sub):
            if i > 0:
                plt.plot([x_coord[i-1], x_coord[i]], [corr_avg[i-1][s],
                    corr_avg[i][s]], color=color, alpha=alpha, linewidth=1)

        # Correlation scores scores
        x = np.repeat(i, n_sub)
        y = corr_avg[i]
        plt.scatter(x, y, s=s_single, color=color, alpha=alpha,
            edgecolors='none')
        plt.scatter(x[0], np.mean(y), s=s_mean, color=color, edgecolors='none')

        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - corr_ci[i][0]
        ci[1] = corr_ci[i][1] - np.mean(y)
        plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)), fmt="none",
            ecolor=color, elinewidth=5, capsize=0)

    # Significance 1
    if p_val_1[model] < 0.05:
        res = np.append(corr_iv1tr_iv1tr_avg[model],
            corr_iv1tr_iv2tr_avg[model])
        y_max = max(res) + sig_offset
        plt.plot([x_coord[0], x_coord[0]], [y_max, y_max+sig_bar_length],
            'k-', [x_coord[0], x_coord[1]],
            [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
            [x_coord[1], x_coord[1]], [y_max+sig_bar_length, y_max], 'k-',
            linewidth=linewidth_sig_bar)
        x_mean = np.mean(np.asarray((x_coord[0], x_coord[1])))
        y = y_max + sig_bar_length + sig_star_offset_top
        plt.text(x_mean, y, s='*', fontsize=fontsize_sig, color='k',
            fontweight='bold', ha='center', va='center')

    # Significance 2
    if p_val_2[model] < 0.05:
        res = np.append(corr_iv1tr_iv2tr_avg[model],
            corr_iv1tr_is_avg[model])
        y_max = max(res) + sig_offset
        plt.plot([x_coord[1], x_coord[1]], [y_max, y_max+sig_bar_length],
            'k-', [x_coord[1], x_coord[2]],
            [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
            [x_coord[2], x_coord[2]], [y_max+sig_bar_length, y_max], 'k-',
            linewidth=linewidth_sig_bar)
        x_mean = np.mean(np.asarray((x_coord[1], x_coord[2])))
        y = y_max + sig_bar_length + sig_star_offset_top
        plt.text(x_mean, y, s='*', fontsize=fontsize_sig, color='k',
            fontweight='bold', ha='center', va='center')

    # x-axis parameters
    xticks = x_coord
    plt.xticks(ticks=xticks, labels=labels, rotation=0, ha='center')
    plt.xlim(left=-0.5, right=2.5)

    # y-axis parameters
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.yticks(ticks=yticks, labels=ylabels)
    ylabel = "Pearson's $r$"
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.ylim(bottom=0, top=0.4)

    # Save the figure
    file_name = os.path.join(save_dir, 'noise_analysis.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
        format='svg')
    plt.close()