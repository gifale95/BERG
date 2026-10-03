"""Plot explanation accuracy as a function of prediction accuracy, for in
silico fMRI responses.

Parameters
----------
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Plot <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'relationship_prediction_explanation',
    'fmri', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'relationship_prediction_explanation', 'fmri', 'stats', 'stats.npy')

results = np.load(results_dir, allow_pickle=True).item()

encoding_models = np.array(results['encoding_models'])
fmri_subjects = np.array(results['fmri_subjects'])
prediction_accuracy_nsdcore = results['prediction_accuracy_nsdcore']
prediction_accuracy_nsdsynthetic = results['prediction_accuracy_nsdsynthetic']
insilico_validation_scores = results['insilico_validation_scores']
corr_nsdcore = results['corr_nsdcore']
corr_nsdsynthetic = results['corr_nsdsynthetic']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
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
colors = [
    (0/255, 0/255, 0/255),
    (150/255, 150/255, 150/255),
    (139/255, 0/255, 0/255)
    ]

titles = [
    'Polar angle',
    'Eccentricity',
    'Face selectivity',
    'Body selectivity',
    'Place selectivity',
    'AlexNet layerwise modeling',
    'LLM modeling',
    'Behavioral modeling'
]

y_labels = [
    "Pearson's $r$",
    "Pearson's $r$",
    "Pearson's $r$",
    "Pearson's $r$",
    "Pearson's $r$",
    "Spearman's $ρ$",
    "Δ Pearson's $r$",
    "Δ Pearson's $r$"
]

signatures = [
    'corr_polar_angle_silico_vivo',
    'corr_eccentricity_silico_vivo',
    'corr_tval_silico_vivo_faces',
    'corr_tval_silico_vivo_bodies',
    'corr_tval_silico_vivo_places',
    'corr_best_layer_hierarchy_score',
    'diff_llm_rsa_high_early',
    'diff_behavioral_rsa_high_early'
]


# =============================================================================
# Plot the results (NSD-core)
# =============================================================================
fig, axs = plt.subplots(2, 4, sharex=True, sharey=False, figsize=(40, 17.5))
axs = np.reshape(axs, -1)

fig.supylabel("Explanation accuracy", fontsize=fontsize, x=0.075)
fig.supxlabel("Prediction accuracy", fontsize=fontsize)

for i, key in enumerate(signatures):

    val = insilico_validation_scores[key]

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Scatter plot of the insilico prediction accuracy vs. explanation accuracy
    acc = np.empty(0)
    validation = np.empty(0)
    for m in range(len(encoding_models)):
        axs[i].scatter(prediction_accuracy_nsdcore[m], val[m], s=200,
            color=colors[m], label=f'{encoding_models[m][19:]}', alpha=0.75,
            zorder=2)
        acc = np.append(acc, prediction_accuracy_nsdcore[m])
        validation = np.append(validation, val[m])

    # Plot the subject connection lines
    acc_array = np.array(prediction_accuracy_nsdcore)
    val_array = np.array(val)
    for s in range(len(fmri_subjects)):
        axs[i].plot(acc_array[:,s], val_array[:,s], color='k', linewidth=1,
            alpha=.1, zorder=1)

    # Print the correlation score between prediction accuracy and explanation
    # accuracy
    x = 0.34
    y = min(validation) + (max(validation) - min(validation)) * 0.05
    if corr_nsdcore[key][1] < 0.0001:
        s = f'$r$={np.round(corr_nsdcore[key][0], 2):0.2f}***'
    elif corr_nsdcore[key][1] < 0.001:
        s = f'$r$={np.round(corr_nsdcore[key][0], 2):0.2f}**'
    elif corr_nsdcore[key][1] < 0.01:
        s = f'$r$={np.round(corr_nsdcore[key][0], 2):0.2f}*'
    else:
        s = f'$r$={np.round(corr_nsdcore[key][0], 2):0.2f}'
    axs[i].text(x, y, s, fontsize=fontsize)

    # Plot the correlation line
    m, b = np.polyfit(acc, validation, 1)
    x_line = np.linspace(acc.min(), acc.max(), 100)
    y_line = m * x_line + b
    axs[i].plot(x_line, y_line, color='k', linewidth=2, alpha=0.3, zorder=1)

    # Plot title
    axs[i].set_title(f'{titles[i]}', fontsize=fontsize)

    # x-axis parameters
    if i in [4, 5, 6, 7]:
        axs[i].set_xlabel("Pearson's $r$", fontsize=fontsize)
        xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        xlabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        axs[i].set_xticks(ticks=xticks, labels=xlabels)
        axs[i].set_xlim(left=0.13, right=.48)

    # y-axis parameters
    axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
    # yticks = [10, 15, 20, 25, 30]
    # ylabels = [10, 15, 20, 25, 30]
    # axs[i].set_yticks(ticks=yticks, labels=ylabels)
    # axs[i].set_ylim(bottom=8, top=29)

    # Legend
    if i == 0:
        axs[i].legend(ncol=3, fontsize=fontsize, loc=0, frameon=False,
            bbox_to_anchor=(3.9, 1.3), markerscale=2)

# Save the figure
file_name = os.path.join(save_dir, f'scatterplots_nsdcore.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)


# =============================================================================
# Plot the results (NSD-synthetic)
# =============================================================================
fig, axs = plt.subplots(2, 4, sharex=True, sharey=False, figsize=(40, 17.5))
axs = np.reshape(axs, -1)

fig.supylabel("Explanation accuracy", fontsize=fontsize, x=0.075)
fig.supxlabel("Prediction accuracy", fontsize=fontsize)

for i, key in enumerate(signatures):

    val = insilico_validation_scores[key]

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Scatter plot of the insilico prediction accuracy vs. explanation accuracy
    acc = np.empty(0)
    validation = np.empty(0)
    for m in range(len(encoding_models)):
        axs[i].scatter(prediction_accuracy_nsdsynthetic[m], val[m], s=200,
            color=colors[m], label=f'{encoding_models[m][19:]}', alpha=0.75,
            zorder=2)
        acc = np.append(acc, prediction_accuracy_nsdsynthetic[m])
        validation = np.append(validation, val[m])

    # Plot the subject connection lines
    acc_array = np.array(prediction_accuracy_nsdsynthetic)
    val_array = np.array(val)
    for s in range(len(fmri_subjects)):
        axs[i].plot(acc_array[:,s], val_array[:,s], color='k', linewidth=1,
            alpha=.1, zorder=1)

    # Print the correlation score between prediction accuracy and explanation
    # accuracy
    x = 0.19
    y = min(validation) + (max(validation) - min(validation)) * 0.05
    if corr_nsdsynthetic[key][1] < 0.0001:
        s = f'$r$={np.round(corr_nsdsynthetic[key][0], 2):0.2f}***'
    elif corr_nsdsynthetic[key][1] < 0.001:
        s = f'$r$={np.round(corr_nsdsynthetic[key][0], 2):0.2f}**'
    elif corr_nsdsynthetic[key][1] < 0.01:
        s = f'$r$={np.round(corr_nsdsynthetic[key][0], 2):0.2f}*'
    else:
        s = f'$r$={np.round(corr_nsdsynthetic[key][0], 2):0.2f}'
    axs[i].text(x, y, s, fontsize=fontsize)

    # Plot the correlation line
    m, b = np.polyfit(acc, validation, 1)
    x_line = np.linspace(acc.min(), acc.max(), 100)
    y_line = m * x_line + b
    axs[i].plot(x_line, y_line, color='k', linewidth=2, alpha=0.3, zorder=1)

    # Plot title
    axs[i].set_title(f'{titles[i]}', fontsize=fontsize)

    # x-axis parameters
    if i in [4, 5, 6, 7]:
        axs[i].set_xlabel("Pearson's $r$", fontsize=fontsize)
        xticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        xlabels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        axs[i].set_xticks(ticks=xticks, labels=xlabels)
        axs[i].set_xlim(left=0.06, right=.28)

    # y-axis parameters
    axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
    # yticks = [10, 15, 20, 25, 30]
    # ylabels = [10, 15, 20, 25, 30]
    # axs[i].set_yticks(ticks=yticks, labels=ylabels)
    # axs[i].set_ylim(bottom=8, top=29)

    # Legend
    if i == 0:
        axs[i].legend(ncol=3, fontsize=fontsize, loc=0, frameon=False,
            bbox_to_anchor=(3.9, 1.3), markerscale=2)

# Save the figure
file_name = os.path.join(save_dir, f'scatterplots_nsdsynthetic.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)