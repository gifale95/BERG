"""Plot explanation accuracy as a function of prediction accuracy, for in
silico EEG responses.

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
    'eeg', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'relationship_prediction_explanation', 'eeg', 'stats', 'stats.npy')

results = np.load(results_dir, allow_pickle=True).item()

encoding_models = np.array(results['encoding_models'])
eeg_subjects = np.array(results['eeg_subjects'])
prediction_accuracy = results['prediction_accuracy']
insilico_validation_scores = results['insilico_validation_scores']
corr = results['corr']


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
    'N170 faces',
    'Object decoding dynamics',
    'AlexNet layerwise modeling',
    'LLM modeling',
    'Behavioral modeling'
]

y_labels = [
    'Δ μV',
    "Δ Decoding peak time (ms)",
    "Spearman's $ρ$",
    "Δ Pearson's $r$",
    "Δ Pearson's $r$"
]

signatures = [
    'erp_diff_avg',
    'decoding_peaks_diff',
    'corr_dnn_layer_eeg_times',
    'diff_llm_rsa_late_early',
    'diff_beh_rsa_late_early'
]


# =============================================================================
# Plot the results
# =============================================================================
# Print the average correlation score across all the dynamics of vision
correlation_avg = np.mean([val[0] for val in corr.values()])
print(f'Average correlation score: {correlation_avg:.4f}')

fig, axs = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(30, 17.5))
axs = np.reshape(axs, -1)

fig.supylabel("Explanation accuracy", fontsize=fontsize, x=0.05)
fig.supxlabel("Prediction accuracy", fontsize=fontsize)

for i, key in enumerate(signatures):

    val = insilico_validation_scores[key]

    # Enforce same length of x- and y-axes
    axs[i].set_box_aspect(1)

    # Scatter plot of the insilico prediction accuracy vs. explanation accuracy
    acc = np.empty(0)
    validation = np.empty(0)
    for m in range(len(encoding_models)):
        axs[i].scatter(prediction_accuracy[m], val[m], s=200, color=colors[m],
            label=f'{encoding_models[m][17:]}', alpha=0.75, zorder=2)
        acc = np.append(acc, prediction_accuracy[m])
        validation = np.append(validation, val[m])

    # Plot the subject connection lines
    acc_array = np.array(prediction_accuracy)
    val_array = np.array(val)
    for s in range(len(eeg_subjects)):
        axs[i].plot(acc_array[:,s], val_array[:,s], color='k', linewidth=1,
            alpha=.1, zorder=1)

    # Print the correlation score between prediction accuracy and explanation
    # accuracy
    x = 0.35
    y = min(validation) + (max(validation) - min(validation)) * 0.05
    if corr[key][1] < 0.0001:
        s = f'$r$={np.round(corr[key][0], 2):0.2f}***'
    elif corr[key][1] < 0.001:
        s = f'$r$={np.round(corr[key][0], 2):0.2f}**'
    elif corr[key][1] < 0.01:
        s = f'$r$={np.round(corr[key][0], 2):0.2f}*'
    else:
        s = f'$r$={np.round(corr[key][0], 2):0.2f}'
    axs[i].text(x, y, s, fontsize=fontsize)

    # Plot the correlation line
    m, b = np.polyfit(acc, validation, 1)
    x_line = np.linspace(acc.min(), acc.max(), 100)
    y_line = m * x_line + b
    axs[i].plot(x_line, y_line, color='k', linewidth=2, alpha=0.3, zorder=1)

    # Plot title
    axs[i].set_title(f'{titles[i]}', fontsize=fontsize)

    # x-axis parameters
    if i in [3, 4, 5]:
        axs[i].set_xlabel("Pearson's $r$", fontsize=fontsize)
        xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        xlabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        axs[i].set_xticks(ticks=xticks, labels=xlabels)
        axs[i].set_xlim(left=.06, right=.53)

    # y-axis parameters
    axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
    # yticks = [10, 15, 20, 25, 30]
    # ylabels = [10, 15, 20, 25, 30]
    # axs[i].set_yticks(ticks=yticks, labels=ylabels)
    # axs[i].set_ylim(bottom=8, top=29)

    # Legend
    if i == 0:
        axs[i].legend(ncol=3, fontsize=fontsize, loc=0, frameon=False,
            bbox_to_anchor=(2.5, 1.3), markerscale=2)

# Save the figure
file_name = os.path.join(save_dir, f'scatterplots.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)