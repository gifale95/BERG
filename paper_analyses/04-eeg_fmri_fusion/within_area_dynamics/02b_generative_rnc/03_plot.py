"""Plot the generative univariate RNC results.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
roi_pair : int
    Integer indicating the chosen pairwise ROI combination on which to perform
    multivariate RNC. Possible values are '0' (V1-V2), '1' (V1-V3), '2'
    (V1-hV4), '3' (V2-V3), '4' (V2-hV4), '5' (V3-hV4).
evolutions : list
    Genetic optimization evolutions for which the results are plotted.
image_generator_name : str
    Name of the used image generator. Available options are 'DeePSiM' (a GAN).
    or 'cd_imagenet64_l2' (a class-conditioned diffusion model trained on the
    1000 ILSVRC-2012 classes).
image_generator_class : int
    Integer between 0 and 999 indicating the ILSVRC-2012 class the generated
    image belongs to (if image_generator_name=='cd_imagenet64_l2').
baseline_margin : float
    Margin to be added to the baseline univariate response score to define the
    neural control threshold. If set to None, keep on optimizing the neural
    control scores, without optimizing image complexity.
imageset : str
    Imageset from which the univariate RNC baseline scores have been computed.
    Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--roi_pair', type=str, default='V1-hV4')
parser.add_argument('--evolutions', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--image_generator_name', type=str, default='DeePSiM')
parser.add_argument('--image_generator_class', type=int, default=0)
parser.add_argument('--baseline_margin', type=float, default=0.6)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]

if roi_2 == 'hV4':
    roi_2 = 'V4'
if roi_1 == 'hV4':
    roi_1 = 'V4'


# =============================================================================
# Neural control types
# =============================================================================
control_types = ['high_1_high_2', 'high_1_low_2', 'low_1_high_2', 'low_1_low_2']


# =============================================================================
# Load the generative univariate RNC results
# =============================================================================
fmri_roi_1 = {}
fmri_roi_2 = {}
baseline_penalty_train = {}
images_complexity = {}

for c in control_types:

    fmri_roi_1_evo = []
    fmri_roi_2_evo = []
    baseline_penalty_train_evo = []
    images_complexity_evo = []

    for e in args.evolutions:

        data_dir = os.path.join(args.project_dir, 'generative_univariate_rnc',
            'optimization_scores', 'cv-0', args.roi_pair, 'control_condition-'+
            c, 'image_generator-'+args.image_generator_name, 'evolution-'+
            format(e, '02'), 'baseline_margin-'+str(args.baseline_margin),
            'optimization_scores.npy')

        data_dict = np.load(data_dir, allow_pickle=True).item()

        fmri_roi_1_evo.append(np.mean(np.squeeze(
            data_dict['best_fmri_roi_1']), 1))
        fmri_roi_2_evo.append(np.mean(np.squeeze(
            data_dict['best_fmri_roi_2']), 1))
        baseline_penalty_train_evo.append(np.squeeze(
            data_dict['best_baseline_penalty_train']))
        images_complexity_evo.append(np.squeeze(
            data_dict['best_images_complexity']) / 1024) # Convert bytes to kB

    fmri_roi_1[c] = np.asarray(fmri_roi_1_evo)
    fmri_roi_2[c] = np.asarray(fmri_roi_2_evo)
    baseline_penalty_train[c] = np.asarray(baseline_penalty_train_evo)
    images_complexity[c] = np.asarray(images_complexity_evo)

    del fmri_roi_1_evo, fmri_roi_2_evo, baseline_penalty_train_evo, \
        images_complexity_evo


# =============================================================================
# Get the null distribution threshold generation
# =============================================================================
threshold = {}

for key, val in baseline_penalty_train.items():
    threshold_evo = []
    for e in range(len(args.evolutions)):
        idx = np.where(val[e] == 1e+10)[0]
        threshold_evo.append(idx[-1] + 1)
    threshold[key] = np.asarray(threshold_evo)


# =============================================================================
# Compute the confidence intervals (across evolutions)
# =============================================================================
n_iter = 1000

ci_fmri_roi_1 = {}
ci_fmri_roi_2 = {}
ci_images_complexity = {}

for c in tqdm(control_types):

    # Empty confidence intervals arrays
    ci_fmri_roi_1_c = np.zeros((2, fmri_roi_1[c].shape[1]))
    ci_fmri_roi_2_c = np.zeros(ci_fmri_roi_1_c.shape)
    ci_images_complexity_c = np.zeros(ci_fmri_roi_1_c.shape)

    for g in tqdm(range(fmri_roi_1[c].shape[1])):

        # Empty CI distribution arrays
        ci_fmri_roi_1_dist = np.zeros((n_iter))
        ci_fmri_roi_2_dist = np.zeros((n_iter))
        ci_images_complexity_dist = np.zeros((n_iter))

        # Compute the CI distributions
        for i in range(n_iter):
            idx_resample = resample(np.arange(len(fmri_roi_1[c])))
            ci_fmri_roi_1_dist[i] = np.mean(fmri_roi_1[c][idx_resample,g])
            ci_fmri_roi_2_dist[i] = np.mean(fmri_roi_2[c][idx_resample,g])
            ci_images_complexity_dist[i] = np.mean(
                images_complexity[c][idx_resample,g])

        # Get the 5th and 95th CI distributions percentiles
        ci_fmri_roi_1_c[0,g] = np.percentile(ci_fmri_roi_1_dist, 2.5)
        ci_fmri_roi_1_c[1,g] = np.percentile(ci_fmri_roi_1_dist, 97.5)
        ci_fmri_roi_2_c[0,g] = np.percentile(ci_fmri_roi_2_dist, 2.5)
        ci_fmri_roi_2_c[1,g] = np.percentile(ci_fmri_roi_2_dist, 97.5)
        ci_images_complexity_c[0,g] = np.percentile(
            ci_images_complexity_dist, 2.5)
        ci_images_complexity_c[1,g] = np.percentile(
            ci_images_complexity_dist, 97.5)

    ci_fmri_roi_1[c] = ci_fmri_roi_1_c
    ci_fmri_roi_2[c] = ci_fmri_roi_2_c
    ci_images_complexity[c] = ci_images_complexity_c


# =============================================================================
# Load the univariate RNC stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
    'nsd_encoding_models', 'stats', 'cv-0', 'imageset-'+args.imageset,
    args.roi_pair, 'stats.npy')

stats = np.load(data_dir, allow_pickle=True).item()

uni_resp = stats['uni_resp']
baseline_images_score = stats['baseline_resp']


# =============================================================================
# Set the plot parameters
# =============================================================================
fontsize = 30
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = False
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(4/255, 178/255, 153/255), (217/255, 214/255, 111/255),
    (214/255, 83/255, 117/255), (130/255, 201/255, 240/255)]


# =============================================================================
# Plot the generative univariate RNC optimization curves
# =============================================================================
for c, control in enumerate(control_types):

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3))
    axs = np.reshape(axs, (-1))

    # Plot
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.set_zorder(3)
    ax2.set_zorder(2)
    ax1.patch.set_visible(False)
    x = np.arange(len(fmri_roi_1[control][0]))
    # Plot the baseline threshold
    ax2.plot([np.mean(threshold[control]), np.mean(threshold[control])],
        [-1e+10, 1e+10], '--', color='grey', label='__no_label__', linewidth=3)
    # Plot the image complexity
    ax2.plot(x, np.mean(images_complexity[control], 0), linewidth=3, color='k',
        label='Image complexity', zorder=1)
    ax2.fill_between(x, ci_images_complexity[control][0],
        ci_images_complexity[control][1], color='k', alpha=.2)
    # Plot the fMRI univariate responses
    ax1.plot(x, np.mean(fmri_roi_1[control], 0), linewidth=3, color=colors[c],
        label=roi_1, zorder=2)
    ax1.fill_between(x, ci_fmri_roi_1[control][0], ci_fmri_roi_1[control][1],
        color=colors[c], alpha=.2)
    ax1.plot(x, np.mean(fmri_roi_2[control], 0), '--', linewidth=3,
        color=colors[c], label=roi_2, zorder=2)
    ax1.fill_between(x, ci_fmri_roi_2[control][0], ci_fmri_roi_2[control][1],
        color=colors[c], alpha=.2)

    # y-axis parameters
    ax1.set_ylabel('Univariate\nresponse', fontsize=fontsize, color=colors[c])
    yticks = [-1, 0, 1]
    ylabels = [-1, 0, 1]
    ax1.set_yticks(ticks=yticks, labels=ylabels, color=colors[c])
    ax1.set_ylim(bottom=-1.25, top=1.25)
    ax2.set_ylabel('PNG file\nsize (kB)', fontsize=fontsize)
    yticks = [50, 100, 150]
    ylabels = [50, 100, 150]
    ax2.set_yticks(ticks=yticks, labels=ylabels)
    ax2.set_ylim(bottom=0, top=175)

    # x-axis parameters
    ax1.set_xlabel('Generations', fontsize=fontsize)
    xticks = [0, 250, 499]
    xlabels = [0, 250, 500]
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.xlim(min(x), max(x))

    # Save the figure
    fig_name = 'generative_univariate_rnc_optimization_curves_' + control + \
        '_' + args.image_generator_name + '_baseline_margin-' + \
        str(args.baseline_margin) + '.svg'
    fig.savefig(fig_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the univariate responses for the generative univariate RNC controlling
# images on scatterplots
# =============================================================================
fig = plt.figure(figsize=(6,6))

# Diagonal dashed line
plt.plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2,
    alpha=.4, label='_nolegend_')

# Baseline images dashed lines
baseline_roi_1 = baseline_images_score[0]
plt.plot([baseline_roi_1, baseline_roi_1], [-3, 3], '--w', linewidth=2,
    alpha=.6, label='_nolegend_')
baseline_roi_2 = baseline_images_score[1]
plt.plot([-3, 3], [baseline_roi_2, baseline_roi_2], '--w', linewidth=2,
    alpha=.6, label='_nolegend_')

# All images
plt.scatter(uni_resp[0], uni_resp[1], s=10, c='w', alpha=.1,
    edgecolors='k', label='_nolegend_')

# Controlling images
for c, control in enumerate(control_types):
    plt.scatter(np.mean(fmri_roi_1[control][:,-1]),
        np.mean(fmri_roi_2[control][:,-1]), s=1000, c=colors[c], alpha=0.8)

# x-axis
xlabel = roi_1 + ' univariate response'
plt.xlabel(xlabel, fontsize=fontsize)
xticks = [-2, -1, 0, 1, 2]
xlabels = [-2, -1, 0, 1, 2]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=-1.5, right=1)

# y-axis
ylabel = roi_2 + ' univariate response'
plt.ylabel(ylabel, fontsize=fontsize)
yticks = [-2, -1, 0, 1, 2]
ylabels = [-2, -1, 0, 1, 2]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-1.5, top=1)

# Title
title = 'Generated images'
plt.title(title, fontsize=fontsize)

# Save the figure
fig_name = 'generative_univariate_rnc_scatterplots_cv-0_' + \
    args.image_generator_name + '_baseline_margin-' + \
    str(args.baseline_margin) + '.svg'
fig.savefig(fig_name, bbox_inches='tight', dpi=1000, transparent=True,
    format='svg')
