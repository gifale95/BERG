"""Plot the univariate RNC cross-subject validated results.

Parameters
----------
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
roi_pair : str
    Used pairwise ROI combination.
imagenet_split : str
    Whether to use the 'train' or 'val' split of ILSVRC-2012.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--roi_pair', default='V1-ventral', type=str)
parser.add_argument('--imagenet_split', default='train', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Get the total dataset subjects
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]
rois = [roi_1, roi_2]


# =============================================================================
# Load the neural control results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'stats', f'cv-{args.cv}',
    f'stats_{args.roi_pair}.npy')

res = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Create the plot save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem',
    f'imagenet_split-{args.imagenet_split}', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot parameters
# =============================================================================
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
colors = [(4/255, 178/255, 153/255), (130/255, 201/255, 240/255),
    (217/255, 214/255, 111/255), (214/255, 83/255, 117/255)]


# =============================================================================
# Plot the univariate responses for the controlling images on scatterplots
# =============================================================================
fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10, 10))
axs = np.reshape(axs, (-1))

# Diagonal dashed line
axs[0].plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2, alpha=.4,
    label='_nolegend_')

# Baseline images dashed lines
base_1 = np.mean(res['base_resp'][roi_1])
axs[0].plot([base_1, base_1], [-3, 3], '--w', linewidth=2, alpha=.6,
    label='_nolegend_')
base_2 = np.mean(res['base_resp'][roi_2])
axs[0].plot([-3, 3], [base_2, base_2], '--w', linewidth=2, alpha=.6,
    label='_nolegend_')

# Univariate responses for all images
for s in range(len(all_subjects)):
    axs[0].scatter(res['fmri'][roi_1][s], res['fmri'][roi_2][s], c='w',
        alpha=.1, edgecolors='k', label='_nolegend_')

# Univariate responses for the controlling images
for s in range(len(all_subjects)):
    for key in res['cv_resp_roi_1'][s].keys():
        # 1. high_1_high_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1'][s][key]['high_1_high_2']),
            np.mean(res['cv_resp_roi_2'][s][key]['high_1_high_2']),
            c=colors[0], s=200, alpha=0.8)
        # 2. low_1_low_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1'][s][key]['low_1_low_2']),
            np.mean(res['cv_resp_roi_2'][s][key]['low_1_low_2']),
            c=colors[1], s=200, alpha=0.8)
        # 3. high_1_low_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1'][s][key]['high_1_low_2']),
            np.mean(res['cv_resp_roi_2'][s][key]['high_1_low_2']),
            c=colors[2], s=200, alpha=0.8)
        # 4. low_1_high_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1'][s][key]['low_1_high_2']),
            np.mean(res['cv_resp_roi_2'][s][key]['low_1_high_2']),
            c=colors[3], s=200, alpha=0.8)

# Add the correlation scores the two ROI responses for all images
if args.cv == 0:
    x = -1.8
    y = 0.6
    s = '$r$=' + str(np.round(np.mean(res['roi_pair_corr_control_img']), 2))
    axs[0].text(x, y, s, fontsize=fontsize)

# x-axis parameters
xlabel = f'Univariate response\n{roi_1}'
axs[0].set_xlabel(xlabel, fontsize=fontsize)
xticks = [-1.5, 0, 1.5]
xlabels = [-1.5, 0, 1.5]
axs[0].set_xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=-2, right=2)

# y-axis parameters
ylabel = f'Univariate response\n{roi_2}'
axs[0].set_ylabel(ylabel, fontsize=fontsize)
yticks = [-.5, 0, .5]
ylabels = [-.5, 0, .5]
axs[0].set_yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-1, top=1)

# Aspect
axs[0].set_aspect('equal')

# Title
axs[0].set_title('ILSVRC-2012 (train)\n(10 cats, 4 imgs per control condition)', fontsize=fontsize)
plt.show()

# Save the figure
file_name = f'univariate_rnc_scatterplots_cv-{args.cv}_imagenet-{args.imagenet_split}_{args.roi_pair}.png'
fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
    transparent=False, format='png')
plt.close()


# =============================================================================
# Plot the univariate responses for the controlling images on scatterplots
# (Rotem-selected images)
# =============================================================================
fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10, 10))
axs = np.reshape(axs, (-1))

# Diagonal dashed line
axs[0].plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2, alpha=.4,
    label='_nolegend_')

# Baseline images dashed lines
base_1 = np.mean(res['base_resp'][roi_1])
axs[0].plot([base_1, base_1], [-3, 3], '--w', linewidth=2, alpha=.6,
    label='_nolegend_')
base_2 = np.mean(res['base_resp'][roi_2])
axs[0].plot([-3, 3], [base_2, base_2], '--w', linewidth=2, alpha=.6,
    label='_nolegend_')

# Univariate responses for all images
for s in range(len(all_subjects)):
    axs[0].scatter(res['fmri'][roi_1][s], res['fmri'][roi_2][s], c='w',
        alpha=.1, edgecolors='k', label='_nolegend_')

# Univariate responses for the controlling images
for s in range(len(all_subjects)):
    for key in res['cv_resp_roi_1_rotem'][s].keys():
        # 1. high_1_high_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1_rotem'][s][key]['high_1_high_2']),
            np.mean(res['cv_resp_roi_2_rotem'][s][key]['high_1_high_2']),
            c=colors[0], s=200, alpha=0.8)
        # 2. low_1_low_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1_rotem'][s][key]['low_1_low_2']),
            np.mean(res['cv_resp_roi_2_rotem'][s][key]['low_1_low_2']),
            c=colors[1], s=200, alpha=0.8)
        # 3. high_1_low_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1_rotem'][s][key]['high_1_low_2']),
            np.mean(res['cv_resp_roi_2_rotem'][s][key]['high_1_low_2']),
            c=colors[2], s=200, alpha=0.8)
        # 4. low_1_high_2
        axs[0].scatter(
            np.mean(res['cv_resp_roi_1_rotem'][s][key]['low_1_high_2']),
            np.mean(res['cv_resp_roi_2_rotem'][s][key]['low_1_high_2']),
            c=colors[3], s=200, alpha=0.8)

# Add the correlation scores the two ROI responses for all images
if args.cv == 0:
    x = -1.8
    y = 0.6
    s = '$r$=' + str(np.round(np.mean(res['roi_pair_corr_control_img_rotem']), 2))
    axs[0].text(x, y, s, fontsize=fontsize)

# x-axis parameters
xlabel = f'Univariate response\n{roi_1}'
axs[0].set_xlabel(xlabel, fontsize=fontsize)
xticks = [-1.5, 0, 1.5]
xlabels = [-1.5, 0, 1.5]
axs[0].set_xticks(ticks=xticks, labels=xlabels)
axs[0].set_xlim(left=-2, right=2)

# y-axis parameters
ylabel = f'Univariate response\n{roi_2}'
axs[0].set_ylabel(ylabel, fontsize=fontsize)
yticks = [-.5, 0, .5]
ylabels = [-.5, 0, .5]
axs[0].set_yticks(ticks=yticks, labels=ylabels)
axs[0].set_ylim(bottom=-1, top=1)

# Aspect
axs[0].set_aspect('equal')

# Title
axs[0].set_title('ILSVRC-2012 (train)\n(10 cats, 4 imgs per control condition)', fontsize=fontsize)
plt.show()

# Save the figure
file_name = f'univariate_rnc_scatterplots_cv-{args.cv}_imagenet-{args.imagenet_split}_{args.roi_pair}_rotem_images.png'
fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
    transparent=False, format='png')
plt.close()