"""Plot the vertex-mean responses of high-level visual cortex ROIs for images
of different categories.

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
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'hvc_selectivity_univariate_responses', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'hvc_selectivity_univariate_responses', 'stats',
    'stats.npy')

data = np.load(data_dir, allow_pickle=True).item()

insilico_fmri = data['insilico_fmri']
vertex_mean_resp = data['vertex_mean_resp']
pval_cat_diff = data['pval_cat_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']


# =============================================================================
# Plot the vertex-mean responses of each ROI # !!!
# =============================================================================
categories = ['Bodies', 'Faces', 'Objects', 'Scenes']
rois = ['EBA', 'FBA', 'FFA', 'OFA', 'PPA', 'OPA', 'RSC']
n_sub = 8

# Plot parameters
x_coord = np.arange(len(rois))
dist = 0.3
x_dist = np.asarray((-0.75, -0.25, 0.25, 0.75)) * dist
x_dist_sig = np.asarray((-.75, -0.25, 0.25, .75)) * dist
alpha = 0.2
fontsize_sig = 20
marker = 'o'
s = 500
s_mean = 750
sig_offset = 7
sig_bar_length = 3
linewidth_sig_bar = 1
sig_star_offset_top = 2
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
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(204/255, 102/255, 119/255), (230/255, 159/255, 0/255),
    (86/255, 180/255, 233/255), (17/255, 119/255, 51/255)]

# Plot
fig = plt.figure(figsize=(20,9))

for r, roi in enumerate(rois):
    for c, cat in enumerate(categories):

        # Univariate response scores
        x = np.repeat(r+x_dist[c], n_sub)
        y = vertex_mean_resp[roi+'_'+cat]
        plt.scatter(x, y, s=s, color=colors[c], alpha=alpha,
            edgecolors='none', label='_nolegend_')
        if r == 0:
            plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[c],
            edgecolors='none', label=cat)
        else:
            plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[c],
            edgecolors='none', label='_nolegend_')

        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - ci_vertex_mean_resp[roi+'_'+cat][0]
        ci[1] = ci_vertex_mean_resp[roi+'_'+cat][1] - np.mean(y)
        plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)),
            fmt="none", ecolor=colors[c], elinewidth=5, capsize=0)

# Significance 1 # !!! ADD
# if all(sig_gt1tr_gt1tr_vs_gt1tr_gt2tr < 0.05):
#     res = np.append(acc_gt1tr_gt1tr, acc_gt1tr_gt2tr)
#     y_max = max(res) + sig_offset
#     plt.plot([x_coord[0], x_coord[0]], [y_max, y_max+sig_bar_length],
#         'k-', [x_coord[0], x_coord[1]],
#         [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
#         [x_coord[1], x_coord[1]], [y_max+sig_bar_length, y_max], 'k-',
#         linewidth=linewidth_sig_bar)
#     x_mean = np.mean(np.asarray((x_coord[0], x_coord[1])))
#     y = y_max + sig_bar_length + sig_star_offset_top
#     for r, roi in enumerate(evc_rois):
#         plt.text(x_mean+x_dist_sig[r], y, s='*', fontsize=fontsize_sig,
#             color=colors_2[r], fontweight='bold', ha='center', va='center')

# Significance 2 # !!! ADD
# if all(sig_gt1tr_gt2tr_vs_gt1tr_synt < 0.05):
#     res = np.append(acc_gt1tr_gt2tr, acc_gt1tr_synt)
#     y_max = max(res) + sig_offset
#     plt.plot([x_coord[1], x_coord[1]], [y_max, y_max+sig_bar_length],
#         'k-', [x_coord[1], x_coord[2]],
#         [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
#         [x_coord[2], x_coord[2]], [y_max+sig_bar_length, y_max], 'k-',
#         linewidth=linewidth_sig_bar)
#     x_mean = np.mean(np.asarray((x_coord[1], x_coord[2])))
#     y = y_max + sig_bar_length + sig_star_offset_top
# for r, roi in enumerate(evc_rois):
#     plt.text(x_mean+x_dist_sig[r], y, s='*', fontsize=fontsize_sig,
#         color=colors_2[r], fontweight='bold', ha='center', va='center')

# x-axis parameters
xticks = x_coord
plt.xticks(ticks=xticks, labels=rois, rotation=0)
xlabel = 'ROIs'
#plt.xlabel(xlabel, fontsize=fontsize)
plt.xlim(left=-0.5, right=6.5)

# y-axis parameters
yticks = [-1, -0.5, 0, 0.5, 1]
ylabels = ['-1', '-0.5', '0', '0.5', '1']
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
ylabel = 'Univariate response ($z$-scored)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=-.85, top=.85)

# Legend
plt.legend(loc=2, ncol=2, fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'roi_univariate_resposes.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
    format='svg')