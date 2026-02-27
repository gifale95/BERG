"""Plot the tripartite organization effect (Konkle & Caramazza, 2013) results.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
images : str
    Whether to use 'naturalistic' or 'texforms' images.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import cortex
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-vit_b_32')
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--images', type=str, default='naturalistic')
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'tripartite_organization', 'stats', args.encoding_model,
    'stats_images-'+args.images+'.npy')

data = np.load(data_dir, allow_pickle=True).item()

vertex_overlap = data['vertex_overlap']
pval_vertex_overlap = data['pval_vertex_overlap']
ci_vertex_overlap = data['ci_vertex_overlap']
lh_tripartite_organization_sub_avg = data['lh_tripartite_organization_sub_avg']
rh_tripartite_organization_sub_avg = data['rh_tripartite_organization_sub_avg']
lh_tripartite_organization_sub_single = data['lh_tripartite_organization_sub_single']
rh_tripartite_organization_sub_single = data['rh_tripartite_organization_sub_single']


# =============================================================================
# Plot the ROI vertex overlap with categorical zones
# =============================================================================
rois = ['FFA', 'OFA', 'EBA', 'FBA', 'PPA', 'OPA', 'RSC']
categories = ['animals', 'big_objects', 'small_objects']
n_sub = 8

# Plot parameters
x_coord = np.arange(len(rois))
dist = 0.4
x_dist = np.asarray((-0.5, 0, 0.5)) * dist
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
category_labels = ['Animals', 'Big objects', 'Small objects']
fontsize = 30
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
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
colors = [(143/255, 25/255, 250/255), (43/255, 141/255, 248/255),
    (243/255, 85/255, 20/255)]

# Plot
fig = plt.figure(figsize=(20,9))

for r, roi in enumerate(rois):
    for c, cat in enumerate(categories):

        # Vertex overlap scores
        x = np.repeat(r+x_dist[c], n_sub)
        y = vertex_overlap[roi+'_'+cat]
        plt.scatter(x, y, s=s, color=colors[c], alpha=alpha,
            edgecolors='none', label='_nolegend_')
        if r == 0:
            plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[c],
            edgecolors='none', label=category_labels[c])
        else:
            plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[c],
            edgecolors='none', label='_nolegend_')

        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - ci_vertex_overlap[roi+'_'+cat][0]
        ci[1] = ci_vertex_overlap[roi+'_'+cat][1] - np.mean(y)
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
yticks = [0, 20, 40, 60, 80, 100]
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Vertex overlap (%)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=100)

# Legend
plt.legend(loc=2, ncol=2, fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'vertex_overlap_images-'+args.images+'.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
    format='svg')


# =============================================================================
# Plot the tripartite organization results on a brain surface (subject average)
# =============================================================================
# Plot parameters
subject = 'fsaverage_nsd_sub-01'
custom_cmap = mcolors.ListedColormap([(103/255, 78/255, 167/255),
    (90/255, 130/255, 200/255), (230/255, 135/255, 60/255)])

# Append the results across left and right hemispheres
data = np.append(lh_tripartite_organization_sub_avg,
    rh_tripartite_organization_sub_avg)

# Create the surface maps
vertex_data = cortex.Vertex(data, subject, cmap=custom_cmap, vmin=0, vmax=2,
    with_colorbar=False)

# Plot the results on a flat surface
fig = cortex.quickshow(vertex_data,
    height=2000, # Increase resolution of map and ROI contours
    with_curvature=True,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    linewidth=3,
    linecolor=(1, 1, 1),
    with_labels=True,
    labelsize=25,
    curvature_brightness=0.4,
    with_colorbar=False
    )

# Save the figure
file_name = os.path.join(save_dir,
    f'tripartite_organization_sub-avg_flat_images-{args.images}.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
    format='svg')
plt.close()

# Plot results on inflated surfaces # !!! DELETE?
#import nilearn
#from nilearn.plotting import view_surf
#from nilearn.datasets import load_fsaverage # type: ignore
# # Get the fsaverage mesh
# fsaverage_meshes = load_fsaverage(mesh='fsaverage')
# # Create the inflated surface plot
# view = view_surf(
#     surf_mesh=fsaverage_meshes["inflated"],
#     surf_map=data,
#     hemi="both", # type: ignore
#     title=None
# )
# view
# view.save_as_html("inflated_surface_plot.html")
# # Save the figure
# file_name = os.path.join(save_dir, 'tripartite_organization_inflated_images-'+
#     args.images+'.svg')
# fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
#     format='svg')


# =============================================================================
# Plot the tripartite organization results on a brain surface (single subjects)
# =============================================================================
# Plot parameters
custom_cmap = mcolors.ListedColormap([(103/255, 78/255, 167/255),
    (90/255, 130/255, 200/255), (230/255, 135/255, 60/255)])

# Loop across fMRI subjects
for s, sub in enumerate(tqdm(args.fmri_subjects)):

    # Append the results across left and right hemispheres
    data = np.append(lh_tripartite_organization_sub_single[s],
        rh_tripartite_organization_sub_single[s])

    # Create the surface maps
    subject = 'fsaverage_nsd_sub-0' + str(sub)
    vertex_data = cortex.Vertex(data, subject, cmap=custom_cmap, vmin=0,
        vmax=2, with_colorbar=False)

    # Plot the results on a flat surface
    fig = cortex.quickshow(vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'FFA-1', 'FFA-2', 'OFA', 'EBA', 'PPA', 'OPA', 'RSC'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=25,
        curvature_brightness=0.4,
        with_colorbar=False
        )

    # Save the figure
    file_name = os.path.join(save_dir,
        f'tripartite_organization_sub-0{sub}_flat_images-{args.images}.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
        format='svg')
    plt.close()