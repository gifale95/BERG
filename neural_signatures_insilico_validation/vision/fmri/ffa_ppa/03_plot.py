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
    'vision', 'fmri', 'ffa_ppa', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot parameters
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
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot FFA's sensitivity to face shapes across texture variation
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa', 'stats', 'ffa_face_texture.npy')
data = np.load(data_dir, allow_pickle=True).item()
vertex_mean_resp = data['vertex_mean_resp']
pval_diff = data['pval_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']

# Sort the image types based on their univariate responses
image_types = ['Faces-Animals', 'Faces-Blank', 'Faces-Cartoon',
    'Faces-EyesOnly', 'Faces-Inverted', 'Faces-Rearranged', 'Pareidolia-Face',
    'Objects']
labels = ['Animals', 'Blank', 'Cartoon', 'Eyes only', 'Inverted',
    'Rearranged', 'Pareidolia', 'Objects']
uni_resp = np.empty(0)
for itype in image_types:
    uni_resp = np.append(uni_resp, np.mean(vertex_mean_resp[itype]))
idx_sort = np.argsort(uni_resp)[::-1]
image_types_sorted = [image_types[i] for i in idx_sort]
labels_sorted = [labels[i] for i in idx_sort]

# Plot parameters
n_sub = 8
x_coord = np.arange(len(image_types))
alpha = 0.2
marker = 'o'
s = 500
s_mean = 750
color = 'k'

# Plot the results
fig = plt.figure(figsize=(20,9))
for i, itype in enumerate(image_types_sorted):
    # Univariate response scores
    x = np.repeat(i, n_sub)
    y = vertex_mean_resp[itype]
    plt.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors='none')
    plt.scatter(x[0], np.mean(y), s=s_mean, color=color, edgecolors='none')
    # Confidence intervals
    ci = np.zeros(2)
    ci[0] = np.mean(y) - ci_vertex_mean_resp[itype][0]
    ci[1] = ci_vertex_mean_resp[itype][1] - np.mean(y)
    plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)), fmt="none",
        ecolor=color, elinewidth=5, capsize=0)
# x-axis parameters
xticks = x_coord
plt.xticks(ticks=xticks, labels=labels_sorted, rotation=-30, ha='center')
plt.xlim(left=-0.5, right=7.5)
# y-axis parameters
yticks = [-1, -0.5, 0, 0.5, 1]
ylabels = ['-1', '-0.5', '0', '0.5', '1']
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
ylabel = 'Univariate response ($z$-scored)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=-.5, top=1)

# Save the figure
file_name = os.path.join(save_dir, 'ffa_face_texture_variation.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
    format='svg')


# =============================================================================
# Plot PPA's sensitivity to spatial layout information
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa', 'stats', 'ppa_spatial_layout.npy')
data = np.load(data_dir, allow_pickle=True).item()
vertex_mean_resp = data['vertex_mean_resp']
pval_diff = data['pval_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']

# Sort the image types based on their univariate responses
image_types = ['Objects-MultFurniture', 'Objects-SingleFurniture',
    'Scenes-EmptyRooms', 'Scenes-Rearranged']
labels = ['Multiple\nfurniture', 'Furniture', 'Empty\nroom', 'Surfaces']
uni_resp = np.empty(0)
for itype in image_types:
    uni_resp = np.append(uni_resp, np.mean(vertex_mean_resp[itype]))
idx_sort = np.argsort(uni_resp)[::-1]
image_types_sorted = [image_types[i] for i in idx_sort]
labels_sorted = [labels[i] for i in idx_sort]

# Plot parameters
n_sub = 8
x_coord = np.arange(len(image_types))
alpha = 0.2
marker = 'o'
s = 500
s_mean = 750
color = 'k'

# Plot the results
fig = plt.figure(figsize=(12,9))
for i, itype in enumerate(image_types_sorted):
    # Univariate response scores
    x = np.repeat(i, n_sub)
    y = vertex_mean_resp[itype]
    plt.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors='none')
    plt.scatter(x[0], np.mean(y), s=s_mean, color=color, edgecolors='none')
    # Confidence intervals
    ci = np.zeros(2)
    ci[0] = np.mean(y) - ci_vertex_mean_resp[itype][0]
    ci[1] = ci_vertex_mean_resp[itype][1] - np.mean(y)
    plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)), fmt="none",
        ecolor=color, elinewidth=5, capsize=0)
# x-axis parameters
xticks = x_coord
plt.xticks(ticks=xticks, labels=labels_sorted, rotation=-30, ha='center')
plt.xlim(left=-0.5, right=3.5)
# y-axis parameters
yticks = [-0.25, 0, 0.25, 0.5, 0.75]
ylabels = ['-0.25', '0', '0.25', '0.5', '0.75']
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
ylabel = 'Univariate response ($z$-scored)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=-.25, top=.75)

# Save the figure
file_name = os.path.join(save_dir, 'ppa_spatial_layout.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
    format='svg')


# =============================================================================
# Plot FFA and PPA's curvature preferences
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa', 'stats', 'curvature.npy')
data = np.load(data_dir, allow_pickle=True).item()
vertex_mean_resp = data['vertex_mean_resp']
pval_diff = data['pval_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']

# Image types
image_types = ['Curvature-Shapes-Circles', 'Curvature-Shapes-Diamonds',
    'Curvature-Tex-Curvy', 'Curvature-Tex-Rectilinear', 'Curvature-Obj-Curvy',
    'Curvature-Obj-Rectilinear']
labels = ['Curved', 'Rectilinear', 'Curved', 'Rectilinear', 'Curved',
    'Rectilinear']

# Plot parameters
n_sub = 8
x_coord = np.arange(len(image_types))
alpha = 0.2
marker = 'o'
s = 500
s_mean = 750
color = 'k'

# Loop across ROIs
for roi in ['FFA', 'PPA']:

    # Plot the results
    fig = plt.figure(figsize=(15,9))
    for i, itype in enumerate(image_types):
        # Univariate response scores
        x = np.repeat(i, n_sub)
        y = vertex_mean_resp[roi][itype]
        plt.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors='none')
        plt.scatter(x[0], np.mean(y), s=s_mean, color=color, edgecolors='none')
        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - ci_vertex_mean_resp[roi][itype][0]
        ci[1] = ci_vertex_mean_resp[roi][itype][1] - np.mean(y)
        plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)), fmt="none",
            ecolor=color, elinewidth=5, capsize=0)
    # x-axis parameters
    xticks = x_coord
    plt.xticks(ticks=xticks, labels=labels, rotation=-30, ha='center')
    plt.xlim(left=-0.5, right=5.5)
    # y-axis parameters
    yticks = [-0.5, -0.25, 0, 0.25]
    ylabels = ['-0.5', '-0.25', '0', '0.25']
    plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
    ylabel = 'Univariate response ($z$-scored)'
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.ylim(bottom=-.5, top=.05)

    # Save the figure
    file_name = os.path.join(save_dir, roi+'_curvature.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
        format='svg')


# =============================================================================
# Plot FFA and PPA's visual size versus category effects
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'ffa_ppa', 'stats', 'visual_size.npy')
data = np.load(data_dir, allow_pickle=True).item()
vertex_mean_resp = data['vertex_mean_resp']
pval_diff = data['pval_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']

# Image types
image_types = {}
image_types['FFA'] = ['Faces', 'Objects', 'Faces-Small', 'Objects-Small']
image_types['PPA'] = ['Scenes', 'Objects', 'Scenes-Small', 'Objects-Small']
labels = {}
labels['FFA'] = ['Faces', 'Objects', 'Faces', 'Objects']
labels['PPA'] = ['Scenes', 'Objects', 'Scenes', 'Objects']

# Plot parameters
n_sub = 8
x_coord = np.arange(len(image_types['FFA']))
alpha = 0.2
marker = 'o'
s = 500
s_mean = 750
color = 'k'

# Loop across ROIs
for roi in ['FFA', 'PPA']:

    # Plot the results
    fig = plt.figure(figsize=(10,9))
    for i, itype in enumerate(image_types[roi]):
        # Univariate response scores
        x = np.repeat(i, n_sub)
        y = vertex_mean_resp[roi][itype]
        plt.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors='none')
        plt.scatter(x[0], np.mean(y), s=s_mean, color=color, edgecolors='none')
        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - ci_vertex_mean_resp[roi][itype][0]
        ci[1] = ci_vertex_mean_resp[roi][itype][1] - np.mean(y)
        plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)), fmt="none",
            ecolor=color, elinewidth=5, capsize=0)
    # x-axis parameters
    xticks = x_coord
    plt.xticks(ticks=xticks, labels=labels[roi], rotation=-30, ha='center')
    plt.xlim(left=-0.5, right=3.5)
    # y-axis parameters
    yticks = [-0.5, 0, 0.5, 1]
    ylabels = ['-0.5', '0', '0.5', '1']
    plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
    ylabel = 'Univariate response ($z$-scored)'
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.ylim(bottom=-.5, top=1)

    # Save the figure
    file_name = os.path.join(save_dir, roi+'_visual_size.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True, # type: ignore
        format='svg')