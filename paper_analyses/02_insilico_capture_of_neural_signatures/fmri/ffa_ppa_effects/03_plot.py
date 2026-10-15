"""Plot the vertex-mean responses of high-level visual cortex ROIs for images
of different categories.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'plots', args.encoding_model)
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


# =============================================================================
# Plot FFA and PPA's visual size versus category effects
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'stats', args.encoding_model, 'visual_size.npy')
data = np.load(data_dir, allow_pickle=True).item()
vertex_mean_resp = data['vertex_mean_resp']
pval_diff = data['pval_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']

# Image types
image_types = {}
image_types['FFA'] = ['Faces', 'Objects', 'Faces-Small', 'Objects-Small']
image_types['PPA'] = ['Scenes', 'Objects', 'Scenes-Small', 'Objects-Small']
labels = {}
labels['FFA'] = ['Faces\n(big)', 'Objects\n(big)', 'Faces\n(small)',
    'Objects\n(small)']
labels['PPA'] = ['Scenes\n(big)', 'Objects\n(big)', 'Scenes\n(small)',
    'Objects\n(small)']

# Significance types
sig_types_ffa = ['FFA_Faces>Objects', 'FFA_Faces-Small>Objects',
    'FFA_Faces-Small>Objects-Small']
sig_types_ppa = ['PPA_Scenes>Objects', 'PPA_Scenes-Small>Objects',
    'PPA_Scenes-Small>Objects-Small']

# Plot parameters
n_sub = 8
x_coord = np.arange(len(image_types['FFA']))
alpha = 0.2
marker = 'o'
s = 500
s_mean = 750
color = 'k'

# Create the plot figure
fig, axs = plt.subplots(1, 2, sharex=False, sharey=True,
    figsize=(17.5, 5))
axs = np.reshape(axs, -1)

# Loop across ROIs
for r, roi in enumerate(['FFA', 'PPA']):

    # Loop across image types
    for i, itype in enumerate(image_types[roi]):

        # Univariate response scores
        x = np.repeat(i, n_sub)
        y = vertex_mean_resp[roi][itype]
        axs[r].scatter(x, y, s=s, color=color, alpha=alpha,
            edgecolors='none')
        axs[r].scatter(x[0], np.mean(y), s=s_mean, color=color,
            edgecolors='none')

        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - ci_vertex_mean_resp[roi][itype][0]
        ci[1] = ci_vertex_mean_resp[roi][itype][1] - np.mean(y)
        axs[r].errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)),
            fmt="none", ecolor=color, elinewidth=5, capsize=0)

    # Significance (FFA)
    if roi == 'FFA':
        for sig_type in sig_types_ffa:
            if pval_diff[sig_type] < 0.0001:
                sig = '***'
            elif pval_diff[sig_type] < 0.001:
                sig = '**'
            elif pval_diff[sig_type] < 0.01:
                sig = '*'
            else:
                sig = ''
            if sig_type == 'FFA_Faces>Objects' and sig != '':
                x = [0, 1]
                y = [np.mean(vertex_mean_resp[roi]['Faces']),
                    np.mean(vertex_mean_resp[roi]['Objects'])]
            if sig_type == 'FFA_Faces-Small>Objects' and sig != '':
                x = [1, 2]
                y = [np.mean(vertex_mean_resp[roi]['Objects']),
                    np.mean(vertex_mean_resp[roi]['Faces-Small'])]
            if sig_type == 'FFA_Faces-Small>Objects-Small' and sig != '':
                x = [2, 3]
                y = [np.mean(vertex_mean_resp[roi]['Faces-Small']),
                    np.mean(vertex_mean_resp[roi]['Objects-Small'])]
            axs[r].plot(x, y, color='k', linewidth=2, alpha=0.5)
            axs[r].text(np.mean(x), np.mean(y), sig, fontsize=fontsize,
                ha='center', va='bottom', rotation_mode='anchor',
                rotation=np.degrees(np.arctan2(y[1]-y[0], x[1]-x[0])),
                transform_rotates_text=True)

    # Significance (PPA)
    if roi == 'PPA':
        for sig_type in sig_types_ppa:
            if pval_diff[sig_type] < 0.0001:
                sig = '***'
            elif pval_diff[sig_type] < 0.001:
                sig = '**'
            elif pval_diff[sig_type] < 0.01:
                sig = '*'
            else:
                sig = ''
            if sig_type == 'PPA_Scenes>Objects' and sig != '':
                x = [0, 1]
                y = [np.mean(vertex_mean_resp[roi]['Scenes']),
                    np.mean(vertex_mean_resp[roi]['Objects'])]
            if sig_type == 'PPA_Scenes-Small>Objects' and sig != '':
                x = [1, 2]
                y = [np.mean(vertex_mean_resp[roi]['Objects']),
                    np.mean(vertex_mean_resp[roi]['Scenes-Small'])]
            if sig_type == 'PPA_Scenes-Small>Objects-Small' and sig != '':
                x = [2, 3]
                y = [np.mean(vertex_mean_resp[roi]['Scenes-Small']),
                    np.mean(vertex_mean_resp[roi]['Objects-Small'])]
            axs[r].plot(x, y, color='k', linewidth=2, alpha=0.5)
            axs[r].text(np.mean(x), np.mean(y), sig, fontsize=fontsize,
                ha='center', va='bottom', rotation_mode='anchor',
                rotation=np.degrees(np.arctan2(y[1]-y[0], x[1]-x[0])),
                transform_rotates_text=True)

    # Title
    axs[r].set_title(roi, fontsize=fontsize)

    # x-axis parameters
    xticks = x_coord
    axs[r].set_xticks(ticks=xticks, labels=labels[roi], rotation=0,
        ha='center')
    axs[r].set_xlim(left=-0.5, right=3.5)

    # y-axis parameters
    if r == 0:
        yticks = [-0.5, 0, 0.5, 1]
        ylabels = ['-0.5', '0', '0.5', '1']
        axs[r].set_yticks(ticks=yticks, labels=ylabels)
        ylabel = 'Univariate response\n($z$-scored)'
        axs[r].set_ylabel(ylabel, fontsize=fontsize)
        axs[r].set_ylim(bottom=-.5, top=1)

# Save the figure
file_name = os.path.join(save_dir, 'visual_size.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot FFA and PPA's curvature preferences
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'stats', args.encoding_model, 'curvature.npy')
data = np.load(data_dir, allow_pickle=True).item()
vertex_mean_resp = data['vertex_mean_resp']
pval_diff = data['pval_diff']
ci_vertex_mean_resp = data['ci_vertex_mean_resp']

# Image types
image_types = ['Curvature-Tex-Curvy', 'Curvature-Tex-Rectilinear',
    'Curvature-Obj-Curvy', 'Curvature-Obj-Rectilinear']
labels = ['Textures\n(curv.)', 'Textures\n(rect.)', 'Objects\n(curv.)',
    'Objects\n(rect.)']

# Significance types
sig_types_ffa = ['FFA_Tex-Curvy>Tex-Rectilinear',
    'FFA_Obj-Curvy>Obj-Rectilinear']
sig_types_ppa = ['PPA_Tex-Curvy<Tex-Rectilinear',
    'PPA_Obj-Curvy<Obj-Rectilinear']

# Plot parameters
n_sub = 8
x_coord = np.arange(len(image_types))
alpha = 0.2
marker = 'o'
s = 500
s_mean = 750
color = 'k'

# Create the plot figure
fig, axs = plt.subplots(1, 2, sharex=False, sharey=True,
    figsize=(17.5, 5))
axs = np.reshape(axs, -1)

# Loop across ROIs
for r, roi in enumerate(['FFA', 'PPA']):

    # Loop across image types
    for i, itype in enumerate(image_types):

        # Univariate response scores
        x = np.repeat(i, n_sub)
        y = vertex_mean_resp[roi][itype]
        axs[r].scatter(x, y, s=s, color=color, alpha=alpha, edgecolors='none')
        axs[r].scatter(x[0], np.mean(y), s=s_mean, color=color,
            edgecolors='none')

        # Confidence intervals
        ci = np.zeros(2)
        ci[0] = np.mean(y) - ci_vertex_mean_resp[roi][itype][0]
        ci[1] = ci_vertex_mean_resp[roi][itype][1] - np.mean(y)
        axs[r].errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)),
            fmt="none", ecolor=color, elinewidth=5, capsize=0)

    # Significance (FFA)
    if roi == 'FFA':
        for sig_type in sig_types_ffa:
            if pval_diff[sig_type] < 0.0001:
                sig = '***'
            elif pval_diff[sig_type] < 0.001:
                sig = '**'
            elif pval_diff[sig_type] < 0.01:
                sig = '*'
            else:
                sig = ''
            if sig_type == 'FFA_Tex-Curvy>Tex-Rectilinear' and sig != '':
                x = [0, 1]
                y = [np.mean(vertex_mean_resp[roi]['Curvature-Tex-Curvy']),
                    np.mean(vertex_mean_resp[roi]['Curvature-Tex-Rectilinear'])]
            if sig_type == 'FFA_Obj-Curvy>Obj-Rectilinear' and sig != '':
                x = [2, 3]
                y = [np.mean(vertex_mean_resp[roi]['Curvature-Obj-Curvy']),
                    np.mean(vertex_mean_resp[roi]['Curvature-Obj-Rectilinear'])]
            axs[r].plot(x, y, color='k', linewidth=2, alpha=0.5)
            axs[r].text(np.mean(x), np.mean(y), sig, fontsize=fontsize,
                ha='center', va='bottom', rotation_mode='anchor',
                rotation=np.degrees(np.arctan2(y[1]-y[0], x[1]-x[0])),
                transform_rotates_text=True)

    # Significance (PPA)
    if roi == 'PPA':
        for sig_type in sig_types_ppa:
            if pval_diff[sig_type] < 0.0001:
                sig = '***'
            elif pval_diff[sig_type] < 0.001:
                sig = '**'
            elif pval_diff[sig_type] < 0.01:
                sig = '*'
            else:
                sig = ''
            if sig_type == 'PPA_Tex-Curvy<Tex-Rectilinear' and sig != '':
                x = [0, 1]
                y = [np.mean(vertex_mean_resp[roi]['Curvature-Tex-Curvy']),
                    np.mean(vertex_mean_resp[roi]['Curvature-Tex-Rectilinear'])]
            if sig_type == 'PPA_Obj-Curvy<Obj-Rectilinear' and sig != '':
                x = [2, 3]
                y = [np.mean(vertex_mean_resp[roi]['Curvature-Obj-Curvy']),
                    np.mean(vertex_mean_resp[roi]['Curvature-Obj-Rectilinear'])]
            axs[r].plot(x, y, color='k', linewidth=2, alpha=0.5)
            axs[r].text(np.mean(x), np.mean(y), sig, fontsize=fontsize,
                ha='center', va='bottom', rotation_mode='anchor',
                rotation=np.degrees(np.arctan2(y[1]-y[0], x[1]-x[0])),
                transform_rotates_text=True)

    # Title
    axs[r].set_title(roi, fontsize=fontsize)

    # x-axis parameters
    xticks = x_coord
    axs[r].set_xticks(ticks=xticks, labels=labels, rotation=0, ha='center')
    axs[r].set_xlim(left=-0.5, right=3.5)

    # y-axis parameters
    if r == 0:
        yticks = [-0.6, -0.3, 0, 0.3]
        ylabels = ['-0.6', '-0.3', '0', '0.3']
        axs[r].set_yticks(ticks=yticks, labels=ylabels)
        ylabel = 'Univariate response\n($z$-scored)'
        axs[r].set_ylabel(ylabel, fontsize=fontsize)
        axs[r].set_ylim(bottom=-.6, top=.15)

# Save the figure
file_name = os.path.join(save_dir, 'curvature.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot FFA's sensitivity to face shapes across texture variation
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'stats', args.encoding_model,
    'ffa_face_texture.npy')
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

# Create the figure
fig = plt.figure(figsize=(17.5, 5))

# Loop across image types
for i, itype in enumerate(image_types_sorted):

    # Plot the baseline object score
    if itype != 'Objects':
        y = np.mean(vertex_mean_resp['Objects'])
        plt.plot([i-0.15, i+0.15], [y, y], '--', color='gray', linewidth=2,
            zorder=1)

    # Univariate response scores
    x = np.repeat(i, n_sub)
    y = vertex_mean_resp[itype]
    plt.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors='none',
        zorder=2)
    plt.scatter(x[0], np.mean(y), s=s_mean, color=color, edgecolors='none',
        zorder=3)

    # Confidence intervals
    ci = np.zeros(2)
    ci[0] = np.mean(y) - ci_vertex_mean_resp[itype][0]
    ci[1] = ci_vertex_mean_resp[itype][1] - np.mean(y)
    plt.errorbar(x[0], np.mean(y), yerr=np.reshape(ci, (-1,1)), fmt="none",
        ecolor=color, elinewidth=5, capsize=0)

    # Significance
    if itype != 'Objects':
        if pval_diff[f'{itype}>Objects'] < 0.0001:
            sig = '***'
        elif pval_diff[f'{itype}>Objects'] < 0.001:
            sig = '**'
        elif pval_diff[f'{itype}>Objects'] < 0.01:
            sig = '*'
        else:
            sig = ''
        plt.text(x[0], max(y)+0.1, sig, fontsize=fontsize, ha='center',
            va='center')

# x-axis parameters
xticks = x_coord
plt.xticks(ticks=xticks, labels=labels_sorted, rotation=-30, ha='center')
plt.xlim(left=-0.5, right=7.5)

# y-axis parameters
yticks = [-1, -0.5, 0, 0.5, 1]
ylabels = ['-1', '-0.5', '0', '0.5', '1']
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Univariate response\n($z$-scored)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=-.4, top=1.1)

# Save the figure
file_name = os.path.join(save_dir, 'ffa_face_texture_variation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot PPA's sensitivity to spatial layout information
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'insilico_capture_of_neural_signatures',
    'fmri', 'ffa_ppa_effects', 'stats', args.encoding_model,
    'ppa_spatial_layout.npy')
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

# Create the figure
fig = plt.figure(figsize=(10,5))

# Loop across image types
for i, itype in enumerate(image_types_sorted):

    # Plot the baseline empty room score
    if itype != 'Scenes-EmptyRooms':
        y = np.mean(vertex_mean_resp['Scenes-EmptyRooms'])
        plt.plot([i-0.15, i+0.15], [y, y], '--', color='gray', linewidth=2,
            zorder=1)

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

    # Significance
    if itype != 'Scenes-EmptyRooms':
        if pval_diff[f'Scenes-EmptyRooms>{itype}'] < 0.0001:
            sig = '***'
        elif pval_diff[f'Scenes-EmptyRooms>{itype}'] < 0.001:
            sig = '**'
        elif pval_diff[f'Scenes-EmptyRooms>{itype}'] < 0.01:
            sig = '*'
        else:
            sig = ''
        plt.text(x[0], min(y)-0.1, sig, fontsize=fontsize, ha='center',
            va='center')

# x-axis parameters
xticks = x_coord
plt.xticks(ticks=xticks, labels=labels_sorted, rotation=0, ha='center')
plt.xlim(left=-0.5, right=3.5)

# y-axis parameters
yticks = [0, 0.5]
ylabels = ['0', '0.5']
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Univariate response\n($z$-scored)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=-.275, top=.75)

# Save the figure
file_name = os.path.join(save_dir, 'ppa_spatial_layout.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()