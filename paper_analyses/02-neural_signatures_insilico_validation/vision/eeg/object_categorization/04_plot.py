"""Plot the of object exemplar and animacy decoding accuracy of in silico EEG
responses.

Parameters
----------
subjects : list
    The subject identifiers for the EEG encoding models. Since the used
    encoding models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
berg_dir : str
    Directory of the BERG.

"""

import argparse
from operator import sub
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directories
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_categorization', 'plots')
os.makedirs(save_dir, exist_ok=True)

save_dir_mds = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_categorization', 'plots', 'mds')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'object_categorization', 'stats', 'stats_'+'channels-'+
    '-'.join(args.channels)+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

decoding_exemplars = results['decoding_exemplars']
decoding_objects = results['decoding_objects']
decoding_animacy = results['decoding_animacy']
ci_exemplars = results['ci_exemplars']
ci_objects = results['ci_objects']
ci_animacy = results['ci_animacy']
sig_exemplars = results['sig_exemplars']
sig_objects = results['sig_objects']
sig_animacy = results['sig_animacy']
ci_peak_latency_exemplars = results['ci_peak_latency_exemplars']
ci_peak_latency_objects = results['ci_peak_latency_objects']
ci_peak_latency_animacy = results['ci_peak_latency_animacy']
times = results['times']
eeg_mds = results['eeg_mds']
eeg_mds_single_sub = results['eeg_mds_single_sub']


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
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(166/255, 77/255, 121/255), (100/255, 149/255, 237/255),
    (105/255, 105/255, 105/255)] # (169/255, 169/255, 169/255)]


# =============================================================================
# Plot the decoding accuracy results
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
    figsize=(13, 7))
axs = np.reshape(axs, (-1)) # type: ignore

# Plot the chance and stimulus onset dashed lines
axs[0].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
    linewidth=3, alpha=.5, label='_nolegend_')

# Plot the decoding subject-average results
# Exemplar decoding
label = 'Exemplar'
peak = times[np.argmax(np.mean(decoding_exemplars, 0))]
max_dec = max(np.mean(decoding_exemplars, 0))
axs[0].plot(times, np.mean(decoding_exemplars, 0), color=colors[0],
    linewidth=3, label=label)
axs[0].fill_between(times, ci_exemplars[0], ci_exemplars[1], color=colors[0],
    alpha=.2)
axs[0].scatter(peak, max_dec, color=colors[0], s=200, marker='o',
    edgecolors='k', linewidths=1, zorder=3)
ci_low = peak - ci_peak_latency_exemplars[0]
ci_up = ci_peak_latency_exemplars[1] - peak
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
axs[0].errorbar(peak, max_dec, xerr=conf_int, fmt="none", ecolor=colors[0],
    elinewidth=1, capsize=3)

# Animacy decoding
label = 'Animacy'
peak = times[np.argmax(np.mean(decoding_animacy, 0))]
max_dec = max(np.mean(decoding_animacy, 0))
axs[0].scatter(peak, max_dec, color=colors[1], s=200, marker='o',
    edgecolors='k', linewidths=1, zorder=3)
axs[0].plot(times, np.mean(decoding_animacy, 0), color=colors[1], linewidth=3,
    label=label)
axs[0].fill_between(times, ci_animacy[0], ci_animacy[1], color=colors[1],
    alpha=.2)
ci_low = peak - ci_peak_latency_animacy[0]
ci_up = ci_peak_latency_animacy[1] - peak
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
axs[0].errorbar(peak, max_dec, xerr=conf_int, fmt="none", ecolor=colors[1],
    elinewidth=1, capsize=3)

# Object decoding
label = 'Object'
peak = times[np.argmax(np.mean(decoding_objects, 0))]
max_dec = max(np.mean(decoding_objects, 0))
axs[0].scatter(peak, max_dec, color=colors[2], s=200, marker='o',
    edgecolors='k', linewidths=1, zorder=3)
axs[0].plot(times, np.mean(decoding_objects, 0), color=colors[2], linewidth=3,
    label=label)
axs[0].fill_between(times, ci_objects[0], ci_objects[1], color=colors[2],
    alpha=.2)
ci_low = peak - ci_peak_latency_objects[0]
ci_up = ci_peak_latency_objects[1] - peak
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
axs[0].errorbar(peak, max_dec, xerr=conf_int, fmt="none", ecolor=colors[2],
    elinewidth=1, capsize=3)

# x-axis parameters
axs[0].set_xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels) # type: ignore
axs[0].set_xlim(left=min(times), right=max(times))

# y-axis parameters
axs[0].set_ylabel('Decoding accuracy (%)', fontsize=fontsize)
yticks = [50, 60, 70, 80, 90, 100]
ylabels = [50, 60, 70, 80, 90, 100]
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
axs[0].set_ylim(bottom=45, top=100)

# Legend
axs[0].legend(ncol=1, fontsize=fontsize, loc=1, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'decoding_accuray_channels-'+
    '-'.join(args.channels)+'.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the EEG responses in MDS space, color coded based on animacy
# =============================================================================
# Plot parameters
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
colors = [(100/255, 149/255, 237/255), (169/255, 169/255, 169/255)]

# Loop across time points
for t in tqdm(range(len(times))):

    # Create the figure
    fig, ax = plt.subplots(figsize=(13, 13))

    # Plot the animate images
    plt.scatter(eeg_mds[:eeg_mds.shape[0]//2,0,t],
        eeg_mds[:eeg_mds.shape[0]//2,1,t],	s=500, color=colors[0],
        linewidths=0, alpha=.9)

    # Plot the inanimate images
    plt.scatter(eeg_mds[eeg_mds.shape[0]//2:,0,t],
        eeg_mds[eeg_mds.shape[0]//2:,1,t],	s=500, color=colors[1],
        linewidths=0, alpha=.9)

    # x-axis
    plt.xticks([])
#    plt.xlim(left=min(eeg_mds[:,0].flatten()),
#        right=max(eeg_mds[:,0].flatten()))

    # y-axis
    plt.yticks([])
#    plt.ylim(bottom=min(eeg_mds[:,1].flatten()),
#        top=max(eeg_mds[:,1].flatten()))
    # Title
    title = 'Time: ' + str(np.round((times[t]*1000))) + ' ms'
    plt.title(title, fontsize=fontsize)

    # Save the figure
    file_name = os.path.join(save_dir_mds, 'mds_animacy_time-'+format(t, '03')+
        '.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    file_name = os.path.join(save_dir_mds, 'mds_animacy_time-'+format(t, '03')+
        '.png')
    fig.savefig(file_name, bbox_inches='tight', transparent=False, format='png')

    # Close the figure
    plt.close(fig)


# =============================================================================
# Plot the EEG responses in MDS space, color coded based on animacy
# (single subjects)
# =============================================================================
# Plot parameters
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
colors = [(100/255, 149/255, 237/255), (169/255, 169/255, 169/255)]

# Loop acros subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Select the MDS results from the subject of interest
    eeg_mds_sub = eeg_mds_single_sub[s]

    # Loop across time points
    for t in tqdm(range(len(times))):

        # Create the figure
        fig, ax = plt.subplots(figsize=(13, 13))

        # Plot the animate images
        plt.scatter(eeg_mds_sub[:eeg_mds_sub.shape[0]//2,0,t],
            eeg_mds_sub[:eeg_mds_sub.shape[0]//2,1,t],	s=500, color=colors[0],
            linewidths=0, alpha=.9)

        # Plot the inanimate images
        plt.scatter(eeg_mds_sub[eeg_mds_sub.shape[0]//2:,0,t],
            eeg_mds_sub[eeg_mds_sub.shape[0]//2:,1,t],	s=500, color=colors[1],
            linewidths=0, alpha=.9)

        # x-axis
        plt.xticks([])
    #    plt.xlim(left=min(eeg_mds[:,0].flatten()),
    #        right=max(eeg_mds[:,0].flatten()))

        # y-axis
        plt.yticks([])
    #    plt.ylim(bottom=min(eeg_mds[:,1].flatten()),
    #        top=max(eeg_mds[:,1].flatten()))
        # Title
        title = 'Time: ' + str(np.round((times[t]*1000))) + ' ms'
        plt.title(title, fontsize=fontsize)

        # Save the figure
        file_name = os.path.join(save_dir_mds, 'mds_animacy_sub-'+
            format(sub, '02')+'_time-'+format(t, '03')+'.svg')
        fig.savefig(file_name, bbox_inches='tight', transparent=True,
            format='svg')
        file_name = os.path.join(save_dir_mds, 'mds_animacy_sub-'+
            format(sub, '02')+'_time-'+format(t, '03')+'.png')
        fig.savefig(file_name, bbox_inches='tight', transparent=False,
            format='png')

        # Close the figure
        plt.close(fig)


# =============================================================================
# Plot the MDS results with images at peak decoding time points
# =============================================================================
# Plot parameters
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False

# Read all images into memory
images = []
image_path = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'object_categorization', 'stimuli')
animate_imgs = os.listdir(os.path.join(image_path, 'animate'))
animate_imgs.sort()
for img_file in tqdm(animate_imgs):
    img = Image.open(os.path.join(image_path, 'animate', img_file))
    img = np.asarray(img) / 255
    images.append(img)
inanimate_imgs = os.listdir(os.path.join(image_path, 'inanimate'))
inanimate_imgs.sort()
for img_file in tqdm(inanimate_imgs):
    img = Image.open(os.path.join(image_path, 'inanimate', img_file))
    img = np.asarray(img) / 255
    images.append(img)

# Get the peak decoding time indices
peak_indices = {
    'exemplars': np.argmax(np.mean(decoding_exemplars, 0)),
    'objects': np.argmax(np.mean(decoding_objects, 0)),
    'animacy': np.argmax(np.mean(decoding_animacy, 0))
}

# Loop across decoding types
for key, val in tqdm(peak_indices.items()):

    # Create the figure
    fig, ax = plt.subplots(figsize=(13, 13))

    # Loop across images
    for i in range(len(images)):

        # Get the image
        img = images[i]

        # Plot the image
        imagebox = OffsetImage(img, zoom=0.15) # Adjust zoom as needed
        ab = AnnotationBbox(
            imagebox,
            (eeg_mds[i,0,val], eeg_mds[i,1,val]),
            frameon=True,
            pad=0,
            bboxprops=dict(edgecolor=None, linewidth=0))
        ax.add_artist(ab)

    # x-axis
    plt.xticks([])
    plt.xlim(left=min(eeg_mds[:,0,val]), right=max(eeg_mds[:,0,val]))

    # y-axis
    plt.yticks([])
    plt.ylim(bottom=min(eeg_mds[:,1,val]), top=max(eeg_mds[:,1,val]))

    # Title
    title = f'EEG MDS at peak {key} decoding time ({int((times[val]*1000))} ms)'
    plt.title(title, fontsize=fontsize)

    # Save the figure
    file_name = os.path.join(save_dir, 'mds_images_'+key+
        '_peak_decoding_channels-'+'-'.join(args.channels)+'.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')

    # Close the figure
    plt.close(fig)