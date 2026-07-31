"""Plot t-fMRI response animacy MDS results.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the MDS results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'animacy_mds', 'mds',
    'mds.npy')

results = np.load(data_dir, allow_pickle=True).item()

msd_sub_single = results['msd_sub_single']
msd_sub_all = results['msd_sub_all']
metadata_fmri = results['metadata_fmri']
times = results['times']
del results


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 25
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(100/255, 149/255, 237/255), (169/255, 169/255, 169/255)]


# =============================================================================
# Animacy indices
# =============================================================================
idx_an = np.array([1, 9, 11, 23, 32, 33, 36, 38, 45, 51, 52, 57, 62, 64, 68,
    69, 71, 75, 85, 86, 87, 88, 96, 103, 105, 109, 110, 116, 117, 126, 128,
    132, 135, 136, 141, 143, 149, 150, 151, 160, 175, 182, 189])

idx_in = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 39, 40, 41, 42, 43,
    44, 46, 47, 48, 49, 50, 53, 54, 55, 56, 58, 59, 60, 61, 63, 65, 66, 67, 70,
    72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94, 95,
    97, 98, 99, 100, 101, 102, 104, 106, 107, 108, 111, 112, 113, 114, 115,
    118, 119, 120, 121, 122, 123, 124, 125, 127, 129, 130, 131, 133, 134, 137,
    138, 139, 140, 142, 144, 145, 146, 147, 148, 152, 153, 154, 155, 156, 157,
    158, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199])


# =============================================================================
# Plot the t-fMRI responses in MDS space, color coded based on animacy
# (single subjects)
# =============================================================================
# ROI list
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'PPA', 'EBA', 'early', 'intermediate',
    'ventral', 'lateral', 'parietal']

# Loop acros fMRI, subjects, ROIs, and EEG time points
for fs, fsub in enumerate(tqdm(args.fmri_subjects)):
    for r, roi in enumerate(rois):

        # Create the plots save directory
        save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'animacy_mds', 'plots', 'subjects-single',
            f'fmri_sub-0{fsub}', roi)
        os.makedirs(save_dir, exist_ok=True)

        for t in range(len(times)):

            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot the animate images
            plt.scatter(msd_sub_single[f's{fsub}_{roi}'][idx_an,0,t],
                msd_sub_single[f's{fsub}_{roi}'][idx_an,1,t], s=250,
                color=colors[0], linewidths=0, alpha=.9)

            # Plot the inanimate images
            plt.scatter(msd_sub_single[f's{fsub}_{roi}'][idx_in,0,t],
                msd_sub_single[f's{fsub}_{roi}'][idx_in,1,t], s=250,
                color=colors[1], linewidths=0, alpha=.9)

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
            # file_name = os.path.join(save_dir, f'mds_time-{t:03}.svg')
            # fig.savefig(file_name, bbox_inches='tight', transparent=True,
            #     format='svg')
            file_name = os.path.join(save_dir, f'mds_time-{t:03}.png')
            fig.savefig(file_name, bbox_inches='tight', transparent=False,
                format='png')
            plt.close(fig)


# =============================================================================
# Plot the t-fMRI responses in MDS space, color coded based on animacy
# (all subjects)
# =============================================================================
# ROI list
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'PPA', 'EBA', 'early', 'intermediate',
    'ventral', 'lateral', 'parietal']

# Loop acros ROIs and EEG time points
for r, roi in enumerate(tqdm(rois)):

    # Create the plots save directory
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'animacy_mds', 'plots', 'subjects-all', roi)
    os.makedirs(save_dir, exist_ok=True)

    for t in range(len(times)):

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the animate images
        plt.scatter(msd_sub_all[roi][idx_an,0,t], msd_sub_all[roi][idx_an,1,t],
            s=250, color=colors[0], linewidths=0, alpha=.9)

        # Plot the inanimate images
        plt.scatter(msd_sub_all[roi][idx_in,0,t], msd_sub_all[roi][idx_in,1,t],
            s=250, color=colors[1], linewidths=0, alpha=.9)

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
        # file_name = os.path.join(save_dir, f'mds_time-{t:03}.svg')
        # fig.savefig(file_name, bbox_inches='tight', transparent=True,
        #     format='svg')
        file_name = os.path.join(save_dir, f'mds_time-{t:03}.png')
        fig.savefig(file_name, bbox_inches='tight', transparent=False,
            format='png')
        plt.close(fig)