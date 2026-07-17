"""Plot the vertex-mean responses of high-level visual cortex ROIs for images
of different categories.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list
    List of subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
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

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'hvc_selectivity_nsd_floc_stimuli', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
lh_tval = []
rh_tval = []

for sub in args.subjects:

    # Load the results
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'hvc_selectivity_nsd_floc_stimuli', 't_values', args.encoding_model,
        f'results_sub-{sub:02d}.npy')
    data = np.load(data_dir, allow_pickle=True).item()
    lh_tval_sub = data['lh_tval']
    rh_tval_sub = data['rh_tval']

    # NCSNR vertex selection
    idx_nan_lh = data['metadata']['fmri']['lh_ncsnr'] >= \
        args.ncsnr_threshold
    idx_nan_rh = data['metadata']['fmri']['rh_ncsnr'] >= \
        args.ncsnr_threshold
    for key in lh_tval_sub.keys():
        lh_tval_sub[key][~idx_nan_lh] = np.nan
        rh_tval_sub[key][~idx_nan_rh] = np.nan

    # Store the results
    lh_tval.append(lh_tval_sub)
    rh_tval.append(rh_tval_sub)
    del data, lh_tval_sub, rh_tval_sub


# =============================================================================
# Plot parameters
# =============================================================================
# Plot parameters for colorbar
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot the t-values of each subject and category
# =============================================================================
# Loop across subjects and categories
for s, sub in enumerate(tqdm(args.subjects)):
    for cat in ['bodies', 'faces', 'objects', 'places']:

        # Select the ROIs to plot
        if cat == 'faces':
            roi_list = ['OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces']
        if cat == 'bodies':
            roi_list = ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies']
        elif cat == 'places':
            roi_list = ['OPA', 'PPA', 'RSC']
        elif cat == 'objects':
            roi_list = ['EBA', 'FFA-1', 'FFA-2', 'RSC', 'PPA', 'OPA']

        # Append the results across left and right hemishperes
        data = np.append(lh_tval[s][cat], rh_tval[s][cat])

        # Create the flat brain surface
        subject = 'fsaverage_nsd_sub-0' + str(sub)
        vertex_data = cortex.Vertex(
            data,
            subject=subject,
            cmap='inferno',
            vmin=0,
            vmax=30,
            with_colorbar=True
            )

        # Plot the flat brain surface
        fig = cortex.quickshow(
            vertex_data,
            height=2000, # Increase resolution of map and ROI contours
            with_curvature=True,
            with_rois=True,
            roi_list=roi_list,
            linewidth=3,
            linecolor=(1, 1, 1),
            with_labels=True,
            labelsize=35,
            curvature_brightness=0.4,
            with_colorbar=True
            )

        # Save the figure
        file_name = os.path.join(save_dir, f'tval-{cat}_sub-{sub}.svg')
        fig.savefig(file_name, bbox_inches='tight', transparent=True,
            format='svg')
        plt.close()