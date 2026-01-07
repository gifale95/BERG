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
    integers from 1 8.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
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
parser.add_argument('--encoding_threshold', default=20, type=float) # 20
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'hvc_selectivity', 'plots', args.encoding_model)
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
        'hvc_selectivity', 't_values', args.encoding_model,
        f'results_sub-{sub:02d}.npy')
    data = np.load(data_dir, allow_pickle=True).item()
    lh_tval_sub = data['lh_tval']
    rh_tval_sub = data['rh_tval']

    # NCSNR and noise ceiling vertex selection
    idx_ncsnr_lh = data['metadata']['fmri']['lh_ncsnr'] >= args.ncsnr_threshold
    idx_ncsnr_rh = data['metadata']['fmri']['rh_ncsnr'] >= args.ncsnr_threshold
    idx_encoding_lh = data['metadata']['encoding_models']\
        ['lh_explained_variance_nsdcore'] >= args.encoding_threshold
    idx_encoding_rh = data['metadata']['encoding_models']\
        ['rh_explained_variance_nsdcore'] >= args.encoding_threshold
    idx_nan_lh = ~np.logical_and(idx_ncsnr_lh, idx_encoding_lh)
    idx_nan_rh = ~np.logical_and(idx_ncsnr_rh, idx_encoding_rh)
    for key in lh_tval_sub.keys():
        lh_tval_sub[key][idx_nan_lh] = np.nan
        rh_tval_sub[key][idx_nan_rh] = np.nan

    # Threshold based on significance and store the results
    for key in lh_tval_sub.keys():
        lh_tval_sub[key][~data['lh_sig'][key]] = np.nan
        rh_tval_sub[key][~data['rh_sig'][key]] = np.nan
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
    for cat in lh_tval[s].keys():

        # Append the results across left and right hemishperes
        data = np.append(lh_tval[s][cat], rh_tval[s][cat])

        # Create the flat brain surface
        subject = 'fsaverage_nsd_sub-0' + str(sub)
        vertex_data = cortex.Vertex(
            data,
            subject=subject,
            cmap='viridis',
            vmin=0,
            vmax=10,
            with_colorbar=True
            )

        # Plot the flat brain surface
        fig = cortex.quickshow(
            vertex_data,
            height=2000, # Increase resolution of map and ROI contours
            with_curvature=True,
            with_rois=True,
            roi_list=['Early', 'FFA-1', 'FFA-2', 'OFA', 'FBA-1', 'FBA-2',
                'EBA', 'PPA', 'OPA', 'RSC'],
            linewidth=3,
            linecolor=(1, 1, 1),
            with_labels=True,
            labelsize=20,
            curvature_brightness=0.5,
            with_colorbar=True
            )

        # Save the figure
        file_name = os.path.join(save_dir, f'tval-{cat}_sub-{sub}.svg')
        fig.savefig(file_name, bbox_inches='tight', transparent=True,
            format='svg')
        plt.close()