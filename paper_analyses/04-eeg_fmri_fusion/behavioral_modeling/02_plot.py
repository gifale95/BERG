"""Plot the searchlight RSA scores between t-fMRI responses and behavioral
embeddings.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
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
from berg import BERG
from tqdm import tqdm
import cortex
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = []
rh_rsa = []

for sub in args.fmri_subjects:
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'behavioral_modeling', 'rsa', f'rsa_sub-{sub:02d}_{hemi}.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        # NCSNR and noise ceiling vertex selection
        ncsnr = results['metadata']['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr > args.ncsnr_threshold
        encoding = results['metadata']['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding > args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
        rsa = results['rsa']
        rsa[idx_nan] = np.nan

        # Store the RSA results
        if hemi == 'lh':
            lh_rsa.append(rsa)
        elif hemi == 'rh':
            rh_rsa.append(rsa)
        del rsa

lh_rsa = np.array(lh_rsa)
rh_rsa = np.array(rh_rsa)


# =============================================================================
# Load the EEG time points
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)

times = metadata_eeg['eeg']['times']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 40
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'


# =============================================================================
# Plot the RSA results
# =============================================================================
# Loop over EEG time points
for t, time in enumerate(tqdm(times)):

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(lh_rsa[:,:,t], 0),
        np.nanmean(rh_rsa[:,:,t], 0))
    
    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject,
        cmap='afmhot',
        vmin=0,
        vmax=0.3,
        with_colorbar=True)

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        #height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=15,
        curvature_brightness=0.5,
        with_colorbar=True
        )

    # Add title
    title = f'Time (s): {np.round(time, 3)}'
    plt.title(title, fontsize=fontsize)

    # Save the plot
    plot_file = os.path.join(save_dir, f'rsa_time-{t:03d}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()