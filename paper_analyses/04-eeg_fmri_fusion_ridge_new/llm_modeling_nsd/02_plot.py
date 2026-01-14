"""Plot the searchlight RSA scores between t-fMRI responses and LLM embeddings.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemisphere : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
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
parser.add_argument('--fmri_subjects', default=[1, 2, 5, 7], type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new', 'llm_modeling_nsd',
    'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA results
# =============================================================================
# lh_rsa = []
# rh_rsa = []

# for fsub in args.fmri_subjects:
#     for hemi in args.hemispheres:

#         results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
#             'llm_modeling_nsd', 'rsa', f'rsa_sub-{fsub:02d}_hemi-{hemi}.npy')
#         results = np.load(results_dir, allow_pickle=True).item()

#         # NCSNR and encoding accuracy vertex selection
#         ncsnr = results['metadata']['fmri'][hemi+'_ncsnr']
#         idx_ncsnr = ncsnr >= args.ncsnr_threshold
#         encoding = results['metadata']['encoding_models']\
#             [hemi+'_explained_variance_nsdcore']
#         idx_encoding = encoding >= args.encoding_threshold
#         idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
#         rsa = results['rsa']
#         rsa[idx_nan] = np.nan
#         del results

#         # Store the RSA results
#         if hemi == 'lh':
#             lh_rsa.append(rsa)
#         elif hemi == 'rh':
#             rh_rsa.append(rsa)
#         del rsa

# lh_rsa = np.array(lh_rsa)
# rh_rsa = np.array(rh_rsa)

# # !!! DELETE
# rh_rsa = np.empty(lh_rsa.shape, dtype=np.float32) * np.nan
# # !!! DELETE


# =============================================================================
# Load the RSA results # !!! DELETE # !!!
# =============================================================================
lh_rsa = {}
lh_rsa['rsa_insilicoeeg_avg_tfmri_avg'] = []
lh_rsa['rsa_insilicoeeg_sing_tfmri_avg'] = []
lh_rsa['rsa_insilicoeeg_sing_tfmri_sing'] = []
rh_rsa = {}
rh_rsa['rsa_insilicoeeg_avg_tfmri_avg'] = []
rh_rsa['rsa_insilicoeeg_sing_tfmri_avg'] = []
rh_rsa['rsa_insilicoeeg_sing_tfmri_sing'] = []

for fsub in args.fmri_subjects:
    for hemi in args.hemispheres:

        results_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_ridge_new',
            'llm_modeling_nsd', 'rsa', f'rsa_sub-{fsub:02d}_hemi-{hemi}.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        # NCSNR and encoding accuracy vertex selection
        ncsnr = results['metadata']['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = results['metadata']['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

        rsa_insilicoeeg_avg_tfmri_avg = results['rsa_insilicoeeg_avg_tfmri_avg']
        rsa_insilicoeeg_avg_tfmri_avg[idx_nan] = np.nan
        rsa_insilicoeeg_sing_tfmri_avg = results['rsa_insilicoeeg_sing_tfmri_avg']
        rsa_insilicoeeg_sing_tfmri_avg[idx_nan] = np.nan
        rsa_insilicoeeg_sing_tfmri_sing = results['rsa_insilicoeeg_sing_tfmri_sing']
        rsa_insilicoeeg_sing_tfmri_sing[idx_nan] = np.nan
        del results

        # Store the RSA results
        if hemi == 'lh':
            lh_rsa['rsa_insilicoeeg_avg_tfmri_avg'].append(rsa_insilicoeeg_avg_tfmri_avg)
            lh_rsa['rsa_insilicoeeg_sing_tfmri_avg'].append(rsa_insilicoeeg_sing_tfmri_avg)
            lh_rsa['rsa_insilicoeeg_sing_tfmri_sing'].append(rsa_insilicoeeg_sing_tfmri_sing)
        elif hemi == 'rh':
            rh_rsa['rsa_insilicoeeg_avg_tfmri_avg'].append(rsa_insilicoeeg_avg_tfmri_avg)
            rh_rsa['rsa_insilicoeeg_sing_tfmri_avg'].append(rsa_insilicoeeg_sing_tfmri_avg)
            rh_rsa['rsa_insilicoeeg_sing_tfmri_sing'].append(rsa_insilicoeeg_sing_tfmri_sing)
        del rsa_insilicoeeg_avg_tfmri_avg, rsa_insilicoeeg_sing_tfmri_avg, rsa_insilicoeeg_sing_tfmri_sing

lh_rsa = {key: np.array(value) for key, value in lh_rsa.items()}
rh_rsa = {key: np.array(value) for key, value in rh_rsa.items()}


# =============================================================================
# Load the EEG time points
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)

times = metadata_eeg['eeg']['times']
times = times[np.arange(20, 80)] # !!! CHANGE


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
for key in lh_rsa.keys():

    # Loop over EEG time points
    for t, time in enumerate(tqdm(times)):

        # Average the results across subjects, and append them across left and
        # right hemishperes
        data = np.append(np.nanmean(lh_rsa[key][:,:,t], 0),
            np.nanmean(rh_rsa[key][:,:,t], 0))
        
        # Create the flat brain surface
        vertex_data = cortex.Vertex(
            data,
            subject,
            cmap='afmhot',
            vmin=0,
            vmax=0.5,
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
            curvature_brightness=0.4,
            with_colorbar=True
            )

        # Add title
        title = f'Time (s): {np.round(time, 3)}'
        plt.title(title, fontsize=fontsize)
        
        # Save the plot
        plot_file = os.path.join(save_dir, f'{key}_time-{t:03d}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
        plt.close()