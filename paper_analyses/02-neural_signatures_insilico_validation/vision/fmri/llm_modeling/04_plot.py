"""Plot the searchlight RSA scores between in silico fMRI responses and LLM
embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
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
import cortex
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=0, type=float) # 20
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'llm_modeling', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the RSA results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri', 'llm_modeling',
    'stats', args.encoding_model, 'stats.npy')

results = np.load(results_dir, allow_pickle=True).item()

lh_rsa = np.array(results['lh_rsa'])
rh_rsa = np.array(results['rh_rsa'])
metadata = results['metadata']


# =============================================================================
# NCSNR and encoding accuracy vertex selection
# =============================================================================
for s in range(len(args.subjects)):
    for hemi in ['lh', 'rh']:

        ncsnr = results['metadata'][s]['fmri'][hemi+'_ncsnr']
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = results['metadata'][s]['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)

        if hemi == 'lh':
            lh_rsa[s,idx_nan] = np.nan
        else:
            rh_rsa[s,idx_nan] = np.nan


# =============================================================================
# Plot parameters
# =============================================================================
# Plot parameters for colorbar
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'


# =============================================================================
# Plot the RSA results
# =============================================================================
# Average the results across subjects, and append them across left and right
# hemishperes
data = np.append(np.nanmean(lh_rsa, 0), np.nanmean(rh_rsa, 0))

# Create the flat brain surface
vertex_data = cortex.Vertex(
    data,
    subject=subject,
    cmap='afmhot',
    vmin=0,
    vmax=0.5,
    with_colorbar=True
    )

# Plot the flat brain surface
fig = cortex.quickshow(
    vertex_data,
    height=2000, # Increase resolution of map and ROI contours
    with_curvature=True,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    linewidth=3,
    linecolor=(1, 1, 1),
    with_labels=True,
    labelsize=25,
    curvature_brightness=0.4,
    with_colorbar=True
    )

# Save the figure
file_name = os.path.join(save_dir, 'rsa.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the RSA results (other colormap)
# =============================================================================
# Average the results across subjects, and append them across left and right
# hemishperes
data = np.append(np.nanmean(lh_rsa, 0), np.nanmean(rh_rsa, 0))

# Create the flat brain surface
vertex_data = cortex.Vertex(
    data,
    subject=subject,
    cmap='Reds', # !!! 'afmhot', 'RdBu_r'
    vmin=0,
    vmax=0.5,
    with_colorbar=True
    )

# Plot the flat brain surface
fig = cortex.quickshow(
    vertex_data,
    height=2000, # Increase resolution of map and ROI contours
    with_curvature=True,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    linewidth=3,
    linecolor=(1, 1, 1),
    with_labels=True,
    labelsize=25,
    curvature_brightness=0.4,
    with_colorbar=True
    )

# Save the figure
file_name = os.path.join(save_dir, 'rsa.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()