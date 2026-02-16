"""Plot the prediction accuracy of encoding models trained on the BOLD Moments
Dataset (BMD).

Parameters
----------
subjects : list
    List with all used BMD subjects.
model_id : str
    Unique identifier of the model to load.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import nibabel as nib
from nilearn import plotting
import ast
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=ast.literal_eval, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--model_id', type=str, default='fmri-bmd-<your_model_name>')
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the encoding models' encoding accuracy
# =============================================================================
sub_mask = []
sub_mask_affine = []
rois = []
correlation = []
r2 = []
explained_variance = []

model_name = args.model_id.split('-')[-1]
metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-bmd', 'model-'+model_name, 'metadata')

for sub in tqdm(args.subjects):

    file_name = 'metadata_sub-' + format(sub, '02') + '.npy'
    metadata = np.load(os.path.join(metadata_dir, file_name),
        allow_pickle=True).item()

    sub_mask.append(metadata['fmri']['sub_mask'])
    sub_mask_affine.append(metadata['fmri']['sub_mask_affine'])
    rois.append(metadata['fmri']['rois'])
    correlation.append(metadata['encoding_models']['correlation'])
    r2.append(metadata['encoding_models']['r2'])
    explained_variance.append(metadata['encoding_models']['explained_variance'])


# =============================================================================
# Plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-bmd', 'model-'+model_name, 'encoding_models_accuracy')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the single subject encoding accuracy on brain volumes
# =============================================================================
# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Convert the correlation scores to nifti images for plotting
    mask = sub_mask[s]
    corr_sub = np.empty(mask.shape, dtype=np.float32)
    corr_sub[:] = np.nan
    corr_sub[mask] = correlation[s]
    brain_mask = sub_mask[s]
    corr_sub = nib.Nifti1Image(corr_sub, sub_mask_affine[s])

    # Plot the encoding accuracy
    out_file_name = f'encoding_accuracy_sub-{sub:02d}.png'
    title = f'Encoding accuracy, subject {sub}'
    display = plotting.plot_glass_brain(
        stat_map_img=corr_sub,
        # output_file=os.path.join(save_dir, out_file_name),
        display_mode="lyrz",
        colorbar=True,
        title=title,
        threshold='auto', # 'auto' --> gives bigger brain plots
        cmap='afmhot_r',
        vmin=0,
        vmax=1,
        plot_abs=False,
        symmetric_cbar=False
    )

    # Colorbar
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12,
        fontsize=12)

    # Save
    display.savefig(os.path.join(save_dir, out_file_name), dpi=300)