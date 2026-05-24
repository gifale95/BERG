"""Predict fMRI responses for the THINGS images.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
roi : str
    Used ROI.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Predict fMRI responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Get the fMRI ROI indices
# =============================================================================
# Load the fMRI metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

vertices = {}
n_vertices = 163842

# Loop across hemisphers
for hemi in args.hemispheres:

    # Only select stream vertices of the chosen ROI
    if args.roi in ['V1', 'V2', 'V3']:
        idx_r = np.append(
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}v'],
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}d'])
        idx_r.sort()
    elif args.roi in ['FFA', 'VWFA', 'FBA']:
        idx_r = np.append(
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}-1'],
            metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}-2'])
        idx_r.sort()
    else:
        idx_r = metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}']
        idx_r.sort()
    idx_roi = np.zeros(n_vertices, dtype=bool)
    idx_roi[idx_r] = 1
    idx_roi = np.where(idx_roi)[0]

   # Only select vertices with NCSNR above threshold
    ncsnr = metadata_fmri['fmri'][f'{hemi}_ncsnr']
    idx_ncsnr = np.where(ncsnr >= args.ncsnr_threshold)[0]
    idx = np.intersect1d(idx_roi, idx_ncsnr)

    # Store the vertex indices
    vertices[hemi] = np.zeros(n_vertices, dtype=int)
    vertices[hemi][idx] = 1


# =============================================================================
# Load the fMRI encoding model
# =============================================================================
model = berg.get_encoding_model(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject,
    selection={
        'lh_vertices': vertices['lh'],
        'rh_vertices': vertices['rh']
        }
    )


# =============================================================================
# Access the THINGS images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'images',
    'things.h5')
images = h5py.File(data_dir, 'r')['images'][:]


# =============================================================================
# Predict the fMRI responses to images
# =============================================================================
# Predict the fMRI responses
fmri = berg.encode(model, images)

# Average the predicted fMRI responses across the vertices from the same
# ROI, to get that ROIs univariate response
fmri_uni= np.mean(np.append(fmri[0], fmri[1], 1), 1)
del images, fmri


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'univariate_rnc_rotem', 'fmri_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_things.npy'

np.save(os.path.join(save_dir, file_name), fmri_uni)