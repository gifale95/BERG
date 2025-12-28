"""Create Pycortex' ROI labels based on NSD's pRF visual ROIs (V1, V2, V3, V4).

Parameters
----------
subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemisphere : str
    The hemisphere to use to draw the ROI masks.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import nibabel as nib
from berg import BERG
import cortex
import cortex.polyutils
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the EVC ROIs
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the encoding model
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.subject
    )

# Load the ROI indices
fsaverage_rois = np.empty(163842)
fsaverage_rois[:] = np.nan
rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
for r, roi in enumerate(rois):
    idx = metadata['fmri'][args.hemisphere+'_fsaverage_rois'][roi]
    fsaverage_rois[idx] = r + 1


# =============================================================================
# Create the stream labels
# =============================================================================
# Prepare the data in Pycortex format
data_nan = np.empty(163842)
data_nan[:] = np.nan
if args.hemisphere == 'lh':
    data = np.append(fsaverage_rois, data_nan)
elif args.hemisphere == 'rh':
    data = np.append(data_nan, fsaverage_rois)
subject = 'fsaverage_nsd_sub-0' + str(args.subject)
vertex_data = cortex.Vertex(data, subject, vmin=1, vmax=7, cmap='gist_rainbow',
    with_colorbar=False)

# Create the ROI labels: https://gallantlab.org/pycortex/generated/cortex.utils.add_roi.html
cortex.utils.add_roi(vertex_data, name='prf-visualrois_'+args.hemisphere,
    with_colorbar=False)

# Then manually draw the ROI labels using Inkscape paths.