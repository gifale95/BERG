"""Create Pycortex' ROI labels based on NSD's floc-words ROIs (OVWFA, VWFA-1,
VWFA-2, mfs-words, mTL-words).

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
import numpy as np
from berg import BERG
import cortex

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--berg_dir', default='../brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the ROIs
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
rois = ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']
for r, roi in enumerate(rois):
    idx = metadata['fmri'][args.hemisphere+'_fsaverage_rois'][roi]
    fsaverage_rois[idx] = r + 1


# =============================================================================
# Create the ROI labels
# =============================================================================
# Prepare the data in Pycortex format
data_nan = np.empty(163842)
data_nan[:] = np.nan
if args.hemisphere == 'lh':
    data = np.append(fsaverage_rois, data_nan)
elif args.hemisphere == 'rh':
    data = np.append(data_nan, fsaverage_rois)
subject = 'fsaverage_nsd_sub-0' + str(args.subject)
vertex_data = cortex.Vertex(data, subject, vmin=1, vmax=len(rois), cmap='gist_rainbow',
    with_colorbar=True)

# Create the ROI labels: https://gallantlab.org/pycortex/generated/cortex.utils.add_roi.html
cortex.utils.add_roi(vertex_data, name='floc-words_'+args.hemisphere,
    with_colorbar=FalTruese)

# Then draw the ROI labels in Inkscape.