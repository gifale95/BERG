"""Generate in silico fMRI responses for the 50,000 ImageNet or 40,670 MS COCO
images.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
image_set : str
    The image set to use for generating in silico fMRI responses. Possible
    values are: 'imagenet' (ImageNet) and 'coco' (MS COCO).
berg_dir : str
    Directory of the BERG.
project_dir : str
    Directory of the ImageNet image set.

"""

import argparse
import gc
import os
import pandas as pd
import numpy as np
from berg import BERG
import h5py
import nibabel as nib
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--image_set', default='imagenet', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/ccn_2026_tutorial', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico fMRI responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the fMRI metadata
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )


# =============================================================================
# Get the indices of vertices with NCSNR above threshold
# =============================================================================
idx_ncsnr = {}
for h, hemi in enumerate(['lh', 'rh']):
    idx_ncsnr[hemi] = metadata['fmri'][f'{hemi}_ncsnr'] > args.ncsnr_threshold


# =============================================================================
# Get the vertex number of each NSD ROI
# =============================================================================
nsd_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-2",
    "OFA", "FFA-1", "FFA-2", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", 
    "mfs-words", "early", "midventral", "midlateral", "midparietal", "ventral",
    "lateral", "parietal"]

for roi in nsd_rois:
    n_vertex = 0
    for h, hemi in enumerate(['lh', 'rh']):

        # Get the responses of each ROI
        idx_roi = np.zeros(163842, dtype=bool)
        idx_roi[metadata['fmri'][f'{hemi}_fsaverage_rois'][roi]] = True
        n_vertex += sum(np.logical_and(idx_roi, idx_ncsnr[hemi]))

    # Print the vertex number of each ROI
    print(f"ROI: {roi}, Vertex Number: {(n_vertex)}")


# =============================================================================
# Get the vertex number of each HCP-MMP1 ROI
# =============================================================================
# Load the HCP ROI labels
roi_label_file = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer',
    'fsaverage', 'label', 'HCP_MMP1.mgz.ctab')
roi_label = pd.read_csv(roi_label_file, sep=r"\s+", header=None,
    names=["id", "label"])
hcp_rois = roi_label['label'].to_list()
hcp_rois.remove("Unknown")

# Load the HCP ROI maps
roi_map = {}
lh_roi_file = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer',
    'fsaverage', 'label', 'lh.HCP_MMP1.mgz')
roi_map['lh'] = np.squeeze(nib.load(lh_roi_file).get_fdata())
rh_roi_file = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer',
    'fsaverage', 'label', 'rh.HCP_MMP1.mgz')
roi_map['rh'] = np.squeeze(nib.load(rh_roi_file).get_fdata())

for roi in hcp_rois:
    n_vertex = 0
    for h, hemi in enumerate(['lh', 'rh']):

        # Get the vertex number of each ROI
        roi_id = roi_label[roi_label['label'] == roi].iloc[0, 0]
        idx_roi = np.zeros(163842, dtype=bool)
        idx_roi[np.where(roi_map[hemi] == roi_id)[0]] = True
        n_vertex += sum(np.logical_and(idx_roi, idx_ncsnr[hemi]))
    print(f"ROI: {roi}, Vertex Number: {n_vertex}")