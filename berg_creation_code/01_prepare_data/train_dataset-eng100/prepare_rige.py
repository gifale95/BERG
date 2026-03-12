"""Extract ROI masks from the pycortex database of the LeBel et al. (2023)
dataset and add them to the BERG metadata files.

This script loads hand-defined ROI masks from pycortex functional localizers,
maps them from 3D volume space to the encoding model's voxel space using the
mask_thick.nii.gz brain mask, and stores them as binary arrays in the BERG
metadata under a 'roi' key.

ROIs include regions from visual category, motor, and auditory localizers:
AC, Broca, EBA, FEF, FFA, IPS, M1H, M1F, M1M, OFA, PPA, RSC, sPMv, etc.
ROIs with zero voxels in the functional space are excluded.

If a metadata file already exists for a subject, the 'roi' key is added
(or overwritten). If no metadata file exists, a new one is created.

Parameters
----------
deep_fmri_repo : str
    Path to the cloned deep-fMRI-dataset repository.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
subjects : list of str
    Subject identifiers. Default: all 8 subjects.
"""


import os
import sys
import argparse
import numpy as np
import nibabel as nib
from os.path import join


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--deep_fmri_repo', type=str, required=True,
    help='Path to the cloned deep-fMRI-dataset repository.')
parser.add_argument('--berg_dir', type=str, required=True,
    help='Path to the BERG data directory.')
parser.add_argument('--subjects', nargs='+', type=str,
    default=['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06',
             'UTS07', 'UTS08'],
    help='Subject identifiers. Default: all 8 subjects.')
args = parser.parse_args()

print('>>> Extract ROI masks from pycortex database <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Setup pycortex with the dataset's database
# =============================================================================
db_path = join(args.deep_fmri_repo, 'data', 'ds003020', 'derivative',
    'pycortex-db')
assert os.path.isdir(db_path), \
    f'Could not find pycortex database at: {db_path}. ' \
    f'Make sure the derivative data has been downloaded.'

import cortex
import cortex.utils as cu

new_db = cortex.database.Database(db_path)
cortex.db = new_db
cu.db = new_db


# =============================================================================
# Define BERG metadata path
# =============================================================================
metadata_dir = join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-eng1000', 'model-ridge', 'metadata')


# =============================================================================
# Extract ROIs for each subject
# =============================================================================
for subject in args.subjects:
    print(f'\n{"="*60}')
    print(f'Extracting ROIs for subject: {subject}')
    print(f'{"="*60}')

    # -----------------------------------------------------------------
    # Load the brain mask (mask_thick.nii.gz)
    # -----------------------------------------------------------------
    # Find the transform name for this subject.
    xfm_dir = join(db_path, subject, 'transforms')
    xfm_names = os.listdir(xfm_dir)
    xfm_names = [x for x in xfm_names if not x.startswith('.')]
    assert len(xfm_names) >= 1, \
        f'No transforms found for {subject} in {xfm_dir}'
    xfmname = xfm_names[0]
    print(f'  Transform: {xfmname}')

    mask_path = join(xfm_dir, xfmname, 'mask_thick.nii.gz')
    assert os.path.exists(mask_path), \
        f'mask_thick.nii.gz not found at: {mask_path}'

    # Load and transpose to match pycortex volume axis order.
    # nibabel loads as (x, y, z) but pycortex uses (z, y, x).
    mask_vol = nib.load(mask_path).get_fdata()
    mask_bool = np.transpose(mask_vol, (2, 1, 0)).astype(bool)
    n_voxels = np.count_nonzero(mask_bool)
    print(f'  mask_thick shape (transposed): {mask_bool.shape}')
    print(f'  Number of voxels in mask: {n_voxels}')

    # -----------------------------------------------------------------
    # Load ROI masks from pycortex
    # -----------------------------------------------------------------
    rois = cu.get_roi_masks(subject, xfmname)
    print(f'  ROIs found in pycortex: {len(rois)}')

    # -----------------------------------------------------------------
    # Map ROIs from 3D volume to encoding model voxel space
    # -----------------------------------------------------------------
    roi_dict = {}
    for roi_name in sorted(rois.keys()):
        roi_3d = rois[roi_name]
        roi_flat = roi_3d[mask_bool].astype(bool)
        n_roi_voxels = np.count_nonzero(roi_flat)
        if n_roi_voxels > 0:
            roi_dict[roi_name] = roi_flat
            print(f'    {roi_name}: {n_roi_voxels} voxels')
        else:
            print(f'    {roi_name}: 0 voxels (skipped)')

    print(f'  Total ROIs with voxels: {len(roi_dict)}')

    # -----------------------------------------------------------------
    # Load or create metadata, add ROI key
    # -----------------------------------------------------------------
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = join(metadata_dir, f'sub-{subject}.npy')

    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        print(f'  Loaded existing metadata from: {metadata_path}')
    else:
        metadata = {}
        print(f'  No existing metadata found, creating new.')

    metadata['roi'] = roi_dict
    np.save(metadata_path, metadata)
    print(f'  Saved metadata with ROIs to: {metadata_path}')


print(f'\n{"="*60}')
print('Done. ROI masks extracted and saved.')
print(f'{"="*60}')