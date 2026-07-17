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
parser.add_argument('--ncsnr_threshold', default=0.1, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/ccn_2026_tutorial', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico fMRI responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the fMRI encoding model
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

model = berg.get_encoding_model(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject,
    device='auto'
    )


# =============================================================================
# Access the images
# =============================================================================
# Load the images
data_dir = os.path.join(args.project_dir, 'images',
    f'images_{args.image_set}.h5')
images = h5py.File(data_dir, 'r')['images']

# Swap axes to have the images in the shape (n_images, channels, height, width)
images = np.transpose(images, (0, 3, 1, 2))


# =============================================================================
# Predict the fMRI responses to images
# =============================================================================
# Empty result dictionaries
fmri = {}
fmri['roi_set-nsd'] = {}
fmri['roi_set-hcp'] = {}

# Divide the in silico fMRI responses generation in 10 batches
n_images = len(images)
n_batches = 10
batch_size = n_images // n_batches
for b in tqdm(range(n_batches)):
    start_idx = b * batch_size
    end_idx = start_idx + batch_size if b < n_batches - 1 else n_images

    # Predict the fMRI responses to images
    fmri_wb, metadata = berg.encode(
        model,
        images[start_idx:end_idx],
        return_metadata=True
    )

    # Get the indices of vertices with NCSNR above threshold
    idx_ncsnr = {}
    for h, hemi in enumerate(['lh', 'rh']):
        idx_ncsnr[hemi] = metadata['fmri'][f'{hemi}_ncsnr'] > args.ncsnr_threshold


# =============================================================================
# Get the univariate responses for the NSD ROIs
# =============================================================================
    nsd_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-2",
        "OFA", "FFA-1", "FFA-2", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", 
        "mfs-words", "early", "midventral", "midlateral", "midparietal", "ventral",
        "lateral", "parietal"]

    for roi in nsd_rois:
        for h, hemi in enumerate(['lh', 'rh']):

            # Get the responses of each ROI
            idx_roi = np.zeros(163842, dtype=bool)
            idx_roi[metadata['fmri'][f'{hemi}_fsaverage_rois'][roi]] = True
            idx_vertices = np.logical_and(idx_roi, idx_ncsnr[hemi])
            if h == 0:
                fmri_roi = fmri_wb[h][:,idx_vertices]
            else:
                fmri_roi = np.append(fmri_roi, fmri_wb[h][:,idx_vertices], 1)
            
        # Average the ROI responses across vertices of both hemispheres
        if b == 0:
            fmri['roi_set-nsd'][roi] = np.nanmean(fmri_roi, 1)
        else:
            fmri['roi_set-nsd'][roi] = np.append(fmri['roi_set-nsd'][roi],
                np.nanmean(fmri_roi, 1))
        del fmri_roi


# =============================================================================
# Get the univariate responses for the HPC-MMP1 ROIs
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
        for h, hemi in enumerate(['lh', 'rh']):

            # Get the responses of each ROI
            roi_id = roi_label[roi_label['label'] == roi].iloc[0, 0]
            idx_roi = np.zeros(163842, dtype=bool)
            idx_roi[np.where(roi_map[hemi] == roi_id)[0]] = True
            idx_vertices = np.logical_and(idx_roi, idx_ncsnr[hemi])
            if h == 0:
                fmri_roi = fmri_wb[h][:,idx_vertices]
            else:
                fmri_roi = np.append(fmri_roi, fmri_wb[h][:,idx_vertices], 1)

        # Average the ROI responses across vertices of both hemispheres
        if b == 0:
            fmri['roi_set-hcp'][roi] = np.nanmean(fmri_roi, 1)
        else:
            fmri['roi_set-hcp'][roi] = np.append(fmri['roi_set-hcp'][roi],
                np.nanmean(fmri_roi, 1))
        del fmri_roi
    del fmri_wb
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'insilico_fmri')

# Save the in silico fMRI responses
file_name = (f'insilico_fmri_sub-{args.fmri_subject:02d}_'
    f'imageset-{args.image_set}.h5')
np.save(os.path.join(save_dir, file_name), fmri)