
"""Test the trained encoding model predictions in-distribution (ID) on the
515 test images from NSD-core, and out-of-distribution (OOD) on the 286 images
from NSD-synthetic. Then save the encoding accuracy as part of the trained
encoding models' metadata.

Parameters
----------
subject : int
    Number of the used NSD subject.
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from scipy.stats import zscore
from tqdm import tqdm
from berg import BERG


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--models', type=list, default=['fmri-nsd_fsaverage-huze', 'fmri-nsd_fsaverage-vit_b_32', 'fmri-nsd_fsaverage-alexnet', 'fmri-nsd_fsaverage-alexnet_untrained'])
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the fMRI metadata
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

for s, sub in enumerate(tqdm(args.subjects)):


# =============================================================================
# Load the in vivo NSD-synthetic fMRI responses (and average them across repeats)
# =============================================================================
    # Load the experimental design info
    expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
        'nsdsynthetic', 'nsdsynthetic_expdesign.mat'))
    # Subtract 1 since the indices start with 1 (and not 0)
    masterordering = np.squeeze(expdesign['masterordering'] - 1)
    # Get the NSD synthetic image condition repeats

    # Load the fMRI betas
    betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
        format(sub, '02'), 'fsaverage',
        'nsdsyntheticbetas_fithrf_GLMdenoise_RR')
    lh_file_name = 'lh.betas_nsdsynthetic.mgh'
    rh_file_name = 'rh.betas_nsdsynthetic.mgh'
    lh_betas_all = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
        lh_file_name)).get_fdata())).astype(np.float32)
    rh_betas_all = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
        rh_file_name)).get_fdata())).astype(np.float32)
    # z-score the betas of each vertex within the scan session
    lh_betas_all = zscore(lh_betas_all, nan_policy='omit')
    rh_betas_all = zscore(rh_betas_all, nan_policy='omit')

    # Average the fMRI betas across repeats
    nsdsynthetic_img_num = np.unique(masterordering)
    lh_betas_test_nsdsynthetic = np.zeros((len(nsdsynthetic_img_num),
        lh_betas_all.shape[1]))
    rh_betas_test_nsdsynthetic = np.zeros((len(nsdsynthetic_img_num),
        rh_betas_all.shape[1]))
    for i, img in enumerate(nsdsynthetic_img_num):
        idx = np.where(masterordering == img)[0]
        lh_betas_test_nsdsynthetic[i] = np.nanmean(lh_betas_all[idx], 0)
        rh_betas_test_nsdsynthetic[i] = np.nanmean(rh_betas_all[idx], 0)

    # Compute the ncsnr
    # When computing the ncsnr on image conditions with different amounts of trials
    # (i.e., different sample sizes), I need to correct for this:
    # https://stats.stackexchange.com/questions/488911/combined-variance-estimate-for-samples-of-varying-sizes
    lh_num_var = np.zeros((lh_betas_all.shape[1]))
    rh_num_var = np.zeros((rh_betas_all.shape[1]))
    den_var = np.zeros((lh_betas_all.shape[1]))
    for i, img in enumerate(nsdsynthetic_img_num):
        idx = np.where(masterordering == img)[0]
        lh_num_var += np.var(lh_betas_all[idx], axis=0, ddof=1) * (len(idx) - 1)
        rh_num_var += np.var(rh_betas_all[idx], axis=0, ddof=1) * (len(idx) - 1)
        den_var += len(idx) - 1
    lh_sigma_noise = np.sqrt(lh_num_var/den_var)
    rh_sigma_noise = np.sqrt(rh_num_var/den_var)
    lh_var_data = np.var(lh_betas_all, axis=0, ddof=1)
    rh_var_data = np.var(rh_betas_all, axis=0, ddof=1)
    lh_sigma_signal = lh_var_data - (lh_sigma_noise ** 2)
    rh_sigma_signal = rh_var_data - (rh_sigma_noise ** 2)
    lh_sigma_signal[lh_sigma_signal<0] = 0
    rh_sigma_signal[rh_sigma_signal<0] = 0
    lh_sigma_signal = np.sqrt(lh_sigma_signal)
    rh_sigma_signal = np.sqrt(rh_sigma_signal)
    lh_ncsnr = lh_sigma_signal / lh_sigma_noise
    rh_ncsnr = rh_sigma_signal / rh_sigma_noise

    # Convert the ncsnr to noise ceiling
    img_reps_2 = 236
    img_reps_4 = 32
    img_reps_8 = 8
    img_reps_10 = 8
    norm_term = (img_reps_2/2 + img_reps_4/4 + img_reps_8/8 + img_reps_10/10) / \
        (img_reps_2 + img_reps_4 + img_reps_8 + img_reps_10)
    lh_noise_ceiling_nsdsynthetic = (lh_ncsnr ** 2) / \
        ((lh_ncsnr ** 2) + norm_term)
    rh_noise_ceiling_nsdsynthetic = (rh_ncsnr ** 2) / \
        ((rh_ncsnr ** 2) + norm_term)


# =============================================================================
# Save the NCSNR
# =============================================================================
    for model in args.models:

        metadata = berg.get_model_metadata(
            model,
            subject=sub
            )

        metadata['fmri']['lh_ncsnr_nsdsynthetic'] = lh_ncsnr
        metadata['fmri']['rh_ncsnr_nsdsynthetic'] = rh_ncsnr

        # Save the metadata
        model_name = model.split('-')[-1]
        save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
            'train_dataset-nsd_fsaverage', 'model-'+model_name, 'metadata')
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        file_name = 'metadata_subject-' + format(sub, '02') + '.npy'
        np.save(os.path.join(save_dir, file_name), metadata)