"""Predict t-fMRI responses using in vivo or in silico EEG responses for
different image sets.

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
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
eeg_reps : str
    String indicating whether to use EEG responses averaged across 'even',
    'odd', or 'all' repeats.
images : str
    If 'things_eeg_2_vivo', use the in vivo EEG responses for the 200 THINGS
    EEG2 test images.
    If 'things_eeg_2_silico', use the in silico EEG responses for the 200
    THINGS EEG2 test images.
    If 'nsd_515_shared', use the in silico EEG responses for the 515 NSD shared
    images.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
import h5py
from PIL import Image
import gc
import torch
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_reps', default='all', type=str)
parser.add_argument('--images', default='things_eeg_2_vivo', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Predict t-fMRI responses <<<')
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

idx_v = {}

# Loop across hemisphers
for h, hemi in enumerate(args.hemispheres):

    # Only select vertices falling within the NSD visual streams
    n_vertices = 163842
    idx_streams = np.zeros(n_vertices, dtype=bool)
    streams = ['early', 'midventral', 'midlateral', 'midparietal',
        'ventral', 'lateral', 'parietal']
    for stream in streams:
        idx_streams[metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
    idx_streams = np.where(idx_streams)[0]

    # Only select stream vertices with NCSNR above threshold
    ncsnr = metadata_fmri['fmri'][f'{hemi}_ncsnr']
    idx_ncsnr = np.where(ncsnr[idx_streams] >= args.ncsnr_threshold)[0]

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
    idx_roi = idx_roi[idx_streams]
    idx_roi = np.where(idx_roi)[0]

    # Get the indices of ROI vertices with NCSNR above threshold
    idx_v[hemi] = np.intersect1d(idx_roi, idx_ncsnr)


# =============================================================================
# Load and append the in vivo THINGS EEG2 test responses across subjects
# =============================================================================
if args.images == 'things_eeg_2_vivo':

    # Loop across subjects
    for es, esub in enumerate(tqdm(args.eeg_subjects)):

        # Load the EEG responses
        eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
            'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
        eeg_test_sub = h5py.File(eeg_dir_test, 'r')['eeg'][:].astype(np.float32)

        # Average the EEG responses across repeats
        if args.eeg_reps == 'even':
            idx = np.arange(0, eeg_test_sub.shape[1], 2)
        elif args.eeg_reps == 'odd':
            idx = np.arange(1, eeg_test_sub.shape[1], 2)
        elif args.eeg_reps == 'all':
            idx = np.arange(0, eeg_test_sub.shape[1], 1)
        eeg_test_sub = np.mean(eeg_test_sub[:,idx], 1)

        # Append the EEG channel responses across subjects
        if es == 0:
            eeg = eeg_test_sub
        else:
            eeg = np.append(eeg, eeg_test_sub, 1)
        del eeg_test_sub



# =============================================================================
# Load the images for which the in silico EEG responses will be generated
# =============================================================================
else:

    # 200 THINGS EEG2 test images
    if args.images == 'things_eeg_2_silico':
        # Initialize BERG
        berg = BERG(berg_dir=args.berg_dir)
        # Load the metadata
        metadata_eeg = berg.get_model_metadata(
            'eeg-things_eeg_2-vit_b_32',
            subject=1
            )
        # Get the test image file names
        test_img_files = metadata_eeg['encoding_models']['test_img_info']\
            ['test_img_files']
        # Loop across test image files
        images = []
        for file in tqdm(test_img_files):
            # Find correct subfolder
            img_path = None
            for root, _, files in os.walk(os.path.join(args.things_dir)):
                if file in files:
                    img_path = os.path.join(root, file)
                    break
            # Load and transform the image
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
            img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
            images.append(img)
        # Format the images to a numpy array
        images = np.array(images)

    # 515 NSD shared images
    elif args.images == 'nsd_515_shared':
        # Initialize BERG
        berg = BERG(berg_dir=args.berg_dir)
        # Get the test image number
        metadata = berg.get_model_metadata(
            'fmri-nsd_fsaverage-huze',
            subject=1
        )
        test_img_num = metadata['encoding_models']['test_img_num']
        # Load the test images
        sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli',
            'stimuli', 'nsd', 'nsd_stimuli.hdf5'), 'r')
        sdataset = sf.get('imgBrick')
        images = sdataset[test_img_num]
        images = np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)


# =============================================================================
# Predict the in silico EEG responses
# =============================================================================
    # Loop across EEG subjects
    for es, esub in enumerate(args.eeg_subjects):

        # Load the EEG encoding model
        model = berg.get_encoding_model(
            'eeg-things_eeg_2-vit_b_32',
            subject=esub
            )

        # Predict the in silico EEG responses, and append them across subjects
        # across the channels dimension
        if es == 0:
            eeg = berg.encode(model, images)
        else:
            eeg = np.append(eeg, berg.encode(model, images), 2)

        # Remove the model from memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Average the EEG responses across repeats
    if args.eeg_reps == 'even':
        idx = np.arange(0, eeg.shape[1], 2)
    elif args.eeg_reps == 'odd':
        idx = np.arange(1, eeg.shape[1], 2)
    elif args.eeg_reps == 'all':
        idx = np.arange(0, eeg.shape[1], 1)
    eeg = np.mean(eeg[:,idx], 1)


# =============================================================================
# Loop across EEG time points
# =============================================================================
# Get the EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=args.eeg_subjects[0]
)
times = metadata_eeg['eeg']['times']

# Loop across EEG time points
for t in tqdm(range(len(times))):


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
    # Loop across hemisphers
    for h, hemi in enumerate(args.hemispheres):

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
            f'hemi-{hemi}_eeg_train_trials-all_eeg_time-{t:03d}.npy')
        reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'encoding_fusion_weights', file_name), allow_pickle=True).item()

        # Instantiate the fusion regression model
        reg = LinearRegression()
        reg.coef_ = reg_param['coef_'][idx_v[hemi]]
        reg.intercept_ = reg_param['intercept_'][idx_v[hemi]]
        reg.n_features_in_ = reg_param['n_features_in_']

        # Generate the t-fMRI responses
        tfmri_part = np.expand_dims(reg.predict(eeg[:,:,t]), 2)
        del reg_param, reg

        # Append the t-fMRI responses across hemispheres and time points
        if h == 0:
            tfmri_hemi = tfmri_part
        else:
            tfmri_hemi = np.append(tfmri_hemi, tfmri_part, 1)
            if t == 0:
                tfmri = tfmri_hemi
            else:
                tfmri = np.append(tfmri, tfmri_hemi, 2)
        del tfmri_part
    del tfmri_hemi

# Delete the EEG responses
del eeg


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'representational_format_evolution',
    'tfmri_responses')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'tfmri_sub-{args.fmri_subject:02d}_roi-{args.roi}_'
    f'eeg_reps-{args.eeg_reps}_images-{args.images}.npy')

np.save(os.path.join(save_dir, file_name), tfmri)