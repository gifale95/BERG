"""Perform searchlight RSA between the t-fMRI responses and behavioral
embeddings.

To reduce computational load, the EEG/fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 10.
criterion : str
    Criterion to define the searchlight neighborhood: 'radius' for all vertices
    within a geodesic radius, 'nearest' for k-nearest neighbors.
radius_mm : float
    Geodesic radius in millimeters (default = 10 mm), if criterion is 'radius'.
k : int
    Number of nearest geodesic neighbors (default = 10), if criterion is
    'nearest'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr
import h5py
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=20, type=float)
parser.add_argument('--criterion', default='radius', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define the vectorized correlation function
# =============================================================================
def corr_matrix(X):
    """
    Computes the correlation matrix of the input data.
    Parameters
    ----------
    X : (N, M) float array
        Input data matrix with N features and M samples.

    Returns
    -------
    corr : (M, M) float array
        Correlation matrix of the input data.
    """

    Xc = X - X.mean(axis=0)
    Xc /= np.sqrt((Xc**2).sum(axis=0))

    return (Xc.T @ Xc).astype(np.float32)


# =============================================================================
# Create the behavioral RDM
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the THINGS EEG2 test image category number based on the original THINGS
# database
metadata_things = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
    )
test_img_concepts_THINGS = metadata_things['encoding_models']['test_img_info']\
    ['test_img_concepts_THINGS']

# Load the behavioral embeddings (the behavioral emebddings can be downloaded
# from: https://osf.io/f5rn6/overview)
embedding_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'behavioral_modeling', 'spose_embedding_66d_sorted.txt')
beh_embeddings_all = np.array(pd.read_csv(embedding_dir, delim_whitespace=True,
    header=None)).astype(np.float32)

# Retain the embeddings from the 200 test image concepts
idx_test = np.zeros(len(test_img_concepts_THINGS), dtype=int)
for i, img in enumerate(test_img_concepts_THINGS):
    idx_test[i] = int(img[:5]) - 1
beh_embeddings = beh_embeddings_all[idx_test]

# Create the RDM
beh_rdm = 1 - corr_matrix(beh_embeddings.T)

# Take the lower triangle of the behavior RDM
idx_tril = np.tril_indices(len(beh_rdm), -1)
beh_rdm_tril = beh_rdm[idx_tril]


# =============================================================================
# Load and append the in vivo EEG test responses across subjects
# =============================================================================
# Loop across subjects
for es, esub in enumerate(tqdm(args.eeg_subjects)):

    # Load the EEG responses, and average them across repeats
    eeg_dir_test = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', f'eeg_sub-{esub:02d}_split-test.h5')
    eeg_test_sub = np.mean(h5py.File(eeg_dir_test, 'r')['eeg'][:],
        1).astype(np.float32)

    # Append the EEG channel responses across subjects
    if es == 0:
        eeg_test = eeg_test_sub
    else:
        eeg_test = np.append(eeg_test, eeg_test_sub, 1)
    del eeg_test_sub

# Load the EEG time points
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Get the ROI vertex indices
# =============================================================================
roi_idx = {}

# Load the fMRI metadata
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Loop across ROIs
rois = ['V1', 'V2', 'V3', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'OPA', 'PPA',
    'EBA', 'FBA', 'early', 'intermediate', 'ventral', 'lateral', 'parietal']
for r, roi in enumerate(rois):

    # Loop across subjects and hemispheres
    for h, hemi in enumerate(args.hemispheres):

        # Get the indices of the ROI vertices
        if roi in ['V1', 'V2', 'V3']:
            idx_roi = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}v'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}d'])
            idx_roi.sort()
        elif roi in ['FFA', 'VWFA', 'FBA']:
            idx_roi = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-1'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{roi}-2'])
            idx_roi.sort()
        elif roi in ['intermediate']:
            idx_roi = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois']['midventral'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois']['midlateral'])
            idx_roi = np.append(idx_roi,
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois']['midparietal'])
            idx_roi.sort()
        else:
            idx_roi = metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][roi]

        # NCSNR and encoding accuracy vertex selection
        ncsnr = metadata_fmri['fmri'][hemi+'_ncsnr'][idx_roi]
        idx_ncsnr = ncsnr >= args.ncsnr_threshold
        encoding = metadata_fmri['encoding_models']\
            [hemi+'_explained_variance_nsdcore'][idx_roi]
        idx_encoding = encoding >= args.encoding_threshold
        idx_vertex = np.logical_and(idx_ncsnr, idx_encoding)
        idx_roi = idx_roi[idx_vertex]

        # Store the ROI vertex indices
        roi_idx[f'{hemi}_{roi}'] = idx_roi


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Only select vertices falling within the NSD visual streams
n_vertices = 163842
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
idx_v = {}
for hemi in args.hemispheres:
    idx = np.zeros(n_vertices, dtype=int)
    for stream in streams:
        idx[metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
    idx_v[hemi] = np.where(idx == 1)[0]
    del idx

# Empty result dictionary
rsa_roi = {}

# Loop across EEG time points
for t in tqdm(range(len(times))):

    # Empty t-fMRI response array of shape:
    # (N Images, 163842 Vertices)
    tfmri = {}
    for hemi in args.hemispheres:
        tfmri[hemi] = np.zeros((len(eeg_test), n_vertices), dtype=np.float32)

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
            f'hemi-{hemi}_eeg_time-{t:03d}.npy')
        reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'encoding_fusion_weights', file_name), allow_pickle=True).item()

        # Instantiate the fusion regression model
        reg = LinearRegression()
        reg.coef_ = reg_param['coef_']
        reg.intercept_ = reg_param['intercept_']
        reg.n_features_in_ = reg_param['n_features_in_']

        # Generate the t-fMRI responses for the test images with in vivo EEG
        tfmri[hemi][:,idx_v[hemi]] = reg.predict(eeg_test[:,:,t])
        del reg_param, reg


# =============================================================================
# Perform RSA
# =============================================================================
    # Loop across ROIs
    for r, roi in enumerate(rois):

        # Empty ROI correlation array of shape:
        # (140 EEG time points)
        if t == 0:
            rsa_roi[roi] = np.zeros((len(times)), dtype=np.float32)

        # Loop across hemispheres
        for h, hemi in enumerate(args.hemispheres):

            # Append the vertex response of the chosen ROI across the two
            # hemispheres
            if hemi == 'lh':
                response = tfmri[hemi][:,roi_idx[f'{hemi}_{roi}']]
            else:
                response = np.append(response,
                    tfmri[hemi][:,roi_idx[f'{hemi}_{roi}']], 1)

        # Remove NaN values
        idx_nan = np.isnan(response).any(axis=0)
        response = response[:,~idx_nan]

        # Create the fMRI RDM
        fmri_rdm = 1 - corr_matrix(response.T)
        del response

        # Perform RSA
        rsa_roi[roi][t] = pearsonr(beh_rdm_tril, fmri_rdm[idx_tril])[0]
        del fmri_rdm
    del tfmri


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa_roi': rsa_roi,
    'metadata_fmri': metadata_fmri
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'behavioral_modeling_roiwise', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = f'rsa_fmri_sub-{args.fmri_subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), results)