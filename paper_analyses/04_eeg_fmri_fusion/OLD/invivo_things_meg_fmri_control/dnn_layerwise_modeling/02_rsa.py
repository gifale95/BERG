"""Perform RSA between t-fMRI responses and DNN layerwise features,
independently for each fMRI ROI.

Parameters
----------
fmri_subject : int
    THINGS fMRI1 subject identifiers. Valid subject identifiers are integers
    from 1 to 3.
meg_subjects : list
    List containing the subject identifiers for the THINGS MEG1 subjects. Valid
    subject identifiers are integers from 1 to 4.
noise_ceiling_threshold : float
    The threshold on the noise ceiling for voxel selection.
dnn_model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import h5py
from berg import BERG
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--noise_ceiling_threshold', default=20, type=float)
parser.add_argument('--dnn_model', default='alexnet', type=str)
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
# Load the DNN layerwise RDMs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'dnn_layerwise_modeling', 'dnn_rdms',
    'dnn_rdms_'+args.dnn_model+'.npy')

dnn_rdms = np.load(data_dir, allow_pickle=True).item()

# Take the lower triangle of the DNN RDMs
idx_tril = np.tril_indices(len(dnn_rdms[list(dnn_rdms.keys())[0]]), -1)
dnn_rdm_tril = {}
for key, val in dnn_rdms.items():
    dnn_rdm_tril[key] = val[idx_tril]


# =============================================================================
# Get the 100 THINGS fMRI1 test image file names
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the fMRI metadata
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

# Get the test image file names
test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
unique_test_stimuli = np.unique(test_stimuli_fmri)


# =============================================================================
# Load and append the in vivo MEG test responses across subjects
# =============================================================================
# Loop across MEG subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Load the MEG responses
    meg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_dir, 'r')['neural_data']

    # Time point selection
    tmax = 0.595
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]
    meg_test_all = meg_test_all[:,:,time_idx].astype(np.float32)

    # Average the MEG responses across repetitions for the images shared with
    # the fMRI
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_test = meg_test_sub
    else:
        meg_test = np.append(meg_test, meg_test_sub, 1)
    del meg_test_all, meg_test_sub


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Load the encoding fusion model regression weights
weight_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'encoding_fusion_weights',
    f'weights_fmri_sub-{args.fmri_subject:02d}.npy')
reg_param = np.load(weight_dir, allow_pickle=True).item()

# Loop across ROIs
rsa = {}
rois = ['V1', 'V2', 'V3', 'hV4', 'FFA', 'OFA', 'EBA', 'PPA', 'RSC', 'TOS',
    'LOC', 'IT']
for r, roi in enumerate(tqdm(rois)):

    # Empty RSA result arrays of shape:
    # (N MEG time points)
    rsa[roi] = {}
    for key in dnn_rdms.keys():
        rsa[roi][key] = np.zeros((len(times)), dtype=np.float32)

    # Noise ceiling voxel selection
    if roi in ['FFA', 'FFA', 'OFA', 'EBA', 'PPA', 'RSC', 'TOS', 'LOC']:
        noise_ceiling_lh = metadata_fmri['encoding_model']\
            ['noise_ceiling_testset'][metadata_fmri['roi'][f'l{roi}']]
        noise_ceiling_rh = metadata_fmri['encoding_model']\
            ['noise_ceiling_testset'][metadata_fmri['roi'][f'r{roi}']]
        idx_nc_lh = noise_ceiling_lh >= args.noise_ceiling_threshold
        idx_nc_rh = noise_ceiling_rh >= args.noise_ceiling_threshold
    else:
        noise_ceiling = metadata_fmri['encoding_model']\
            ['noise_ceiling_testset'][metadata_fmri['roi'][roi]]
        idx_nc = noise_ceiling >= args.noise_ceiling_threshold

    # Loop across MEG time points
    for t in tqdm(range(len(times))):

        # Instantiate the fusion regression model, while selecting only the
        # voxels that pass the noise ceiling threshold
        if roi in ['FFA', 'FFA', 'OFA', 'EBA', 'PPA', 'RSC', 'TOS', 'LOC']:
            reg = LinearRegression()
            coef_ = reg_param[f'l{roi}']['coef_'][t][idx_nc_lh]
            intercept_ = reg_param[f'l{roi}']['intercept_'][t][idx_nc_lh]
            coef_ = np.append(coef_,
                reg_param[f'r{roi}']['coef_'][t][idx_nc_rh], 0)
            intercept_ = np.append(intercept_,
                reg_param[f'r{roi}']['intercept_'][t][idx_nc_rh], 0)
            reg.coef_ = coef_
            reg.intercept_ = intercept_
            reg.n_features_in_ = reg_param[f'l{roi}']['n_features_in_'][t]
            del coef_, intercept_
        else:
            reg = LinearRegression()
            reg.coef_ = reg_param[roi]['coef_'][t][idx_nc]
            reg.intercept_ = reg_param[roi]['intercept_'][t][idx_nc]
            reg.n_features_in_ = reg_param[roi]['n_features_in_'][t]

        # Generate the t-fMRI responses for the test images with in vivo MEG
        tfmri = reg.predict(meg_test[:,:,t])
        del reg


# =============================================================================
# Perform RSA
# =============================================================================
        # Create the fMRI RDM
        tfmri_rdm = 1 - corr_matrix(tfmri.T)
        del tfmri

        # Perform RSA with each DNN layer
        for key, val in dnn_rdm_tril.items():
            rsa[roi][key][t] = pearsonr(val, tfmri_rdm[idx_tril])[0]
        del tfmri_rdm


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'dnn_layerwise_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'rsa_fmri_sub-{args.fmri_subject:02d}'
            f'_dnn_model-{args.dnn_model}.npy')

np.save(os.path.join(save_dir, file_name), rsa)