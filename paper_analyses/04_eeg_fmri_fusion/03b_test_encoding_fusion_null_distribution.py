"""Use the trained encoding fusion models to predict time-resolved fMRI
(t-fMRI) responses for the 200 THINGS EEG2 test images, and use these responses
to create the prediction accuracy null distribution of each ROI through
permutation tests.

To reduce computational load, the EEG-fMRI fusion encoding models are only
trained, tested, and used for vertices falling within the NSD visual streams.

The in vivo THINGS EEG2 responses are prepared using this code:
https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_eeg_2

Parameters
----------
fmri_subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
roi : str
    Name of used ROI.
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
eeg_train_trials : str
    String indicating which training EEG response trials are used. Possible
    values  are: 'all' (all trials), 'even' (even trials), and 'odd' (odd
    trials).
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
n_iter : int
    Amount of iterations for creating the prediction accuracy null distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import gc
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--eeg_train_trials', default='all', type=str)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Encoding fusion null distribution <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Get the fMRI ROI indices
# =============================================================================
# Load the fMRI metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
n_vertices = 163842
idx_streams = np.zeros(n_vertices, dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_streams[metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_streams = np.where(idx_streams == 1)[0]

# NCSNR vertex selection
ncsnr = metadata_fmri['fmri'][args.hemisphere+'_ncsnr'][idx_streams]
idx_ncsnr = ncsnr >= args.ncsnr_threshold
idx_streams = idx_streams[idx_ncsnr]

# Get the indices of ROI vertices within the NSD visual streams
if args.roi in ['V1', 'V2', 'V3']:
    idx_r = np.append(
        metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][f'{args.roi}v'],
        metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][f'{args.roi}d'])
    idx_r.sort()
elif args.roi in ['FFA', 'VWFA', 'FBA']:
    idx_r = np.append(
        metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][f'{args.roi}-1'],
        metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][f'{args.roi}-2'])
    idx_r.sort()
else:
    idx_r = metadata_fmri['fmri'][f'{args.hemisphere}_fsaverage_rois'][f'{args.roi}']
    idx_r.sort()
idx_roi = np.zeros(n_vertices, dtype=bool)
idx_roi[idx_r] = 1
idx_roi = idx_roi[idx_streams]
idx_roi = np.where(idx_roi)[0]


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
# Load the fMRI responses
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'insilico_fmri_responses', 'imageset-things_eeg_2')
file_name = f'fmri_sub-{args.fmri_subject:02d}_{args.hemisphere}_split-test.h5'
fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri']

# Only select the fMRI vertices falling within the NSD visual streams and from
# the selected ROI
fmri_test = fmri_test[:,idx_streams]
fmri_test = fmri_test[:,idx_roi]

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + eps)
del fmri_test


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
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = metadata_eeg['eeg']['times']


# =============================================================================
# Create the permutation indices for the prediction accuracy null distribution
# =============================================================================
n_img = fmri_test_z.shape[0]

idx_perm = np.zeros((n_img, args.n_iter), dtype=int)
idx_perm_pred = np.zeros((n_img, args.n_iter), dtype=int)

for i in tqdm(range(args.n_iter)):

    idx_perm[:,i] = resample(np.arange(n_img), replace=False)
    idx_perm_pred[:,i] = resample(np.arange(n_img), replace=False)


# =============================================================================
# Loop across EEG time points
# =============================================================================
# Empty correlation null distribution array for each ROI of shape:
# (N ROI vertices, 140 EEG time points, N iterations)
correlation_roi_null_dist = np.zeros((len(idx_roi), len(times), args.n_iter),
    dtype=np.float32)

# Loop across EEG time points
for t in tqdm(range(len(times))):


# =============================================================================
# Create the prediction accuracy null distribution
# =============================================================================
    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_hemi-'
        f'{args.hemisphere}_eeg_train_trials-{args.eeg_train_trials}_'
        f'eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights', file_name), allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_'][idx_roi]
    reg.intercept_ = reg_param['intercept_'][idx_roi]
    reg.n_features_in_ = reg_param['n_features_in_']
    del reg_param

    # Generate the t-fMRI responses for the test images with in vivo EEG
    tfmri = reg.predict(eeg_test[:,:,t])
    del reg

    # Center and normalize the t-fMRI responses
    tfmri_z = (tfmri - tfmri.mean(0)) /  (tfmri.std(0) + eps)

    # Loop across iterations
    for i in range(args.n_iter):

        # Permute the t-fMRI and fMRI test responses
        tfmri_perm = tfmri_z[idx_perm_pred[:,i]]
        fmri_perm = fmri_test_z[idx_perm[:,i]]

        # Correlate the t-fMRI test responses with the fMRI test responses
        correlation_roi_null_dist[:,t,i] = \
            np.diag(tfmri_perm.T @ fmri_perm) / len(tfmri_perm)

    # Delete unused variables
    del tfmri
    gc.collect()


# =============================================================================
# Save the results
# =============================================================================
# Create the save directory
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_accuracy_null_distribution')
os.makedirs(save_dir, exist_ok=True)

# Save the correlation scores
file_name = (f'corr_null_dist_fmri_sub-{args.fmri_subject:02d}_hemi-'
    f'{args.hemisphere}_roi-{args.roi}_eeg_train_trials-'
    f'{args.eeg_train_trials}.npy')
np.save(os.path.join(save_dir, file_name), correlation_roi_null_dist)