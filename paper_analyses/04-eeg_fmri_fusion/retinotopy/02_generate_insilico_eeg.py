"""Generate the in silico EEG responses for the retinotopic mapping stimuli,
later used to generate the t-fMRI responses.

Parameters
----------
test_img : int
    Number of the test image to generate the in silico EEG for. Valid values
    are integers from 0 to 99, corresponding to 100 of the shared images viewed
    by all NSD subjects.
FIELD_SIZE : float
    The total width and height of the simulated visual field in degrees of
    visual angle. The coordinate system spans from -FIELD_SIZE/2 to
    +FIELD_SIZE/2 in both x and y directions.
GRID_RES : int
    The number of probe centers sampled per axis (x and y). The total number of
    probes will be GRID_RES × GRID_RES.
PROBE_SIGMA : float
    The standard deviation of each 2D Gaussian probe in degrees of visual
    angle. Controls the probe size in the visual field.
BG_VALUE : float
    The background (baseline) pixel intensity value of the probe image.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from PIL import Image
from tqdm import tqdm
import h5py
import gc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--test_img', type=int, default=0)
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico EEG <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the images
# =============================================================================
# Get the probe image file names
test_img_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'retinotopy',
'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
'_BG_VALUE-'+str(args.BG_VALUE), 'stimuli')
test_img_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'stimuli') # !!! DELETE
test_img = f'test_img-{args.test_img:04d}'
probe_img_list = os.listdir(os.path.join(test_img_dir, test_img))
probe_img_list.sort()

# Load the probe images into a numpy array using PIL
probe_imgs = []
for probe_img in probe_img_list:
    img = Image.open(os.path.join(test_img_dir, test_img, probe_img))
    img = np.array(img)
    probe_imgs.append(img)
probe_imgs = np.array(probe_imgs)
probe_imgs = np.swapaxes(probe_imgs, 1, 3)  # BHWC to BCHW


# =============================================================================
# Generate the in silico EEG responses
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across EEG subjects
eeg_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for s, sub in enumerate(tqdm(eeg_subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=sub
    )

    # Generate the in silico EEG responses, and average them across repeats
    eeg_sub = np.mean(berg.encode(model, probe_imgs, return_metadata=False),
        1).astype(np.float32)

    # Store the in silico EEG responses
    if s == 0:
        eeg = eeg_sub
    else:
        eeg = np.append(eeg, eeg_sub, 1)

    # Remove unused variables
    del eeg_sub
    torch.cuda.empty_cache()
    gc.collect()

del probe_imgs


# =============================================================================
# Z-score the in silico EEG responses and transform them with PCA
# =============================================================================
# Load the z-score and PCA parameters
param_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_eeg_responses')
scaler_param = np.load(os.path.join(param_dir, 'scaler_param.npy'),
    allow_pickle=True)
pca_param = np.load(os.path.join(param_dir, 'pca_param.npy'),
    allow_pickle=True)

# Loop across EEG time points
for t in range(eeg.shape[2]):

    # Z-score the EEG responses
    scaler = StandardScaler()
    scaler.scale_ = scaler_param[t]['scale_']
    scaler.mean_ = scaler_param[t]['mean_']
    scaler.var_ = scaler_param[t]['var_']
    scaler.n_features_in_ = scaler_param[t]['n_features_in_']
    scaler.n_samples_seen_ = scaler_param[t]['n_samples_seen_']
    eeg[:,:,t] = scaler.transform(eeg[:,:,t])

    # Transform the EEG responses with PCA
    pca = PCA(n_components=eeg.shape[1], random_state=20200220)
    pca.components_ = pca_param[t]['components_']
    pca.explained_variance_ = pca_param[t]['explained_variance_']
    pca.explained_variance_ratio_ = pca_param[t]['explained_variance_ratio_']
    pca.singular_values_ = pca_param[t]['singular_values_']
    pca.mean_ = pca_param[t]['mean_']
    pca.n_components_ = pca_param[t]['n_components_']
    pca.n_samples_ = pca_param[t]['n_samples_']
    pca.noise_variance_ = pca_param[t]['noise_variance_']
    pca.n_features_in_ = pca_param[t]['n_features_in_']
    eeg[:,:,t] = pca.transform(eeg[:,:,t])


# =============================================================================
# Save the in silico EEG responses
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'insilico_eeg')
os.makedirs(save_dir, exist_ok=True)

file_name = f'eeg_test_img-{args.test_img:04d}.h5'
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('eeg', data=eeg, dtype=np.float32)