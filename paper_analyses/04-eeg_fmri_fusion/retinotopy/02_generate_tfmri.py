"""Generate the t-fMRI responses to the retinotopic mapping stimuli.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
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
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate t-fMRI <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the 515 NSD test images # !!!
# =============================================================================
# The test images consist of the 515 images that all NSD subjects saw for three
# times, and which were used to test BERG's encoding models

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the test image number
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=1
)
test_img_num = metadata['encoding_models']['test_img_num']

# Load the test images
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')
images = sdataset[test_img_num]
images = np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)


# =============================================================================
# Generate the in silico EEG responses # !!!
# =============================================================================
# Get the test image condition numbers
test_img_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'stimuli')
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()

# Loop across EEG subjects
eeg_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for s, sub in enumerate(tqdm(eeg_subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=sub
    )

    # Loop across image conditions
    for i, test_img in enumerate(tqdm(test_img_list)):

        # Get the probe image condition numbers
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

        # Generate the in silico EEG responses
        in_silico_eeg = berg.encode(model, probe_imgs)
        if i == 0:
            lh_response = in_silico_fmri[0]
            rh_response = in_silico_fmri[1]
        else:
            lh_response += in_silico_fmri[0]
            rh_response += in_silico_fmri[1]
        del in_silico_fmri, probe_imgs
        torch.cuda.empty_cache()
        gc.collect()



    # Generate and store the in silico EEG responses averaged across repeats
    eeg_sub = berg.encode(model, images, return_metadata=False)
    if s == 0:
        eeg = np.mean(eeg_sub, 1)
    else:
        eeg = np.append(eeg, np.mean(eeg_sub, 1), 1)

    # Delete unused variables
    del eeg_sub
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Z-score the in silico EEG responses and transform them with PCA # !!!
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
# Generate and save the t-fMRI responses # !!!
# =============================================================================
n_vertex = 163842
model_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights')
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'tfmri_responses',
    'nsd_515_test_images')
os.makedirs(save_dir, exist_ok=True)

# Empty t-fMRI response array of shape:
# (515 Images, 163,842 Vertices, 140 Time points)
tfmri = np.zeros((len(eeg), n_vertex, eeg.shape[2]), dtype=np.float32)

# Loop across EEG time points
for t in range(eeg.shape[2]):

    # Load the EEG-fMRI encoding fusion models weights
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
            f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(model_dir, file_name),
        allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_']
    reg.intercept_ = reg_param['intercept_']
    reg.n_features_in_ = reg_param['n_features_in_']

    # Generate the t-fMRI responses
    tfmri[:,:,t] = reg.predict(eeg[:,:,t])

# Save the in t-fMRI responses
file_name = f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}.h5'
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('tfmri', data=tfmri, dtype=np.float32)