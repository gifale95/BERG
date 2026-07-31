"""Generate the t-fMRI responses to face, body, scene, and object images.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import numpy as np
import os
from PIL import Image
import torch
from berg import BERG
from tqdm import tqdm
import gc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate t-fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus images
# =============================================================================
# Image directories
img_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'hvc_selectivity',
    'stimuli')
categories = ['Bodies', 'Faces', 'Objects', 'Scenes']
img_type = ['Sel', 'Test']

# Loop across image categories and types
images = {}
for cat in tqdm(categories):
    img_cat = []
    for itype in img_type:
        # Load the images
        img_list = os.listdir(os.path.join(img_dir, cat+'-'+itype))
        img_list.sort()
        for img_name in img_list:
            img_path = os.path.join(img_dir, cat+'-'+itype, img_name)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img_cat.append(img)
    img_cat = np.array(img_cat)
    img_cat = np.swapaxes(img_cat, 1, 3)  # BHWC to BCHW
    images[cat] = img_cat
    del img_cat


# =============================================================================
# Generate the in silico EEG image responses (of all 10 THINGS EEG2 subjects)
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across EEG subjects
eeg = {}
eeg_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for s, sub in enumerate(tqdm(eeg_subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=sub
    )

    # Loop across image categories
    for key, val in images.items():

        # Generate and store the in silico EEG responses averaged across repeats
        eeg_sub = berg.encode(model, val, return_metadata=False)
        if s == 0:
            eeg[key] = np.mean(eeg_sub, 1)
        else:
            eeg[key] = np.append(eeg[key], np.mean(eeg_sub, 1), 1)

        # Delete unused variables
        del eeg_sub
        torch.cuda.empty_cache()
        gc.collect()
    del model
    torch.cuda.empty_cache()
    gc.collect()


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

# Loop across image categories
for key, val in eeg.items():

    # Loop across EEG time points
    n_time = val.shape[2]
    for t in range(n_time):

        # Z-score the EEG responses
        scaler = StandardScaler()
        scaler.scale_ = scaler_param[t]['scale_']
        scaler.mean_ = scaler_param[t]['mean_']
        scaler.var_ = scaler_param[t]['var_']
        scaler.n_features_in_ = scaler_param[t]['n_features_in_']
        scaler.n_samples_seen_ = scaler_param[t]['n_samples_seen_']
        eeg[key][:,:,t] = scaler.transform(eeg[key][:,:,t])

        # Transform the EEG responses with PCA
        pca = PCA(n_components=eeg[key].shape[1], random_state=20200220)
        pca.components_ = pca_param[t]['components_']
        pca.explained_variance_ = pca_param[t]['explained_variance_']
        pca.explained_variance_ratio_ = pca_param[t]['explained_variance_ratio_']
        pca.singular_values_ = pca_param[t]['singular_values_']
        pca.mean_ = pca_param[t]['mean_']
        pca.n_components_ = pca_param[t]['n_components_']
        pca.n_samples_ = pca_param[t]['n_samples_']
        pca.noise_variance_ = pca_param[t]['noise_variance_']
        pca.n_features_in_ = pca_param[t]['n_features_in_']
        eeg[key][:,:,t] = pca.transform(eeg[key][:,:,t])


# =============================================================================
# Generate and save the t-fMRI responses
# =============================================================================
# Fusion model and save directories
model_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights')
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'hvc_selectivity',
    'tfmri_responses')
os.makedirs(save_dir, exist_ok=True)

# Empty result dictionary
tfmri = {}
n_vertex = 163842

# Loop across EEG time points
for t in tqdm(range(eeg[categories[0]].shape[2])):

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

    # Loop across image categories
    for key, val in eeg.items():

        # Empty t-fMRI response array of shape:
        # (163,842 Vertices, 140 Time points)
        if t == 0:
            tfmri[key] = np.zeros((n_vertex, val.shape[2]), dtype=np.float32)

        # Generate the t-fMRI responses, and average them across images of the
        # same category
        tfmri[key][:,t] = np.mean(reg.predict(val[:,:,t]), 0)

# Save the in t-fMRI responses
file_name = f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}.npy'
np.save(os.path.join(save_dir, file_name), tfmri)