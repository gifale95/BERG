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
from sklearn.linear_model import Ridge

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--eeg_subjects', default=[1, 2], type=list)
parser.add_argument('--eeg_reps', default='average', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate t-fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Only select vertices falling within the NSD visual streams
# =============================================================================
# Load the subject's metadata
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )

# Only select vertices falling within the NSD visual streams
n_vertex = 163842
idx_v = np.zeros(n_vertex, dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]


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
# Generate the in silico EEG image responses
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across EEG subjects
eeg = {}
for s, sub in enumerate(tqdm(args.eeg_subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(
        'eeg-things_eeg_2-vit_b_32',
        subject=sub
    )

    # Loop across image categories
    for key, val in images.items():

        # Generate and store the in silico EEG responses
        eeg[key] = berg.encode(model, val, return_metadata=False)

        # Delete unused variables
        torch.cuda.empty_cache()
        gc.collect()
    del model
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Z-score the in silico EEG responses and transform them with PCA
# =============================================================================
    if args.regression == 'linear':

        # Load the z-score and PCA parameters
        param_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'invivo_eeg_responses')
        file_name_scaler = (f'scaler_param_sub-{sub:02d}_'
            f'eeg_reps-{args.eeg_reps}.npy')
        file_name_pca = (f'pca_param_sub-{sub:02d}_'
            f'eeg_reps-{args.eeg_reps}.npy')
        scaler_param = np.load(os.path.join(param_dir, file_name_scaler),
            allow_pickle=True)
        pca_param = np.load(os.path.join(param_dir, file_name_pca),
            allow_pickle=True)

        # Loop across image categories
        for key in eeg.keys():

            # Loop across EEG time points
            n_time = eeg[key].shape[3]
            for t in range(n_time):

                # Z-score the EEG responses
                scaler = StandardScaler()
                scaler.scale_ = scaler_param[t]['scale_']
                scaler.mean_ = scaler_param[t]['mean_']
                scaler.var_ = scaler_param[t]['var_']
                scaler.n_features_in_ = scaler_param[t]['n_features_in_']
                scaler.n_samples_seen_ = scaler_param[t]['n_samples_seen_']
                for r in range(eeg[key].shape[1]):
                    eeg[key][:,r,:,t] = scaler.transform(eeg[key][:,r,:,t])

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
                for r in range(eeg[key].shape[1]):
                    eeg[key][:,r,:,t] = pca.transform(eeg[key][:,r,:,t])


# =============================================================================
# Generate and save the t-fMRI responses
# =============================================================================
    # Fusion model and save directories
    model_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'encoding_fusion_weights',
        f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
    save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion_variants',
        'hvc_selectivity', 'tfmri_responses',
        f'eeg_reps-{args.eeg_reps}_regression-{args.regression}')
    os.makedirs(save_dir, exist_ok=True)

    # Empty result dictionary
    tfmri = {}

    # Loop across EEG time points
    for t in tqdm(range(eeg[categories[0]].shape[2])):

        # Load the EEG-fMRI encoding fusion models weights
        file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
                    f'hemi-{args.hemisphere}_eeg_sub-{sub:02d}'
                    f'_eeg_time-{t:03d}.npy')
        reg_param = np.load(os.path.join(model_dir, file_name),
            allow_pickle=True).item()

        # Instantiate the fusion regression model
        if args.regression == 'linear':
            reg = LinearRegression()
        if args.regression == 'ridge':
            reg = Ridge()
        reg.coef_ = reg_param['coef_']
        reg.intercept_ = reg_param['intercept_']
        reg.n_features_in_ = reg_param['n_features_in_']

        # Loop across image categories
        for key, val in eeg.items():

            # Empty t-fMRI response array of shape:
            # (163,842 Vertices, 4 EEG repeats, 140 Time points)
            if t == 0:
                tfmri[key] = np.zeros((n_vertex, eeg[key].shape[1],
                    val.shape[2]), dtype=np.float32)

            # Generate the t-fMRI responses, and average them across images of the
            # same category
            for r in range(eeg[key].shape[1]):
                tfmri[key][idx_v,r,t] = np.mean(reg.predict(eeg[key][:,r,:,t]), 0)

    # Save the in t-fMRI responses
    file_name = f'tfmri_sub-{args.fmri_subject:02d}_hemi-{args.hemisphere}_eeg_sub-{sub:02d}.npy'
    np.save(os.path.join(save_dir, file_name), tfmri)