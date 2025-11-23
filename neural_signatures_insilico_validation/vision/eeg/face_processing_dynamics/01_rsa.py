"""Perfom RSA on in silico EEG responses to reveal the dynamics of identity,
gender, and age processing of faces (Dobs et al., 2019).

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subject : int
    The subject identifier for the EEG encoding models. Since the used
    encoidng models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : list
    List containing the EEG channel type(s) retained for the analyses.
    Possible values are: 'O' (occipital), 'P' (posterior), 'T' (temporal),
    'C' (central), 'F' (frontal). Alternatively, the list can also contain the
    names of the individual channels used.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default=['O', 'P'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the model metadata
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
    )
times = metadata['eeg']['times']

# EEG channel selection
ch_names = metadata['eeg']['ch_names']
kept_ch_names = []
for c in ch_names:
    for ch_select in args.channels:
        if ch_select in c:
            kept_ch_names.append(c)
            break

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={'channels': kept_ch_names}
    )


# =============================================================================
# Load the stimulus images
# =============================================================================
image_path = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'face_processing_dynamics', 'stimuli')

img_files = os.listdir(image_path)
img_files.sort()

images = []
for img_file in tqdm(img_files):
    img = Image.open(os.path.join(image_path, 'animate', img_file))
    img = np.asarray(img)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    images.append(img)

images = np.asarray(images)


# =============================================================================
# Generate the in silico EEG responses for the stimulus images
# =============================================================================
eeg = berg.encode(
    model,
    images,
    return_metadata=False
    )


# =============================================================================
# Pairwise decoding (exemplars) (~2h)
# =============================================================================
# Results array of shape:
# (Image conditions × Image conditions × EEG time points)
rdm_eeg = np.zeros((len(eeg), len(eeg), len(times)),
    dtype=np.float32)

# Loop over EEG time points and image-conditions
for t in tqdm(range(len(times))):
    for i1 in range(len(eeg)):
        for i2 in range(i1):

            # Select the image condition data
            eeg_cond_1 = eeg[i1,:,:,t] # type: ignore
            eeg_cond_2 = eeg[i2,:,:,t] # type: ignore

            # SVM target vectors
            y_train = np.zeros(((len(eeg_cond_1)-1)*2))
            y_train[int(len(y_train)/2):] = 1
            y_test = np.asarray((0, 1))
            scores = np.zeros(len(eeg_cond_1))

            # Loop across repeats (leave-one-repeat-out cross-decoding)
            for r in range(len(eeg_cond_1)):

                # Define the train/test partitions
                X_train = np.append(np.delete(eeg_cond_1, r, 0),
                    np.delete(eeg_cond_2, r, 0), 0)
                X_test = np.append(np.expand_dims(eeg_cond_1[r], 0),
                    np.expand_dims(eeg_cond_2[r], 0), 0)

                # Train the classifier
                dec_svm = SVC(kernel='linear')
                dec_svm.fit(X_train, y_train)

                # Test the classifier
                y_pred = dec_svm.predict(X_test)
                scores[r] = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            rdm_eeg[i1,i2,t] = np.mean(scores)
            rdm_eeg[i2,i1,t] = rdm_eeg[i1,i2,t]


# =============================================================================
# Create the stimulus RDMs
# =============================================================================
rdm_stimuli = {}

# Face identity RDM
identity = np.repeat(np.arange(16), 5)
rdm_identity = np.zeros((len(identity), len(identity)), dtype=np.int8)
for i1 in range(len(identity)):
    for i2 in range(i1):
        if identity[i1] != identity[i2]:
            rdm_identity[i1,i2] = 1
            rdm_identity[i2,i1] = rdm_identity[i1,i2]
rdm_stimuli['identity'] = rdm_identity

# Gender RDM
rdm_gender = np.zeros((len(img_files), len(img_files)), dtype=np.int8)
for i1 in range(len(img_files)):
    for i2 in range(i1):
        if img_files[i1][0] != img_files[i2][0]:
            rdm_gender[i1,i2] = 1
            rdm_gender[i2,i1] = rdm_gender[i1,i2]
rdm_stimuli['gender'] = rdm_gender

# Age RDM
rdm_age = np.zeros((len(img_files), len(img_files)), dtype=np.int8)
for i1 in range(len(img_files)):
    for i2 in range(i1):
        if img_files[i1][2] != img_files[i2][2]:
            rdm_age[i1,i2] = 1
            rdm_age[i2,i1] = rdm_age[i1,i2]
rdm_stimuli['age'] = rdm_age


# =============================================================================
# Create the DNN RDM # !!!
# =============================================================================
# VGG Face (see paper)

rdm_stimuli['dnn'] = rdm_dnn


# =============================================================================
# Compute the RSA scores between EEG and model RDMs
# =============================================================================
rsa = {}

# Loop across models
for key_target, val_target in tqdm(rdm_stimuli.items()):

    # Get the lower triangle of the model RDMs
    idx = np.tril_indices(len(val_target), -1)
    model = np.reshape(val_target[idx], (-1,1))
    model_control = []
    for key_other, val_other in rdm_stimuli.items():
        if key_other != key_target:
            model_control.append(val_other[idx])
    model_control = np.transpose(np.array(model_control))

    # Regress out the model variance explained by the control models
    reg_model = LinearRegression().fit(model_control, model)
    model_res = model - reg_model.predict(model_control)

    # Get the lower triangle of the EEG RDM
    eeg = rdm_eeg[idx]

    # Regress out the EEG variance explained by the control models
    reg_eeg = LinearRegression().fit(model_control, eeg)
    eeg_res = eeg - reg_eeg.predict(model_control)

    # Compute the timewise partial correlation between EEG and model RDMs
    rsa_model = np.zeros(times)
    for t in range(len(times)):
        rsa_model[t] = spearmanr(eeg_res[:,t], model_res)[0]
    rsa[key_target] = rsa_model
    del rsa_model


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rdm_eeg': rdm_eeg,
    'rdm_stimuli': rdm_stimuli,
    'rsa': rsa,
    'times': times,
    'kept_ch_names': kept_ch_names
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'face_processing_dynamics', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_channels-' + \
    ''.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore