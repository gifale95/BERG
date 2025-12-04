"""Perform exemplar and animacy pairwise decoding on in silico EEG responses
for 200 ImageNet images (100 animate categories, and 100 inanimate categories).

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
import torchvision
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default=['O', 'P'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Pairwise decoding <<<')
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
images = []
image_path = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'object_exemplar_animacy_categorization', 'stimuli')

# Animate images
animate_imgs = os.listdir(os.path.join(image_path, 'animate'))
animate_imgs.sort()
for img_file in tqdm(animate_imgs):
    img = Image.open(os.path.join(image_path, 'animate', img_file))
    img = np.asarray(img)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    images.append(img)

# Inanimate images
inanimate_imgs = os.listdir(os.path.join(image_path, 'inanimate'))
inanimate_imgs.sort()
for img_file in tqdm(inanimate_imgs):
    img = Image.open(os.path.join(image_path, 'inanimate', img_file))
    img = np.asarray(img)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    images.append(img)

# Convert to numpy array
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
decoding_exemplars = np.zeros((len(eeg), len(eeg), len(times)),
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
            decoding_exemplars[i1,i2,t] = np.mean(scores)
            decoding_exemplars[i2,i1,t] = decoding_exemplars[i1,i2,t]


# =============================================================================
# Pairwise decoding (animacy)
# =============================================================================
# Select the used animate conditions
animate_conditions = np.arange(100)

# Select the used inanimate conditions
inanimate_conditions = np.arange(100, 200)

# Results array of shape: (EEG time points)
decoding_animacy = np.zeros((len(times)), dtype=np.float32)

# Loop over EEG time points
for t in tqdm(range(len(times))):

    # Select the image condition data
    eeg_cond_1 = eeg[animate_conditions,:,:,t] # type: ignore
    eeg_cond_2 = eeg[inanimate_conditions,:,:,t] # type: ignore

    # Loop over EEG repeats (to avoid false positives, the SVMs should be
    # trained and tested on independent in silico EEG response repeats, that
    # is, in silico EEG repeats generated using different encoding models)
    scores = []
    for r in range(eeg_cond_1.shape[1]):

        # Define the train partitions
        eeg_cond_1_train = np.delete(eeg_cond_1, r, 1)
        eeg_cond_2_train = np.delete(eeg_cond_2, r, 1)
        eeg_cond_1_train = np.reshape(eeg_cond_1_train, (-1, len(kept_ch_names)))
        eeg_cond_2_train = np.reshape(eeg_cond_2_train, (-1, len(kept_ch_names)))
        X_train = np.append(eeg_cond_1_train, eeg_cond_2_train, 0)

        # Define the test partitions
        X_test = np.append(eeg_cond_1[:,r], eeg_cond_2[:,r], 0)

        # SVM target vectors
        y_train = np.zeros(len(X_train))
        y_train[int(len(X_train)/2):] = 1
        y_test = np.zeros(len(X_test))
        y_test[int(len(X_test)/2):] = 1

        # Train the classifier
        dec_svm = SVC(kernel='linear')
        dec_svm.fit(X_train, y_train)

        # Test the classifier
        y_pred = dec_svm.predict(X_test)
        scores.append(sum(y_pred == y_test) / len(y_test))

    # Store the accuracy
    decoding_animacy[t] = np.mean(scores)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'decoding_exemplars': decoding_exemplars,
    'decoding_animacy': decoding_animacy,
    'times': times,
    'kept_ch_names': kept_ch_names
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_exemplar_animacy_categorization',
    'pairwise_decoding_results')
os.makedirs(save_dir, exist_ok=True)

file_name = 'pairwise_decoding_sub-' + format(args.subject, '02') + \
    '_channels-' + ''.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results) # type: ignore