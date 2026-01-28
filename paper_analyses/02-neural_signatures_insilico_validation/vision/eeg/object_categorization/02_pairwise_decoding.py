"""Perform exemplar, object category, and animacy pairwise decoding on in
silico EEG responses for 400 ImageNet images (40 objects categories, with 10
exemplars per category; 20 categories are animate, and 20 inanimate).

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subject : int
    The subject identifier for the EEG encoding models.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
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
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
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
    'object_categorization', 'stimuli')

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
eeg = berg.encode(model, images)


# =============================================================================
# Exemplar decoding (~7h)
# =============================================================================
# Results array of shape:
# (Images × Images × EEG time points)
object_cats = 40
exemplars_per_cat = 10
decoding_exemplars = np.zeros((object_cats, len(times)), dtype=np.float32)

# Loop over EEG time points and object categories
for t in tqdm(range(len(times))):
    for c in range(object_cats):

        # Select the data of object category
        idx_c_start = c * exemplars_per_cat
        idx_c_end = idx_c_start + exemplars_per_cat
        eeg_cond = eeg[idx_c_start:idx_c_end,:,:,t]

        # Loop over object exemplars
        scores = []
        for e1 in range(len(eeg_cond)):

            # Select the data of the first image exemplar
            eeg_cond_1 = eeg_cond[e1]

            for e2 in range(e1):

                # Select the data of the second image exemplar
                eeg_cond_2 = eeg_cond[e2]

                # SVM target vectors
                y_train = np.zeros(((len(eeg_cond_1)-1)*2))
                y_train[int(len(y_train)/2):] = 1
                y_test = np.asarray((0, 1))

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
                    scores.append(sum(y_pred == y_test) / len(y_test))

        # Store the accuracy
        decoding_exemplars[c,t] = np.mean(scores)
        decoding_exemplars[c,t] = decoding_exemplars[c,t]


# =============================================================================
# Object decoding (~2h)
# =============================================================================
# Results array of shape:
# (Object categories × Object categories × EEG time points)
decoding_objects = np.zeros((object_cats, object_cats, len(times)),
    dtype=np.float32)

# Loop over EEG time points and object categories
for t in tqdm(range(len(times))):
    for c1 in range(object_cats):

        # Select the data of the first object category
        idx_c1_start = c1 * exemplars_per_cat
        idx_c1_end = idx_c1_start + exemplars_per_cat
        eeg_cond_1 = eeg[idx_c1_start:idx_c1_end,:,:,t]

        for c2 in range(c1):

            # Select the data of the second object category
            idx_c2_start = c2 * exemplars_per_cat
            idx_c2_end = idx_c2_start + exemplars_per_cat
            eeg_cond_2 = eeg[idx_c2_start:idx_c2_end,:,:,t]

            # Loop over object exemplars (to mitigate the risk of the
            # classifiers exploiting low-level visual information, the
            # classifiers are trained and tested on idendependent object
            # exemplars)
            scores = []
            for e in range(len(eeg_cond_1)):

                # Define the train exemplars
                eeg_cond_1_train_ex = np.delete(eeg_cond_1, e, 0)
                eeg_cond_2_train_ex = np.delete(eeg_cond_2, e, 0)

                # Loop over EEG repeats (to reduce false positives, the SVMs
                # should be trained and tested on independent in silico EEG
                # response repeats, that is, in silico EEG repeats generated
                # using different encoding models)
                for r in range(eeg_cond_1.shape[1]):

                    # Define the train repeats
                    eeg_cond_1_train = np.delete(eeg_cond_1_train_ex, r, 1)
                    eeg_cond_2_train = np.delete(eeg_cond_2_train_ex, r, 1)
                    eeg_cond_1_train = np.reshape(eeg_cond_1_train,
                        (-1, len(kept_ch_names)))
                    eeg_cond_2_train = np.reshape(eeg_cond_2_train,
                        (-1, len(kept_ch_names)))
                    X_train = np.append(eeg_cond_1_train, eeg_cond_2_train, 0)

                    # Define the test partition
                    X_test = np.append(np.expand_dims(eeg_cond_1[e,r], 0),
                        np.expand_dims(eeg_cond_2[e,r], 0), 0)

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
            decoding_objects[c1,c2,t] = np.mean(scores)
            decoding_objects[c2,c1,t] = decoding_objects[c1,c2,t]


# =============================================================================
# Animacy decoding (~6h)
# =============================================================================
# Select the used animate conditions
animate_conditions = np.arange(len(animate_imgs))

# Select the used inanimate conditions
inanimate_conditions = np.arange(len(animate_imgs),
    len(animate_imgs)+len(inanimate_imgs))

# Results array of shape: (EEG time points)
decoding_animacy = np.zeros((len(times)), dtype=np.float32)

# Loop over EEG time points
for t in tqdm(range(len(times))):

    # Select the animacy data
    eeg_animate = eeg[animate_conditions,:,:,t]
    eeg_inanimate = eeg[inanimate_conditions,:,:,t]

    # Loop over the animate object categories (to mitigate the risk of the
    # classifiers exploiting low-level visual information, the classifiers are
    # trained and tested on idendependent object categories)
    scores = []
    for c1 in tqdm(range(object_cats//2)): # Half of the total objects are animate

        # Define the animate train object categories
        idx_c1_start = c1 * exemplars_per_cat
        idx_c1_end = idx_c1_start + exemplars_per_cat
        eeg_animate_train_cat = np.delete(
            eeg_animate, np.arange(idx_c1_start, idx_c1_end), 0)

        # Looop over the inanimate object categories
        for c2 in range(object_cats//2): # Half of the total objects are inanimate

            # Define the inanimate train object categories
            idx_c2_start = c2 * exemplars_per_cat
            idx_c2_end = idx_c2_start + exemplars_per_cat
            eeg_inanimate_train_cat = np.delete(
                eeg_inanimate, np.arange(idx_c2_start, idx_c2_end), 0)

            # Loop over EEG repeats (to reduce false positives, the SVMs should
            # be trained and tested on independent in silico EEG response
            # repeats, that is, in silico EEG repeats generated using different
            # encoding models)
            for r in range(eeg_animate.shape[1]):

                # Define the train repeats
                eeg_animate_train = np.delete(eeg_animate_train_cat, r, 1)
                eeg_inanimate_train = np.delete(eeg_inanimate_train_cat, r, 1)
                eeg_animate_train = np.reshape(eeg_animate_train,
                    (-1, len(kept_ch_names)))
                eeg_inanimate_train = np.reshape(eeg_inanimate_train,
                    (-1, len(kept_ch_names)))
                X_train = np.append(eeg_animate_train, eeg_inanimate_train, 0)

                # Define the test partition
                X_test = np.append(
                    eeg_animate[idx_c1_start:idx_c1_end,r],
                    eeg_inanimate[idx_c2_start:idx_c2_end,r], 0)

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
# Perform MDS on the EEG responses of each time point
# =============================================================================
# Empty results array of shape (Images, 2 MDS dimensions, Times)
n_components = 2
eeg_mds = np.zeros((len(eeg), n_components, len(times)), dtype=np.float32)

# Loop across time point
for t in tqdm(range(len(times))):

    # Perform MDS
    embedding = MDS(n_components=n_components, n_init=10, max_iter=1000,
        random_state=20200220)
    eeg_mds[:,:,t] = embedding.fit_transform(np.mean(eeg[:,:,:,t], 1))


# =============================================================================
# Save the results
# =============================================================================
results = {
    'eeg': eeg,
    'eeg_mds': eeg_mds,
    'decoding_exemplars': decoding_exemplars,
    'decoding_objects': decoding_objects,
    'decoding_animacy': decoding_animacy,
    'times': times,
    'kept_ch_names': kept_ch_names
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'object_categorization', 'pairwise_decoding',
    args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'pairwise_decoding_sub-' + format(args.subject, '02') + \
    '_channels-' + '-'.join(args.channels) + '.npy'

np.save(os.path.join(save_dir, file_name), results)