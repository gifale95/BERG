"""Perform category and animacy decoding on in silico EEG responses for 200
ImageNet images (100 animate categories, and 100 inanimate categories).

Parameters
----------
subject : int
    Used subject.
channels : str
    Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
    central ['C'], frontal ['F'], occipital/parital ['OP'], or all ['all']
    channels.
nest_dir : str
    Neural encoding simulation toolkit directory.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from nest.nest import NEST
from sklearn.svm import SVC
from nltk.corpus import wordnet as wn
import torchvision
from torchvision import transforms as trn


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parser.add_argument('--channels', default='OP', type=str) # ['O', 'P', 'T', 'C', 'F', 'OP', 'all']
#parser.add_argument('--nest_dir', default='/home/ale/aaa_stuff/PhD/projects/neural_encoding_simulation_toolkit', type=str)
#parser.add_argument('--imagenet_dir', default='/home/ale/Downloads/imagenet_val', type=str)
#parser.add_argument('--nest_dir', default='/home/ale/scratch/projects/neural_encoding_simulation_toolkit', type=str)
parser.add_argument('--nest_dir', default='/scratch/giffordale95/projects/neural_encoding_simulation_toolkit', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012/', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding | In silico EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Define the 200 ImageNet images used
# =============================================================================
# # List the imagenet synsets
# synsets = os.listdir(os.path.join(args.imagenet_dir, 'val'))
# synsets.sort()

# # List the categories
# categories = [
# 	'animal.n.01', # 398 (* 50 = 19,900)
# 	'food.n.01', # 26 (* 50 = 1,300)
# 	'device.n.01', # 130 (* 50 = 6,500)
# 	'geological_formation.n.01' # 10 (* 50 = 500)
# 	]

# # Categorize each synset
# synset_categories = np.zeros((len(synsets), len(categories)), dtype=np.int16)
# for s, synset in enumerate(tqdm(synsets)):
# 	synset_name = wn.synset_from_pos_and_offset('n', int(synset[1:]))
# 	synset_name = synset_name.name()
# 	synset_name = wn.synset(synset_name)
# 	synset_cat = synset_name.hypernym_paths()
# 	for c, category in enumerate(categories):
# 		category_name = wn.synset(category)
# 		for sc in synset_cat:
# 			if np.isin(category_name, sc):
# 				synset_categories[s,c] = 1

# # Get the indices of 100 animate categories ('animal.n.01')
# idx_animate = np.where(synset_categories[:,0])[0][:100]

# # Get the indices of 100 inanimate categories ('device.n.01')
# idx_inanimate = np.where(synset_categories[:,2])[0][:100]

# # Multiply the indices by 50, since the ILSVRC-2012 validation split has 50
# # images per category (and we will only use one image per each of the 200
# # cateogries)
# idx_animate *= 50
# idx_inanimate *= 50
# idx_all = np.append(idx_animate, idx_inanimate)

# # Access the ILSVRC-2012 validation split
# dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')


# =============================================================================
# Get the image paths
# =============================================================================
image_path = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'pairwise_decoding_eeg', 'images')

animate_imgs = os.listdir(os.path.join(image_path, 'animate'))
animate_imgs.sort()

inanimate_imgs = os.listdir(os.path.join(image_path, 'inanimate'))
inanimate_imgs.sort()


# =============================================================================
# Generate the in silico fMRI responses for the 200 test images
# =============================================================================
# Create the NEST object
nest_object = NEST(args.nest_dir)

# Load the EEG encoding model
encoding_model = nest_object.get_encoding_model(
    modality='eeg', # required
    train_dataset='things_eeg_2', # required
    model='vit_b_32', # required
    subject=args.subject, # required
    roi=None, # default is None, only required if modality=='fmri'
    device='auto' # default is 'auto'
    )

# Encode EEG responses to images
eeg = []
# Animate images
for i in tqdm(animate_imgs):
    # Preprocess the image
    img = Image.open(os.path.join(image_path, 'animate', i))
    transform = trn.Compose([trn.CenterCrop(min(img.size))])
    img = np.asarray(transform(img))
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    img = np.expand_dims(img, 0)
    # Generate the in silico EEG response to the image
    eeg_img, metadata = nest_object.encode(
        encoding_model, # required
        img, # required
        return_metadata=True, # default is True
        device='auto' # default is 'auto'
        )
    eeg.append(np.squeeze(eeg_img))
    times = metadata['eeg']['times']
    ch_names = metadata['eeg']['ch_names']
# Inanimate images
for i in tqdm(inanimate_imgs):
    # Preprocess the image
    img = Image.open(os.path.join(image_path, 'inanimate', i))
    transform = trn.Compose([trn.CenterCrop(min(img.size))])
    img = np.asarray(transform(img))
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    img = np.expand_dims(img, 0)
    # Generate the in silico EEG response to the image
    eeg_img, metadata = nest_object.encode(
        encoding_model, # required
        img, # required
        return_metadata=True, # default is True
        device='auto' # default is 'auto'
        )
    eeg.append(np.squeeze(eeg_img))
    times = metadata['eeg']['times']
    ch_names = metadata['eeg']['ch_names']
eeg = np.asarray(eeg)


# =============================================================================
# Channels selection
# =============================================================================
# Retain the EEG channels of the chosen channel type
if args.channels != 'OP' and args.channels != 'all':
    kept_ch_names = []
    idx_ch = []
    for c, chan in enumerate(ch_names):
        if args.channels in chan:
            kept_ch_names.append(chan)
            idx_ch.append(c)
    idx_ch = np.asarray(idx_ch)
    eeg = eeg[:,:,idx_ch]
    ch_names = kept_ch_names
elif args.channels == 'OP':
    kept_ch_names = []
    idx_ch = []
    for c, chan in enumerate(ch_names):
        if 'O' in chan or 'P' in chan:
            kept_ch_names.append(chan)
            idx_ch.append(c)
    idx_ch = np.asarray(idx_ch)
    eeg = eeg[:,:,idx_ch]
    ch_names = kept_ch_names


# =============================================================================
# Pairwise decoding (exemplars) (~2h)
# =============================================================================
# Results array of shape:
# (Image conditions × Image conditions × EEG time points)
pairwise_decoding_exemplars = np.zeros((len(eeg), len(eeg),
    eeg.shape[3]), dtype=np.float32)

# Loop over EEG time points and image-conditions
for t in tqdm(range(eeg.shape[3])):
    for i1 in range(len(eeg)):
        for i2 in range(i1):

            # Select the image condition data
            eeg_cond_1 = eeg[i1,:,:,t]
            eeg_cond_2 = eeg[i2,:,:,t]

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
            pairwise_decoding_exemplars[i1,i2,t] = np.mean(scores)
            pairwise_decoding_exemplars[i2,i1,t] = \
                pairwise_decoding_exemplars[i1,i2,t]


# =============================================================================
# Pairwise decoding (animacy)
# =============================================================================
# Select the used animate conditions
animate_conditions = np.arange(100)

# Select the used inanimate conditions
inanimate_conditions = np.arange(100, 200)

# Results array of shape: (EEG time points)
pairwise_decoding_animacy = np.zeros((eeg.shape[3]), dtype=np.float32)

# Loop over EEG time points
for t in tqdm(range(eeg.shape[3])):

    # Select the image condition data
    eeg_cond_1 = eeg[animate_conditions,:,:,t]
    eeg_cond_2 = eeg[inanimate_conditions,:,:,t]

    # Loop over image conditions and repeats (to avoid false positives, the SVMs
    # should be trained and tested on independent in silico EEG response
    # repeats, that is, in silico EEG repeats generated using different encoding
    # models)
    scores = []
    for r in range(eeg_cond_1.shape[1]):

        # Define the train partitions
        eeg_cond_1_train = np.delete(eeg_cond_1, r, 1)
        eeg_cond_1_train = np.reshape(eeg_cond_1_train, (-1, len(ch_names)))
        eeg_cond_2_train = np.delete(eeg_cond_2, r, 1)
        eeg_cond_2_train = np.reshape(eeg_cond_2_train, (-1, len(ch_names)))
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
    pairwise_decoding_animacy[t] = np.mean(scores)


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
    'args': args,
    'pairwise_decoding_exemplars': pairwise_decoding_exemplars,
    'pairwise_decoding_animacy': pairwise_decoding_animacy,
    'times': times,
    'ch_names': ch_names
}

save_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'pairwise_decoding_eeg', 'sub-'+format(args.subject,'02'), 'channels-'+
    args.channels)
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

file_name = 'pairwise_decoding_in_silico_sub-' + format(args.subject,'02') + \
    '_channels-' + args.channels + '.npy'

np.save(os.path.join(save_dir, file_name), results_dict)