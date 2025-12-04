"""Generate the in silico EEG responses for 10k ILSVRC-2012 validation
images (the first 10 images of each category)using the Brain Encoding Response
Generator (BERG): https://github.com/gifale95/BERG.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subject : int
    The subject identifier for the EEG encoding models. Since the used
    encoidng models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
import torchvision
from torchvision import transforms as trn
from PIL import Image
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico EEG responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the channel names
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
    )
ch_names = metadata['eeg']['ch_names']

# Get the names of the selected channels
kept_ch_names = []
for c, ch_name in enumerate(ch_names):
    for ch_select in args.channels:
        if ch_select in ch_name:
            kept_ch_names.append(ch_name)
            break

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={'channels': kept_ch_names}
    )


# =============================================================================
# Read the ImageNet validation images
# =============================================================================
images = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')

# Get the indices of the first 10 images for each ILSVRC-2012 category
idx_img = []
for c in range(1000):
    idx_img += list(range(c*50, c*50+10))
idx_img = np.array(idx_img)


# =============================================================================
# Generate the in silico EEG responses to images
# =============================================================================
batches = 10
img_per_batch = int(np.ceil(len(idx_img) / batches))

for b in tqdm(range(batches)):

    img_batch = []
    idx_start = img_per_batch * b
    idx_end = img_per_batch * (b + 1)
    if b == batches - 1:
        idx_end = len(idx_img)

    for i in idx_img[idx_start:idx_end]:

        img, _ = images.__getitem__(i)
        transform = trn.Compose(
            [trn.CenterCrop(min(img.size)),
            trn.Resize((224, 224))
            ])
        img = np.asarray(transform(img))

        # Set the images to the correct format for encoding:
        # Must be a 4-D numpy array of shape
        # (Batch size x 3 RGB Channels x Width x Height) consisting of integer
        # values in the range [0, 255]. Furthermore, the images must be of square
        # size (i.e., equal width and height).
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        img_batch.append(img)

    # Generate the in silico EEG image responses
    img_batch = np.asarray(img_batch).astype(np.uint8)
    if b == 0:
        insilico_eeg = berg.encode(
            model,
            img_batch,
            return_metadata=False
            )
    else:
        insilico_eeg = np.append(insilico_eeg, berg.encode(
            model,
            img_batch,
            return_metadata=False
            ), axis=0)


# =============================================================================
# Save the in silico EEG responses
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'insilico_eeg')
os.makedirs(save_dir, exist_ok=True)

file_name = 'insilico_eeg_responses_sub-' + format(args.subject, '02') + \
    '_channels-' + '-'.join(args.channels) + '.h5'

# Save the h5py file
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('insilico_eeg_responses', data=insilico_eeg,
        dtype=np.float32)