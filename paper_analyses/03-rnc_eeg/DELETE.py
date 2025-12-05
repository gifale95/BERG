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
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Generate images
images = np.random.randint(0, 255, (10, 3, 224, 224)).astype(np.uint8)


# =============================================================================
# THINGS fMRI1
# =============================================================================
# Load the metadata
metadata = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=1
    )

# Generate responses for all voxels
model_all = berg.get_encoding_model(
    'fmri-things_fmri_1-vit_b_32',
    subject=1
    )
fmri_all = berg.encode(model_all, images, return_metadata=False)

# Generate responses for V1 voxels
model_v1 = berg.get_encoding_model(
    'fmri-things_fmri_1-vit_b_32',
    subject=1,
    selection={'roi': ['V1']}
    )
fmri_v1 = berg.encode(model_v1, images, return_metadata=False)

# Generate responses for the first 100 voxels
voxel_index = np.zeros((211339), dtype=int)
voxel_index[:100] = 1
model_100 = berg.get_encoding_model(
    'fmri-things_fmri_1-vit_b_32',
    subject=1,
    selection={'voxel_index': voxel_index}
    )
fmri_100 = berg.encode(model_100, images, return_metadata=False)

# Test V1 voxels
idx_v1 =  metadata['fmri']['roi']['V1']
print(all(fmri_all[idx_v1] == fmri_v1))

# Test first 100 voxels
print(all(fmri_all[:100] == fmri_100))


# =============================================================================
# THINGS MEG1
# =============================================================================





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
# Access the ImageNet validation images
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