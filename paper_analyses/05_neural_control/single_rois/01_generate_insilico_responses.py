"""Generate the in silico monkey electrophysiology responses for the 1.3M
ILSVRC-2012 train split images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : str
    The subject identifier for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
roi: str
    ROI for which the in silico responses are generated. Valid values are "V1",
        "V4", and "IT".
berg_dir : str
    Directory of the BERG.
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
from torch.utils.data import DataLoader
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico responses <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load BERG's encoding model and metadata
# =============================================================================
berg = BERG(berg_dir=args.berg_dir)

model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={'roi': [args.roi]}
    )

metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
)


# =============================================================================
# Access the ILSVRC-2012 train split
# =============================================================================
# Define the image transform
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224)),
    trn.Lambda(lambda img: np.transpose(img, (2, 0, 1))) # HWC to CHW
])

# Access the ILSVRC-2012 train split
images = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train',
    transform=transform)

# Create the dataloader
def collate_numpy(batch):
    imgs, labels = zip(*batch)
    imgs = np.stack(imgs, axis=0)
    labels = np.array(labels)
    return imgs, labels
loader = DataLoader(images, batch_size=1000, shuffle=False, num_workers=4,
    collate_fn=collate_numpy)
loader_iter = iter(loader)


# =============================================================================
# Generate the in silico responses
# =============================================================================
insilico_resp = []

for b in tqdm(range(len(loader_iter))):

    # Get the images
    imgs, _ = next(loader_iter)

    # Generate the in silico neural responses, and average them across
    # electrodes from the chosen ROI
    resp = berg.encode(model, imgs)
    resp = np.mean(resp, 1).astype(np.float32)
    insilico_resp.append(resp)

    # Delete unused variables
    del imgs, resp
    torch.cuda.empty_cache()
    gc.collect()

insilico_resp = np.concatenate(insilico_resp, axis=0)


# =============================================================================
# Save the in silico responses
# =============================================================================
data = {
    'responses': insilico_resp,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_control', 'insilico_responses',
    args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'insilico_responses_sub-{args.subject}_roi-{args.roi}.npy'

np.save(os.path.join(save_dir, file_name), data)