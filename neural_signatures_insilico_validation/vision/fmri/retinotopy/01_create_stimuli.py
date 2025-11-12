"""Create the retinotopic mapping stimuli used to define polar angle and
eccentricity maps. One set of retinotopic mapping stimuli is created for each
of NSD's 515 shared images that all subjects viewed for 3 times during the NSD
experiment. These images were also used to test the encoding models.

Parameters
----------
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
IMG_SIZE : int
    The pixel resolution of each generated probe image (square). Determines the
    size of the stimulus fed into the encoding model.
BG_VALUE : float
    The background (baseline) pixel intensity value of the probe image.
nsd_dir : str
    Directory of the NSD.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py
from PIL import Image
from berg import BERG
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--IMG_SIZE', type=int, default=224)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Create retinotopy stimuli <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Get the encoding model image test image condition number
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Get the metadata for the NSD huze encoding model
metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=1)

# Get the test image condition number
test_img_cond = metadata['encoding_models']['test_img_num']


# =============================================================================
# Access the NSD stimulus images
# =============================================================================
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')

sdataset = sf.get('imgBrick')


# =============================================================================
# Define grid of probe centers (x,y) in degrees
# =============================================================================
coords = np.linspace(-args.FIELD_SIZE/2, args.FIELD_SIZE/2, args.GRID_RES)
xx, yy = np.meshgrid(coords, coords, indexing="xy")
centers = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape [n_probes, 2]
n_probes = len(centers)


# =============================================================================
# Image generation functions
# =============================================================================
# Define Gaussian mask generator
def make_gaussian_masks(center, sigma_deg=args.PROBE_SIGMA,
    field_size=args.FIELD_SIZE, img_size=args.IMG_SIZE):
    """Return a 2D Gaussian mask centered at (x_deg, y_deg) in visual
    coordinates."""
    x = np.linspace(-field_size/2, field_size/2, img_size)
    y = np.linspace(-field_size/2, field_size/2, img_size)
    X, Y = np.meshgrid(x, y, indexing="xy")
    x0, y0 = center
    mask = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma_deg**2))
    mask /= mask.max()
    return mask.astype(np.float32)

# Define probe generator using Gaussian-masked image patches
def make_gaussian_crop(mask, img_rgb):
    """
    Create an RGB probe by applying a Gaussian transparency mask centered at
    the specified visual field coordinates.
    """
    # Blend natural image with background color
    bg = np.ones_like(img_rgb) * args.BG_VALUE  # neutral background
    probe_rgb = img_rgb * mask + bg * (1 - mask)
    return probe_rgb.astype(np.float32)


# =============================================================================
# Generate the probe images (RGB Gaussian crops from natural images)
# =============================================================================
# Make the Gaussian masks
masks = np.array([make_gaussian_masks(center) for center in centers])
masks = masks[..., None]  # shape [n_probes, H, W, 1]

# Loop across the 515 NSD test images
for i, img in enumerate(tqdm(test_img_cond)):

    # Create the saving directory
    save_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'retinotopy', 'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+
        str(args.PROBE_SIGMA)+'_BG_VALUE-'+str(args.BG_VALUE), 'stimuli',
        'test_img-'+str(i).zfill(4))
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    # Load the image
    img_rgb = Image.fromarray(sdataset[img]).convert('RGB') # type: ignore
    img_rgb = img_rgb.resize((args.IMG_SIZE, args.IMG_SIZE))
    img_rgb = np.asarray(img_rgb).astype(np.float32) / 255.0  # normalize to [0,1]

    # Loop across all masks
    for m, mask in enumerate(masks):

        # Create the probe image
        probe_img = make_gaussian_crop(mask, img_rgb)
        probe_img = Image.fromarray((probe_img * 255).astype(np.uint8))

        # Save the probe image
        probe_fname = 'mask-' + str(m).zfill(5)+'.png'
        probe_img.save(os.path.join(save_dir, probe_fname))
        del probe_img

    del img_rgb


# =============================================================================
# Visualize a few probes
# =============================================================================
# plt.figure(figsize=(12, 3))
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(probes[i])
#     plt.axis('off')
# plt.suptitle("Example Gaussian crops from natural image")
# plt.show()
