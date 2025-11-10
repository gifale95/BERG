"""Generate the retinotopic mapping stimuli used to define polar angle and
eccentricity maps.

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
N_SPLITS : int
    Number of splits for reliability estimation. The probe set is randomly
    split into this many subsets, and separate retinotopy estimates are
    computed for each.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5) # !!! Try smaller/larger values!
parser.add_argument('--IMG_SIZE', type=int, default=224)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--N_SPLITS', type=int, default=2)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Create retinotopy stimuli <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define grid of probe centers (x,y) in degrees
# =============================================================================
coords = np.linspace(-args.FIELD_SIZE/2, args.FIELD_SIZE/2, args.GRID_RES)
xx, yy = np.meshgrid(coords, coords, indexing="xy")
centers = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape [n_probes, 2]
n_probes = len(centers)


# =============================================================================
# Generate probe images (RGB Gaussian crops from a natural image)
# =============================================================================
# !!! Convert images to RGB, and replace white probes with naturalistic stimuli.

# --- INPUT: naturalistic image ---
# Replace this path with your own image (RGB)
image_path = "example_natural_image.jpg"
img_rgb = Image.open(image_path).convert("RGB")
img_rgb = img_rgb.resize((args.IMG_SIZE, args.IMG_SIZE))
img_rgb = np.asarray(img_rgb).astype(np.float32) / 255.0  # normalize to [0,1]

# Define Gaussian mask generator
def make_gaussian_mask(center, sigma_deg=args.PROBE_SIGMA,
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
def make_gaussian_crop(center):
    """
    Create an RGB probe by applying a Gaussian transparency mask centered at
    the specified visual field coordinates.
    """
    mask = make_gaussian_mask(center)
    mask = mask[..., None]  # shape [H,W,1]
    # Blend natural image with background color
    bg = np.ones_like(img_rgb) * args.BG_VALUE  # neutral background
    probe_rgb = img_rgb * mask + bg * (1 - mask)
    return probe_rgb.astype(np.float32)

# Generate all probes
probes = np.stack([make_gaussian_crop(c) for c in centers], axis=0)  # shape [n_probes, H, W, 3]
print(f"Generated {probes.shape[0]} RGB probes of size {args.IMG_SIZE}×{args.IMG_SIZE}")


# =============================================================================
# Optional: visualize a few probes # !!!
# =============================================================================
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(probes[i])
    plt.axis('off')
plt.suptitle("Example Gaussian crops from natural image")
plt.show()


# =============================================================================
# Save the probe stimuli # !!!
# =============================================================================