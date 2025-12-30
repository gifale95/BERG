"""Use BERG to generate the in silico fMRI responses to the retinotopic mapping
stimuli. Then, for each vertex estimate the retinotopic maps (polar angle and
eccentricity) from the in silico fMRI responses.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in surface space.
subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
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
BG_VALUE : float
    The background (baseline) pixel intensity value of the probe image.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from PIL import Image
import gc
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Estimate retinotopic maps <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the encoding model
model = berg.get_encoding_model(args.encoding_model, subject=args.subject)


# =============================================================================
# Generate the in silico fMRI responses using BERG
# =============================================================================
# Get the test image condition numbers
test_img_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'stimuli')
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()

# Loop across test images
for i, test_img in enumerate(tqdm(test_img_list)):

    # Get the probe image condition numbers
    probe_img_list = os.listdir(os.path.join(test_img_dir, test_img))
    probe_img_list.sort()

    # Load the probe images into a numpy array using PIL
    probe_imgs = []
    for probe_img in probe_img_list:
        img = Image.open(os.path.join(test_img_dir, test_img, probe_img))
        img = np.array(img)
        probe_imgs.append(img)
    probe_imgs = np.array(probe_imgs)
    probe_imgs = np.swapaxes(probe_imgs, 1, 3)  # BHWC to BCHW

    # Generate the in silico fMRI responses using BERG
    in_silico_fmri = berg.encode(model, probe_imgs)
    if i == 0:
        lh_response = in_silico_fmri[0]
        rh_response = in_silico_fmri[1]
    else:
        lh_response += in_silico_fmri[0]
        rh_response += in_silico_fmri[1]
    del in_silico_fmri, probe_imgs
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Define the probe location that elicits the maximum in silico fMRI response
# =============================================================================
# Define grid of probe centers (x,y) in degrees
coords = np.linspace(-args.FIELD_SIZE/2, args.FIELD_SIZE/2, args.GRID_RES)
xx, yy = np.meshgrid(coords, coords, indexing="xy")
centers = np.stack([xx.ravel(), yy.ravel()], axis=1)

# Define the probe location that elicits the maximum in silico fMRI responses
max_idx_lh = np.argmax(lh_response, axis=0) # type: ignore
max_idx_rh = np.argmax(rh_response, axis=0) # type: ignore
x0s_lh = centers[max_idx_lh, 0]
y0s_lh = centers[max_idx_lh, 1]
x0s_rh = centers[max_idx_rh, 0]
y0s_rh = centers[max_idx_rh, 1]


# =============================================================================
# Estimate the retinotopic maps (polar angle and eccentricity)
# =============================================================================
# arctan2 outputs values in the range [−π +π], but here polar angles set to the
# range [0 2π] to facilitate later color-coded visualization.

# LH
polar_angle_lh = np.mod(np.arctan2(y0s_lh, x0s_lh), 2 * np.pi)
eccentricity_lh = np.sqrt(x0s_lh**2 + y0s_lh**2)

# RH
polar_angle_rh = np.mod(np.arctan2(y0s_rh, x0s_rh), 2 * np.pi)
eccentricity_rh = np.sqrt(x0s_rh**2 + y0s_rh**2)


# =============================================================================
# Save the retinotopic maps
# =============================================================================
results = {
    'polar_angle_lh': polar_angle_lh,
    'eccentricity_lh': eccentricity_lh,
    'polar_angle_rh': polar_angle_rh,
    'eccentricity_rh': eccentricity_rh
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'retinotopic_maps',
    'encoding_model-'+args.encoding_model+'_subject-'+str(args.subject))
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.savez_compressed(os.path.join(save_dir, 'retinotopic_maps.npz'),
    data=results) # type: ignore