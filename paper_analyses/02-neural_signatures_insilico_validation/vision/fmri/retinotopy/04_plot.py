"""Plot the the retinotopic maps (polar angle and eccentricity) computed on the
in silico fMRI responses.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
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
from tqdm import tqdm
import cortex
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-alexnet')
parser.add_argument('--fmri_subjects', default=[1], type=list) # !!! 1, 2, 3, 4, 5, 6, 7, 8
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=0, type=float)
parser.add_argument('--FIELD_SIZE', type=float, default=8.4)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'retinotopy', 'GRID_RES-'+str(args.GRID_RES)+
    '_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+'_BG_VALUE-'+str(args.BG_VALUE),
    'plots', args.encoding_model)

os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the results
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'retinotopy', 'GRID_RES-'+str(args.GRID_RES)+
    '_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+'_BG_VALUE-'+str(args.BG_VALUE),
    'stats', args.encoding_model, 'stats.npy')

results = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Load the retinotopic maps
# =============================================================================
# Loop across fMRI subjects
for s, sub in enumerate(tqdm(args.fmri_subjects)):

    # Get the metadata
    metadata = results['metadata'][s]

    # Get the polar angle
    polar_angle_lh = results['polar_angle_lh_silico'][s]
    polar_angle_rh = results['polar_angle_rh_silico'][s]
    # Rotate the polar angles by 150° for nicer color visualization
    polar_angle_lh = (polar_angle_lh + 150) % 360
    polar_angle_rh = (polar_angle_rh + 150) % 360

    # Get the eccentricity
    eccentricity_lh = results['eccentricity_lh_silico'][s]
    eccentricity_rh = results['eccentricity_rh_silico'][s]
    # Normalize the eccentricities to [0,1]
    max_ecc = np.append(eccentricity_lh, eccentricity_rh).max()
    max_ecc = 12 # !!! Check if this is true, and if so hard-code it so that it is constant across encoding models
    eccentricity_lh_norm = eccentricity_lh / max_ecc
    eccentricity_rh_norm = eccentricity_rh / max_ecc

    # Only retain vertices that have above threshold (i) NCSNR AND
    # (ii) encoding prediction accuracy.
    # Left hemisphere
    lh_ncsnr = metadata['fmri']['lh_ncsnr']
    idx_ncsnr = lh_ncsnr >= args.ncsnr_threshold
    lh_encoding = metadata['encoding_models']['lh_explained_variance_nsdcore']
    idx_encoding = lh_encoding >= args.encoding_threshold
    idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
    polar_angle_lh[idx_nan] = np.nan
    eccentricity_lh_norm[idx_nan] = np.nan
    # Right hemisphere
    rh_ncsnr = metadata['fmri']['rh_ncsnr']
    idx_ncsnr = rh_ncsnr >= args.ncsnr_threshold
    rh_encoding = metadata['encoding_models']['rh_explained_variance_nsdcore']
    idx_encoding = rh_encoding >= args.encoding_threshold
    idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
    polar_angle_rh[idx_nan] = np.nan
    eccentricity_rh_norm[idx_nan] = np.nan


# =============================================================================
# Plot the polar angle results
# =============================================================================
    # Append the results across left and right hemishperes
    data = np.append(polar_angle_lh, polar_angle_rh)

    # Create the flat brain surface
    subject = 'fsaverage_nsd_sub-0' + str(sub)
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='hsv',
        vmin=0,
        vmax=360,
        with_colorbar=True
        )

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'FFA-1', 'FFA-2', 'EBA', 'PPA', 'RSC'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=25,
        curvature_brightness=0.4,
        with_colorbar=False
        )

    # Save the figure
    file_name = os.path.join(save_dir, 'polar_angle_encoding_model-'+
        args.encoding_model+'_subject-'+str(sub)+'.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
        format='svg')
    plt.close(fig)


# =============================================================================
# Plot the eccentricity results
# =============================================================================
    # Append the results across left and right hemishperes
    data = np.append(eccentricity_lh_norm, eccentricity_rh_norm)

    # Create the flat brain surface
    subject = 'fsaverage_nsd_sub-0' + str(sub)
    vertex_data = cortex.Vertex(
        data,
        subject=subject,
        cmap='gist_rainbow',
        vmin=0,
        vmax=1,
        with_colorbar=True
        )

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'FFA-1', 'FFA-2', 'EBA', 'PPA', 'RSC'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=25,
        curvature_brightness=0.4,
        with_colorbar=False
        )

    # Save the figure
    file_name = os.path.join(save_dir, 'eccentricity_encoding_model-'+
        args.encoding_model+'_subject-'+str(sub)+'.svg')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
        format='svg')
    plt.close(fig)


# =============================================================================
# Plot the polar angle colorwheel
# =============================================================================
# Create a grid of coordinates
size = 2000  # image size of the colorwheel
radius = size // 2
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# Compute polar coordinates
R = np.sqrt(X**2 + Y**2)
theta = np.degrees(np.arctan2(Y, X))  # range: -180 to 180

# Mask outside the unit circle
mask = R <= 1

# Rotate by 240° (i.e. 90° to match the original brain maps + additional 150°
# since the brain maps were also rotated by 150°)
shift = 240 # (150° + 90°)
# Convert to degrees in the range [0, 360) and apply the rotation
theta_rot = (theta + shift) % 360

# Convert polar angle to HSV color wheel
# H = angle / 360
H = theta_rot / 360.0      # hue: 0–1
S = np.ones_like(H)        # full saturation
V = np.ones_like(H)        # full brightness

HSV = np.stack((H, S, V), axis=-1)
RGB = hsv_to_rgb(HSV)

# Apply mask
RGB[~mask] = np.nan

# Plot color wheel
fig = plt.figure(figsize=(6, 6))
plt.imshow(RGB, origin="lower")
plt.axis("off")
plt.show()

# Save the polar angle color wheel
file_name = os.path.join(save_dir, 'polar_angle_colorwheel.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
    format='svg')
plt.close(fig)


# =============================================================================
# Plot the eccentricity map # !!!
# =============================================================================
# Parameters
size = 2000
square_ecc = args.FIELD_SIZE
cmap = "gist_rainbow"

# Create coordinate grid
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# Convert to polar coordinates
R = np.sqrt(X**2 + Y**2)  # radius (0 at center, 1 at boundary)

# Normalize R to 0–1, then scale to max_ecc if desired
ecc = np.clip(R, 0, 1) * max_ecc

# Mask outside the circle
mask = R > 1

# Compute square boundary in normalized img space
s = square_ecc / max_ecc

# Prepare the image
ecc_img = ecc.copy()
ecc_img[mask] = np.nan  # transparent background outside circle

# Plot the colormap
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(ecc_img, cmap=cmap, origin="lower", extent=[-1,1,-1,1])
ax.axis("off")
# Draw dashed square
ax.plot([-s,  s], [ s,  s], 'k--', linewidth=2)
ax.plot([-s,  s], [-s,-s], 'k--', linewidth=2)
ax.plot([-s,-s], [-s, s], 'k--', linewidth=2)
ax.plot([ s, s], [-s, s], 'k--', linewidth=2)

# Save the eccentricity map
file_name = os.path.join(save_dir, 'eccentricity_map.svg')
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
    format='svg')
plt.close(fig)