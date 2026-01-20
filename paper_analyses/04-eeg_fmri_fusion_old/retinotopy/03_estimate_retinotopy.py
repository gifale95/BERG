"""Generate the t-fMRI responses to the retinotopic mapping stimuli. Then, for
each vertex and time point estimate the retinotopic maps (polar angle and
eccentricity) from the t-fMRI responses.

Parameters
----------
fmri_subject : int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
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
import h5py
from berg import BERG
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate t-fMRI <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define grid of probe centers (x,y) in degrees
# =============================================================================
coords = np.linspace(-args.FIELD_SIZE/2, args.FIELD_SIZE/2, args.GRID_RES)
xx, yy = np.meshgrid(coords, coords, indexing="xy")
centers = np.stack([xx.ravel(), yy.ravel()], axis=1)


# =============================================================================
# Create the saving directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'retinotopic_maps')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# List the in silico EEG responses for each test image category
# =============================================================================
eeg_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'insilico_eeg')

eeg_files = os.listdir(eeg_dir)
eeg_files.sort()


# =============================================================================
# Generate the t-fMRI responses
# =============================================================================
# Only use vertices falling within the NSD visual streams
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    'fmri-nsd_fsaverage-huze',
    subject=args.fmri_subject
    )
n_vertex = 163842
idx_v = np.zeros(n_vertex, dtype=int)
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral',
    'lateral', 'parietal']
for stream in streams:
    idx_v[metadata['fmri'][f'{args.hemisphere}_fsaverage_rois'][stream]] = 1
idx_v = np.where(idx_v == 1)[0]

# Loop across EEG time points
eeg_shape = h5py.File(os.path.join(eeg_dir, eeg_files[0]), 'r')['eeg']
for t in tqdm(range(eeg_shape.shape[2])):

    # Load the EEG-fMRI encoding fusion model weights
    model_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
        'encoding_fusion_weights')
    file_name = (f'weights_fmri_sub-{args.fmri_subject:02d}_'
            f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')
    reg_param = np.load(os.path.join(model_dir, file_name),
        allow_pickle=True).item()

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_'][idx_v]
    reg.intercept_ = reg_param['intercept_'][idx_v]
    reg.n_features_in_ = reg_param['n_features_in_']
    del reg_param

    # Empty t-fMRI response array of shape:
    # (1600 Probe images, N Vertices)
    tfmri = np.zeros((eeg_shape.shape[0], len(idx_v)), dtype=np.float32)

    # Loop across EEG responses for the test image categories
    for i, eeg_file in enumerate(eeg_files):

        # Load the in silico EEG responses
        eeg = h5py.File(os.path.join(eeg_dir, eeg_file), 'r')['eeg']

        # Generate and store the t-fMRI responses
        tfmri += reg.predict(eeg[:,:,t])

        # Delete unused variables
        del eeg
    del reg


# =============================================================================
# Define the probe location that elicits the maximum in silico fMRI response
# =============================================================================
    max_idx = np.argmax(tfmri, axis=0)
    x0s = centers[max_idx, 0]
    y0s = centers[max_idx, 1]
    del tfmri


# =============================================================================
# Estimate the retinotopic maps (polar angle and eccentricity)
# =============================================================================
    # arctan2 outputs values in the range [−π +π], but here polar angles set to
    # the range [0 2π] to facilitate later color-coded visualization.
    polar_angle = np.mod(np.arctan2(y0s, x0s), 2 * np.pi)
    eccentricity = np.sqrt(x0s**2 + y0s**2)

    # Set values of vertices outside the visual streams to NaN
    polar_angle_all_vertices = np.empty(n_vertex, dtype=np.float32)
    eccentricity_all_vertices = np.empty(n_vertex, dtype=np.float32)
    polar_angle_all_vertices[:] = np.nan
    eccentricity_all_vertices[:] = np.nan
    polar_angle_all_vertices[idx_v] = polar_angle
    eccentricity_all_vertices[idx_v] = eccentricity


# =============================================================================
# Save the retinotopic maps
# =============================================================================
    results = {
        'polar_angle': polar_angle_all_vertices,
        'eccentricity': eccentricity_all_vertices,
        }

    file_name = (f'retinotopy_sub-{args.fmri_subject:02d}_'
        f'hemi-{args.hemisphere}_eeg_time-{t:03d}.npy')

    np.save(os.path.join(save_dir, file_name), results)