"""Correlate the polar angle and eccentricity maps defined in silico with the
ones defined in vivo from the NSD experiment.

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
nsd_dir : str
	Directory of the Natural Scenes Dataset (NSD).
	https://naturalscenesdataset.org/
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
from nsdcode.nsd_mapdata import NSDmapdata
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float)
parser.add_argument('--encoding_threshold', default=0, type=float)
parser.add_argument('--FIELD_SIZE', type=float, default=16.8)
parser.add_argument('--GRID_RES', type=int, default=40)
parser.add_argument('--PROBE_SIGMA', type=float, default=0.5)
parser.add_argument('--BG_VALUE', type=float, default=0.5)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Initiate NSDmapdata/BERG, empty results lists, and loop across fMRI subjects
# =============================================================================
nsd = NSDmapdata(args.nsd_dir)
berg = BERG(berg_dir=args.berg_dir)

polar_angle_lh_silico = []
polar_angle_rh_silico = []
eccentricity_lh_silico = []
eccentricity_rh_silico = []
polar_angle_lh_vivo = []
polar_angle_rh_vivo = []
eccentricity_lh_vivo = []
eccentricity_rh_vivo = []
prf_r2_lh_vivo = []
prf_r2_rh_vivo = []
corr_polar_angle_silico_vivo = []
corr_eccentricity_silico_vivo = []
metadata = []

for s, sub in enumerate(tqdm(args.fmri_subjects)):


# =============================================================================
# Load the retinotopic maps defined in silico
# =============================================================================
    # Load the retinotopic maps
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'retinotopy', 'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+
        str(args.PROBE_SIGMA)+'_BG_VALUE-'+str(args.BG_VALUE),
        'retinotopic_maps', 'encoding_model-'+args.encoding_model+'_subject-'+
        str(sub), 'retinotopic_maps.npz')
    data = np.load(data_dir, allow_pickle=True)["data"].item()
    polar_angle_lh_silico.append(data['polar_angle_lh'])
    polar_angle_rh_silico.append(data['polar_angle_rh'])
    eccentricity_lh_silico.append(data['eccentricity_lh'])
    eccentricity_rh_silico.append(data['eccentricity_rh'])

    # Load the metadata
    metadata.append(berg.get_model_metadata(
        args.encoding_model,
        subject=sub
        ))


# =============================================================================
# Load the retinotopic maps defined in vivo from NSD
# =============================================================================
    # NSD data directory for the current subject
    data_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer',
    f'subj{sub:02d}', 'label')

    # Convert the polar angle maps from subject native surface space to
    # fsaverage
    polar_angle_lh_vivo.append(np.squeeze(nsd.fit(sub, 'lh.white', 'fsaverage',
        os.path.join(data_dir, 'lh.prfangle.mgz'))))
    polar_angle_rh_vivo.append(np.squeeze(nsd.fit(sub, 'rh.white', 'fsaverage',
        os.path.join(data_dir, 'rh.prfangle.mgz'))))

    # Convert the eccentricity maps from subject native surface space to
    # fsaverage
    eccentricity_lh_vivo.append(np.squeeze(nsd.fit(sub, 'lh.white',
        'fsaverage', os.path.join(data_dir, 'lh.prfeccentricity.mgz'))))
    eccentricity_rh_vivo.append(np.squeeze(nsd.fit(sub, 'rh.white',
        'fsaverage', os.path.join(data_dir, 'rh.prfeccentricity.mgz'))))

    # Convert the variance explained by the pRF model from subject native
    # surface space to fsaverage
    prf_r2_lh_vivo.append(np.squeeze(nsd.fit(sub, 'lh.white',
        'fsaverage', os.path.join(data_dir, 'lh.prfR2.mgz'))))
    prf_r2_rh_vivo.append(np.squeeze(nsd.fit(sub, 'rh.white',
        'fsaverage', os.path.join(data_dir, 'rh.prfR2.mgz'))))


# =============================================================================
# Vertex thresholding
# =============================================================================
    # Only retain vertices that have above threshold (i) NCSNR AND
    # (ii) encoding prediction accuracy.
    # Left hemisphere
    lh_idx_ncsnr = metadata[s]['fmri']['lh_ncsnr'] >= \
        args.ncsnr_threshold
    rh_idx_ncsnr = metadata[s]['fmri']['rh_ncsnr'] >= \
        args.ncsnr_threshold
    lh_idx_encoding = \
        metadata[s]['encoding_models']['lh_explained_variance_nsdcore'] >= \
        args.encoding_threshold
    rh_idx_encoding = \
        metadata[s]['encoding_models']['rh_explained_variance_nsdcore'] >= \
        args.encoding_threshold

    # Only retain vertices with a variance explained by the pRF model trained
    # on the in vivo NSD data of at least 20% (this is to avoid including noisy
    # vertices that are not well explained by the pRF model in the following
    # correlation analysis)
    lh_idx_r2 = prf_r2_lh_vivo[s] >= 20
    rh_idx_r2 = prf_r2_rh_vivo[s] >= 20

    # Threshold the retinotopic maps
    lh_idx = np.logical_and(lh_idx_ncsnr, np.logical_and(lh_idx_encoding,
        lh_idx_r2))
    rh_idx = np.logical_and(rh_idx_ncsnr, np.logical_and(rh_idx_encoding,
        rh_idx_r2))
    polar_angle_vivo = np.append(polar_angle_lh_vivo[s][lh_idx],
        polar_angle_rh_vivo[s][rh_idx])
    polar_angle_silico = np.append(polar_angle_lh_silico[s][lh_idx],
        polar_angle_rh_silico[s][rh_idx])
    eccentricity_vivo = np.append(eccentricity_lh_vivo[s][lh_idx],
        eccentricity_rh_vivo[s][rh_idx])
    eccentricity_silico = np.append(eccentricity_lh_silico[s][lh_idx],
        eccentricity_rh_silico[s][rh_idx])


# =============================================================================
# Correlate the in silico and in vivo retinotopic maps
# =============================================================================
    # Transform the in silico polar angle maps from radians to degrees and
    # rotate them counterclockwise by 90 degrees to match the in vivo polar
    # angle maps
    corr_polar_angle_silico_vivo.append(pearsonr(polar_angle_vivo,
        (np.degrees(polar_angle_silico) - 90) % 360)[0])

    # Clip the eccentricity values at 8.4 degrees of visual angle to avoid
    # outliers (since the NSD stimuli were presented within a circular aperture
    # of 8.4 degrees of visual)
    corr_eccentricity_silico_vivo.append(pearsonr(
        np.clip(eccentricity_vivo, 0, 8.4),
        np.clip(eccentricity_silico, 0, 8.4))[0])


# =============================================================================
# Compute the significance
# =============================================================================
p_val_corr_polar_angle_silico_vivo = ttest_1samp(corr_polar_angle_silico_vivo,
    0, alternative='greater')
p_val_corr_eccentricity_silico_vivo = ttest_1samp(
    corr_eccentricity_silico_vivo, 0, alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'polar_angle_lh_silico': polar_angle_lh_silico,
    'polar_angle_rh_silico': polar_angle_rh_silico,
    'eccentricity_lh_silico': eccentricity_lh_silico,
    'eccentricity_rh_silico': eccentricity_rh_silico,
    'polar_angle_lh_vivo': polar_angle_lh_vivo,
    'polar_angle_rh_vivo': polar_angle_rh_vivo,
    'eccentricity_lh_vivo': eccentricity_lh_vivo,
    'eccentricity_rh_vivo': eccentricity_rh_vivo,
    'prf_r2_lh_vivo': prf_r2_lh_vivo,
    'prf_r2_rh_vivo': prf_r2_rh_vivo,
    'corr_polar_angle_silico_vivo': corr_polar_angle_silico_vivo,
    'corr_eccentricity_silico_vivo': corr_eccentricity_silico_vivo,
    'p_val_corr_polar_angle_silico_vivo': p_val_corr_polar_angle_silico_vivo,
    'p_val_corr_eccentricity_silico_vivo': p_val_corr_eccentricity_silico_vivo,
    'metadata': metadata
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'fmri', 'retinotopy',
    'GRID_RES-'+str(args.GRID_RES)+'_PROBE_SIGMA-'+str(args.PROBE_SIGMA)+
    '_BG_VALUE-'+str(args.BG_VALUE), 'stats',
    'encoding_model-'+args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.save(os.path.join(save_dir, 'stats.npy'), results)