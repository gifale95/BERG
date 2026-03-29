"""Functional localizer t-value maps with the ones defined in vivo from the NSD
experiment.

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

lh_tval_silico = {}
rh_tval_silico = {}
lh_tval_vivo = {}
rh_tval_vivo = {}
lh_floc_r2_vivo = []
rh_floc_r2_vivo = []
corr_tval_silico_vivo = {}
metadata = []

for s, sub in enumerate(tqdm(args.fmri_subjects)):


# =============================================================================
# Load the functional localizer t-value maps defined in silico
# =============================================================================
    # Load the functional localizer t-value maps
    data_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'hvc_selectivity', 't_values', args.encoding_model,
        f'results_sub-{sub:02d}.npy')
    data = np.load(data_dir, allow_pickle=True).item()
    for cat in ['face', 'body', 'house', 'food']:
        if s == 0:
            lh_tval_silico[cat] = []
            rh_tval_silico[cat] = []
        lh_tval_silico[cat].append(data['lh_tval'][cat])
        rh_tval_silico[cat].append(data['rh_tval'][cat])

    # Load the metadata
    metadata.append(berg.get_model_metadata(
        args.encoding_model,
        subject=sub
        ))


# =============================================================================
# Load the functional localizer t-value maps defined in vivo from NSD
# =============================================================================
    for cat in ['face', 'body', 'house']:
        if s == 0:
            lh_tval_vivo[cat] = []
            rh_tval_vivo[cat] = []

    # NSD data directory for the current subject
    data_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer',
    f'subj{sub:02d}', 'label')

    # Convert the functional localizer t-value maps from subject native surface
    # space to fsaverage
    lh_tval_vivo['face'].append(np.squeeze(nsd.fit(sub, 'lh.white', 'fsaverage',
        os.path.join(data_dir, 'lh.floc-faces.mgz'))))
    rh_tval_vivo['face'].append(np.squeeze(nsd.fit(sub, 'rh.white', 'fsaverage',
        os.path.join(data_dir, 'rh.floc-faces.mgz'))))
    lh_tval_vivo['body'].append(np.squeeze(nsd.fit(sub, 'lh.white', 'fsaverage',
        os.path.join(data_dir, 'lh.floc-bodies.mgz'))))
    rh_tval_vivo['body'].append(np.squeeze(nsd.fit(sub, 'rh.white', 'fsaverage',
        os.path.join(data_dir, 'rh.floc-bodies.mgz'))))
    lh_tval_vivo['house'].append(np.squeeze(nsd.fit(sub, 'lh.white', 'fsaverage',
        os.path.join(data_dir, 'lh.floc-places.mgz'))))
    rh_tval_vivo['house'].append(np.squeeze(nsd.fit(sub, 'rh.white', 'fsaverage',
        os.path.join(data_dir, 'rh.floc-places.mgz'))))

    # Convert the variance explained by the functional localizer GLM model
    # from subject native surface space to fsaverage
    lh_floc_r2_vivo.append(np.squeeze(nsd.fit(sub, 'lh.white', 'fsaverage',
        os.path.join(data_dir, 'lh.flocR2.mgz'))))
    rh_floc_r2_vivo.append(np.squeeze(nsd.fit(sub, 'rh.white', 'fsaverage',
        os.path.join(data_dir, 'rh.flocR2.mgz'))))


# =============================================================================
# Vertex thresholding
# =============================================================================
    # Only retain vertices that have above threshold (i) NCSNR AND
    # (ii) encoding prediction accuracy.
    # Left hemisphere
    lh_idx_ncsnr = metadata[s]['fmri']['lh_ncsnr'] >= args.ncsnr_threshold
    rh_idx_ncsnr = metadata[s]['fmri']['rh_ncsnr'] >= args.ncsnr_threshold
    lh_idx_encoding = \
        metadata[s]['encoding_models']['lh_explained_variance_nsdcore'] >= \
        args.encoding_threshold
    rh_idx_encoding = \
        metadata[s]['encoding_models']['rh_explained_variance_nsdcore'] >= \
        args.encoding_threshold

    # Only retain vertices with a variance explained by the functional
    # localizer GLM model trained on the in vivo NSD data of at least 20% (this
    # is to avoid including noisy vertices that are not well explained by the
    # GLM model in the following correlation analysis)
    lh_idx_r2 = lh_floc_r2_vivo[s] >= 20
    rh_idx_r2 = rh_floc_r2_vivo[s] >= 20

    # Threshold the retinotopic maps
    lh_idx = np.logical_and(lh_idx_ncsnr, np.logical_and(lh_idx_encoding,
        lh_idx_r2))
    rh_idx = np.logical_and(rh_idx_ncsnr, np.logical_and(rh_idx_encoding,
        rh_idx_r2))
    tval_silico = {}
    tval_vivo = {}
    for cat in ['face', 'body', 'house']:
        tval_silico[cat] = np.append(lh_tval_silico[cat][s][lh_idx],
            rh_tval_silico[cat][s][rh_idx])
        tval_vivo[cat] = np.append(lh_tval_vivo[cat][s][lh_idx],
            rh_tval_vivo[cat][s][rh_idx])


# =============================================================================
# Correlate the in silico and in vivo functional localizer t-value maps
# =============================================================================
    for cat in ['face', 'body', 'house']:

        if s == 0:
            corr_tval_silico_vivo[cat] = []

        corr_tval_silico_vivo[cat].append(pearsonr(tval_vivo[cat],
            tval_silico[cat])[0])


# =============================================================================
# Compute the significance
# =============================================================================
p_val_corr_tval_silico_vivo = {}

for cat in ['face', 'body', 'house']:

    p_val_corr_tval_silico_vivo[cat] = ttest_1samp(
        np.array(corr_tval_silico_vivo[cat]), 0, alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_tval_silico': lh_tval_silico,
    'rh_tval_silico': rh_tval_silico,
    'lh_tval_vivo': lh_tval_vivo,
    'rh_tval_vivo': rh_tval_vivo,
    'lh_floc_r2_vivo': lh_floc_r2_vivo,
    'rh_floc_r2_vivo': rh_floc_r2_vivo,
    'corr_tval_silico_vivo': corr_tval_silico_vivo,
    'p_val_corr_tval_silico_vivo': p_val_corr_tval_silico_vivo,
    'metadata': metadata
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'hvc_selectivity', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.save(os.path.join(save_dir, 'stats.npy'), results)