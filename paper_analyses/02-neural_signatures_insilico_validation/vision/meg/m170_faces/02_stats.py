"""Compute stats on the results of the M170 ERP analysis. The stats consist of
bootstrapped 95% confidence intervals for the ERP, and of the calculation of
significant differences between the magnituted of the M170 component for faces
versus objects.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico MEG
    responses.
subjects : list
    List of MEG subject identifiers.
sensors : list
    List containing the names of the individual MEG sensors used.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='meg-things_meg_1-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4], type=int)
parser.add_argument('--sensors', default=['MLT23', 'MLT24', 'MLT33', 'MLT34', 'MRT23', 'MRT24', 'MRT33', 'MRT34'], type=list) # !!!
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> MEG M170 - Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in silico MEG responses
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'm170_faces', 'insilico_meg_responses',
    args.encoding_model, 'insilico_meg_responses.npy')

data = np.load(data_dir, allow_pickle=True).item()

meg_faces = data['insilico_meg']['Faces']
meg_objects = data['insilico_meg']['Objects']
metadata = data['metadata']
times = data['times']
del data


# =============================================================================
# MEG sensor selection
# =============================================================================
sensor_names = metadata[0]['sensors']['sensor_names']

# Kept sensor indices
kept_sensor_idx = []
for s, sensor in enumerate(sensor_names):
    for sensor_select in args.sensors:
        if sensor_select in sensor[:5]:
            kept_sensor_idx.append(s)
            break

# Average the MEG responses across the chosen sensors
meg_faces = np.mean(meg_faces[:,:,kept_sensor_idx], 2)
meg_objects = np.mean(meg_objects[:,:,kept_sensor_idx], 2)


# =============================================================================
# Compute the ERPs (average across images from the same condition)
# =============================================================================
erp_faces = np.mean(meg_faces, 1)
erp_objects = np.mean(meg_objects, 1)


# =============================================================================
# Bootstrap the ERP confidence intervals (CIs)
# =============================================================================
ci_erp_faces = np.zeros((2, erp_faces.shape[1]))
ci_erp_objects = np.zeros((ci_erp_faces.shape))

faces_dist = np.zeros((args.n_iter, erp_faces.shape[1]))
objects_dist = np.zeros((faces_dist.shape))

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    faces_dist[i] = np.mean(erp_faces[idx], 0)
    objects_dist[i] = np.mean(erp_objects[idx], 0)

ci_erp_faces[0] = np.percentile(faces_dist, 2.5, axis=0)
ci_erp_faces[1] = np.percentile(faces_dist, 97.5, axis=0)
ci_erp_objects[0] = np.percentile(objects_dist, 2.5, axis=0)
ci_erp_objects[1] = np.percentile(objects_dist, 97.5, axis=0)


# =============================================================================
# Statistical significance
# =============================================================================
# Compute the difference between face and object absolute ERPs
erp_diff = erp_faces - erp_objects

# Compute the p-value
pval_erp_diff = ttest_rel(erp_faces, erp_objects, axis=0,
    alternative='less')[1]

# Multiple comparison correction
sig_erp_diff, pval_erp_diff_corrected, _, _ = multipletests(pval_erp_diff,
    0.05, 'fdr_bh')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'erp_faces': erp_faces,
    'erp_objects': erp_objects,
    'ci_erp_faces': ci_erp_faces,
    'ci_erp_objects': ci_erp_objects,
    'pval_erp_diff': pval_erp_diff,
    'pval_erp_diff_corrected': pval_erp_diff_corrected,
    'sig_erp_diff': sig_erp_diff,
    'metadata': metadata,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'm170_faces', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_sensors-' + '-'.join(args.sensors) + '.npy'

np.save(os.path.join(save_dir, file_name), results)