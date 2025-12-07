"""Perform MDS on the in silico EEG RSMs for all images, or for the controlling
images. The EEG time points are used as samples, and the RSM entries as
features. MDS's goal is to reduce the RSM entries to two dimensions, such that
the multivariate EEG responses of each time point can be plotted in 2D space
based on its multivariate response similarity to other time points.

Parameters
----------
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> MDS multivariate responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Time point selection
# =============================================================================
time_points = ['0.1', '0.2', '0.3', '0.4']

time_point_comb_names = ['0.1-0.2', '0.1-0.3', '0.1-0.4', '0.2-0.3', '0.2-0.4',
    '0.3-0.4']

time_point_comb = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# =============================================================================
# Load the controlling image indices
# =============================================================================
align = {}
disentangle = {}

for time_point_pair in time_point_comb_names:

    data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
        'stats', 'cv-0', time_point_pair, 'stats.npy')
    stats = np.load(data_dir, allow_pickle=True).item()

    align[time_point_pair] = stats['best_generation_image_batches']['align'][-1]
    disentangle[time_point_pair] = stats['best_generation_image_batches']\
        ['disentangle'][-1]


# =============================================================================
# Load the RSMs averaged across subjects
# =============================================================================
img_cond = 10000
rsms_all_images = []
rsms_align = {}
rsms_disentangle = {}
idx_lower_tr_all = np.tril_indices(img_cond, -1)

for t, time in enumerate(time_points):

    # Load the in silico fMRI RSMs
    data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
        'rsms', 'averaged_rsm_time-'+time+'_all_subjects.npy')
    rsm = np.load(data_dir)

    # Store the RSMs for all image conditions
    rsms_all_images.append(rsm[idx_lower_tr_all])

    # Store the RSMs for the controlling images
    for time_point_pair in time_point_comb_names:
        if t == 0:
            rsms_align[time_point_pair] = []
            rsms_disentangle[time_point_pair] = []
        idx_lower_tr_cond = np.tril_indices(len(align[time_point_pair]), -1)
        # Align
        rsm_cond = rsm[align[time_point_pair]]
        rsm_cond = rsm_cond[:,align[time_point_pair]]
        rsm_cond = rsm_cond[idx_lower_tr_cond]
        rsms_align[time_point_pair].append(rsm_cond)
        # Disentangle
        rsm_cond = rsm[disentangle[time_point_pair]]
        rsm_cond = rsm_cond[:,disentangle[time_point_pair]]
        rsm_cond = rsm_cond[idx_lower_tr_cond]
        rsms_disentangle[time_point_pair].append(rsm_cond)
    del rsm, rsm_cond

# Reformat to numpy
rsms_all_images = np.asarray(rsms_all_images)
for time_point_pair in time_point_comb_names:
    rsms_align[time_point_pair] = np.asarray(rsms_align[time_point_pair])
    rsms_disentangle[time_point_pair] = np.asarray(
        rsms_disentangle[time_point_pair])


# =============================================================================
# Perform MDS using all image conditions
# =============================================================================
embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)

mds_all_images = embedding.fit_transform(rsms_all_images)


# =============================================================================
# Perform MDS using only the controlling images
# =============================================================================
mds_align = {}
mds_disentangle = {}

for time_point_pair in time_point_comb_names:

    # Align
    embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
    mds_align[time_point_pair] = embedding.fit_transform(
        rsms_align[time_point_pair])

    # Disentangle
    embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
    mds_disentangle[time_point_pair] = embedding.fit_transform(
        rsms_disentangle[time_point_pair])


# =============================================================================
# Save the results
# =============================================================================
results = {
    'mds_all_images': mds_all_images,
    'mds_align': mds_align,
    'mds_disentangle': mds_disentangle,
    'time_point_comb_names': time_point_comb_names,
    'time_point_comb': time_point_comb
    }

save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'multidimensional_scaling')
os.makedirs(save_dir, exist_ok=True)

file_name = 'mds_multivariate_responses.npy'

np.save(os.path.join(save_dir, file_name), results)