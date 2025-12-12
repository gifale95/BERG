"""Compute the geodesic vertex distances between each vertex in fsaverage
space.

Parameters
----------
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
total_vertex_splits : int
    Total number of splits to divide the vertices into smaller chunks for
    parallel processing.
vertex_split : int
    Index of the vertex split to process (0-indexed).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import nibabel as nib
from nilearn import datasets
import pygeodesic.geodesic as geodesic

parser = argparse.ArgumentParser()
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--total_vertex_splits', default=81, type=int)
parser.add_argument('--vertex_split', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Compute geodesic distance <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the surface to compute the geodesic distances 
# =============================================================================
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

if args.hemisphere == 'lh':
    surf = nib.load(fsaverage['pial_left'])
elif args.hemisphere == 'rh':
    surf = nib.load(fsaverage['pial_right'])

vertices = surf.darrays[0].data
faces = surf.darrays[1].data


# =============================================================================
# Select the vertices from the vertex split
# =============================================================================
n_vertices = vertices.shape[0]
vertices_per_split = n_vertices // args.total_vertex_splits

start_vertex = args.vertex_split * vertices_per_split
if args.vertex_split == args.total_vertex_splits - 1:
    end_vertex = n_vertices
else:
    end_vertex = (args.vertex_split + 1) * vertices_per_split


# =============================================================================
# Compute the geodesic distances
# =============================================================================
# Empty array to store the geodesic distances
geodesic_distances = np.zeros((end_vertex-start_vertex, vertices.shape[0]),
    dtype=np.float32)

# Initialize solver
geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

# Loop across fMRI vertices
for v, vertex in enumerate(tqdm(range(start_vertex, end_vertex))):

    # Compute distances from the target vertex to all other vertices
    geodesic_distances[v] = geoalg.geodesicDistances(
        source_indices=np.array([vertex]), target_indices=None)[0]


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'behavioral_modeling', 'vertex_geodesic_distances')
os.makedirs(save_dir, exist_ok=True)

file_name = 'vertex_geodesic_distances_' + args.hemisphere + '_split-' + \
    format(args.vertex_split, '03') + '.h5'

with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
    f.create_dataset('geodesic_distances', data=geodesic_distances,
        dtype=np.float32)