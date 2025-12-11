"""Perform searchlight RSA between in silico fMRI responses and LLM embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses.
subject : int
    The subject identifier for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
criterion : str
    Criterion to define the searchlight neighborhood: 'radius' for all vertices
    within a geodesic radius, 'nearest' for k-nearest neighbors.
radius_mm : float
    Geodesic radius in millimeters (default = 10 mm), if criterion is 'radius'.
k : int
    Number of nearest geodesic neighbors (default = 10), if criterion is
    'nearest'.
berg_dir : str
    Directory of the BERG.
nsd_dir : str
    Directory of the Natural Scenes Dataset.
    https://naturalscenesdataset.org/
"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
import pandas as pd
from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--criterion', default='radius', type=str)
parser.add_argument('--radius_mm', default=10, type=float)
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Define the vectorized correlation function
# =============================================================================
def corr_matrix(X):
    """
    Computes the correlation matrix of the input data.
    Parameters
    ----------
    X : (N, M) float array
        Input data matrix with N features and M samples.

    Returns
    -------
    corr : (M, M) float array
        Correlation matrix of the input data.
    """

    Xc = X - X.mean(axis=0)
    Xc /= np.sqrt((Xc**2).sum(axis=0))

    return (Xc.T @ Xc).astype(np.float32)


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Select the vertices from the chosen hemisphere
n_vertices = 163842
if args.hemisphere == 'lh':
    lh_vertices = np.ones(n_vertices, dtype=int)
    rh_vertices = np.zeros(n_vertices, dtype=int)
    rh_vertices[0] = 1 # At least one vertex must be selected
elif args.hemisphere == 'rh':
    lh_vertices = np.zeros(n_vertices, dtype=int)
    lh_vertices[0] = 1 # At least one vertex must be selected
    rh_vertices = np.ones(n_vertices, dtype=int)

# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={
        'lh_vertices': lh_vertices,
        'rh_vertices': rh_vertices
        }
    )


# =============================================================================
# Load the test images
# =============================================================================
# The test images consist of the 515 images that all NSD subjects saw for three
# times, and which were used to test BERG's encoding models

# Get the test image number
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
)
test_img_num = metadata['encoding_models']['test_img_num']

# Load the test images
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')
images = sdataset[test_img_num]
images = np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)


# =============================================================================
# Generate the in silico fMRI response
# =============================================================================
fmri, metadata = berg.encode(model, images, return_metadata=True)

# Only retain responses from the hemisphere of interest
if args.hemisphere == 'lh':
    fmri = fmri[0].astype(np.float32)
if args.hemisphere == 'rh':
    fmri = fmri[1].astype(np.float32)


# =============================================================================
# Create the LLM RDM # !!!
# =============================================================================
# Load the captions





dataDir='..'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()













embedding_model = SentenceTransformer('all-mpnet-base-v2')
embedding = embedding_model.encode(['This is my sentence'])



# Create the RDM
llm_rdm = 1 - corr_matrix(llm_embeddings.T)


# =============================================================================
# Perform searchlight RSA
# =============================================================================
# Empty RSA results array
rsa = np.zeros(fmri.shape[1], dtype=np.float32)

# Take the lower triangle of the LLM RDM
idx_tril = np.tril_indices(len(llm_rdm), -1)
llm_rdm_tril = llm_rdm[idx_tril]

# Get info regarding the vertex splits of the geodesic distances
n_vertices = fmri.shape[1]
total_vertex_splits = 81
vertices_per_split = n_vertices // total_vertex_splits

# Loop across fMRI vertices
for v in tqdm(range(fmri.shape[1])):

    # Only load the precomputed geodesic distances for the first vertex of each
    # split
    idx = v % vertices_per_split
    if idx == 0:
        vertex_split = v // vertices_per_split # Get the split of the target vertex
        data_dir = np.load(os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation', 'vision', 'fmri',
            'behavioral_modeling', 'vertex_geodesic_distances',
            'vertex_geodesic_distances_'+args.hemisphere+'_split-'+
            format(vertex_split, '03')+'.h5py'))
        geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances'][:]

    # Select the neighborhood based on the chosen criterion
    if args.criterion == 'nearest':
        # Get k-smallest distances (including the target vertex)
        neighborhood = np.argsort(geodesic_distances[idx])[:args.k]
    elif args.criterion == 'radius':
        # Select all vertices whose distance is within the radius
        mask = geodesic_distances[idx] <= args.radius_mm
        neighborhood = np.where(mask)[0]

    # Create the fMRI RDM
    fmri_rdm = 1 - corr_matrix(fmri[:,neighborhood].T)

    # Take the lower triangle of the fMRI RDM
    fmri_rdm_tril = fmri_rdm[idx_tril]

    # Perform RSA
    rsa[v] = pearsonr(llm_rdm_tril, fmri_rdm_tril)[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'rsa': rsa,
    'metadata': metadata
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'llm_modeling', 'rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsa_sub-' + format(args.subject, '02') + '_' + args.hemisphere + \
    '.npy'

np.save(os.path.join(save_dir, file_name), results)