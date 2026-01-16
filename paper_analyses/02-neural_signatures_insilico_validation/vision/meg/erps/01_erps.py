"""Compute the ERPs of the in silico and in vivo MEG responses for the 200
THINGS MEG1 test images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico MEG
    responses.
subjects : list
    List of MEG subject identifiers.
tmax : float
    Maximum epoch time point for the MEG analyses.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
import h5py
import gc
import torch
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='meg-things_meg_1-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--tmax', default=0.6, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> ERPs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the stimulus images
# =============================================================================
# Load the test images metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_meg = berg.get_model_metadata(
    args.encoding_model,
    subject=1
)
test_image_nr = metadata_meg['encoding_model']['test_image_nr']
test_img_files = metadata_meg['encoding_model']['test_img_files']
unique_test_img_nr = np.unique(test_image_nr)

# Loop across test images
images = []
for img_nr in tqdm(unique_test_img_nr):

    # Get the image directory
    img_file = test_img_files[np.where(test_image_nr == img_nr)[0][0]]
    img_path = os.path.join(args.things_dir, 'image-database_things', img_file)

    # Load and transform the image
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
    images.append(img)

# Format the images to a numpy array
images = np.array(images)


# =============================================================================
# Time point selection
# =============================================================================
times = metadata_meg['meg']['times']

timepoints = np.zeros(len(times), dtype=int)
timepoints[times <= args.tmax] = 1
times = times[times <= args.tmax]


# =============================================================================
# Generate the in silico MEG responses using BERG, and compute the ERPs
# =============================================================================
# Empty result dictionaries
insilico_erps = []
metadata = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(
        args.encoding_model,
        subject=sub,
        selection={'timepoints': timepoints}
    )

    # Generate the in silico MEG responses, and average them across repeats and
    # image conditions
    meg, metadata_sub = berg.encode(model, images, return_metadata=True)
    insilico_erps.append(np.mean(meg, (0, 1)))
    metadata.append(metadata_sub)
    del meg, metadata_sub, model
    torch.cuda.empty_cache()
    gc.collect()

# Convert to numpy arrays
insilico_erps = np.array(insilico_erps)
 

# =============================================================================
# Load the in vivo MEG responses for the same images, and compute the ERPs
# =============================================================================
# The in vivo MEG responses reflect the same data preparation version as the
# one used to train and test BERG's THINGS MEG1 encoding models. The code for
# this data preparation is available at:
# https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_meg_1

invivo_erps = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the preprocessed in vivo MEG responses for the test images, and
    # average them across repeats and image conditions
    meg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', 'meg_P'+str(sub)+'_split-test.h5')
    invivo_erps.append(np.mean(h5py.File(
        meg_dir, 'r')['neural_data'][:,:,np.where(timepoints)[0]], 0))

# Convert to numpy arrays
invivo_erps = np.array(invivo_erps)


# =============================================================================
# Average the ERPs across sensors from the same group
# =============================================================================
# Get the sensor groups from the metadata
sensor_regions = metadata_meg['sensors']['sensor_regions']

insilico_erps_chan_avg = {}
invivo_erps_chan_avg = {}

# Loop across sensor groups
for sensor in np.unique(sensor_regions):

    # Average the ERPs across sensors from the same group
    idx_sensor = np.where(sensor_regions == sensor)[0]
    insilico_erps_chan_avg[sensor] = np.mean(insilico_erps[:,idx_sensor], 1)
    invivo_erps_chan_avg[sensor] = np.mean(invivo_erps[:,idx_sensor], 1)


# =============================================================================
# Bootstrap the ERP confidence intervals
# =============================================================================
# Empty result variables
ci_insilico_erps_chan_avg = {}
ci_invivo_erps_chan_avg = {}
for sensor in np.unique(sensor_regions):
    ci_insilico_erps_chan_avg[sensor] = np.zeros((2, len(times)))
    ci_invivo_erps_chan_avg[sensor] = np.zeros((2, len(times)))
# Empty bootstrap distribution arrays
insilico_dist = {}
invivo_dist = {}
for sensor in np.unique(sensor_regions):
    insilico_dist[sensor] = np.zeros((args.n_iter, len(times)))
    invivo_dist[sensor] = np.zeros((args.n_iter, len(times)))

# Compute the bootstrap distributions
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    for sensor in np.unique(sensor_regions):
        insilico_dist[sensor][i] = np.mean(insilico_erps_chan_avg[sensor][idx],
            0)
        invivo_dist[sensor][i] = np.mean(invivo_erps_chan_avg[sensor][idx], 0)

# Compute the confidence intervals
for sensor in np.unique(sensor_regions):
    ci_insilico_erps_chan_avg[sensor][0] = np.percentile(insilico_dist[sensor],
        2.5, axis=0)
    ci_insilico_erps_chan_avg[sensor][1] = np.percentile(insilico_dist[sensor],
        97.5, axis=0)
    ci_invivo_erps_chan_avg[sensor][0] = np.percentile(invivo_dist[sensor],
        2.5, axis=0)
    ci_invivo_erps_chan_avg[sensor][1] = np.percentile(invivo_dist[sensor],
        97.5, axis=0)


# =============================================================================
# Correlate the in vivo and in silico ERPs
# =============================================================================
corr_erps_chan_avg = {}

# Loop across channel groups
for sensor in np.unique(sensor_regions):

    # Loop across subjects
    corr = []
    for s in range(len(args.subjects)):

        # Compute the correlation
        corr.append(pearsonr(insilico_erps_chan_avg[sensor][s],
            invivo_erps_chan_avg[sensor][s])[0])

    # Store the correlation results
    corr_erps_chan_avg[sensor] = np.array(corr)


# =============================================================================
# Test for significance
# =============================================================================
pval_corr_erps_chan_avg = {}

# Loop across channel groups
for sensor in np.unique(sensor_regions):

    # Compute the significance
    pval_corr_erps_chan_avg[sensor] = ttest_1samp(corr_erps_chan_avg[sensor],
        0, alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'insilico_erps': insilico_erps,
    'invivo_erps': invivo_erps,
    'insilico_erps_chan_avg': insilico_erps_chan_avg,
    'invivo_erps_chan_avg': invivo_erps_chan_avg,
    'ci_insilico_erps_chan_avg': ci_insilico_erps_chan_avg,
    'ci_invivo_erps_chan_avg': ci_invivo_erps_chan_avg,
    'corr_erps_chan_avg': corr_erps_chan_avg,
    'pval_corr_erps_chan_avg': pval_corr_erps_chan_avg,
    'metadata': metadata,
    'times': times
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'erps', 'erps', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'erps.npy'

np.save(os.path.join(save_dir, file_name), results)