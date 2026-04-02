"""Compute the ERPs of the in silico and in vivo EEG responses for the 200
THINGS EEG2 test images.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
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
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
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
# Load the EEG metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    args.encoding_model,
    subject=1
)

# Load the test images
test_img_files = metadata_eeg['encoding_models']['test_img_info']\
    ['test_img_files']

# Loop across test images
images = []
for file in tqdm(test_img_files):
    # Find correct subfolder
    img_path = None
    for root, _, files in os.walk(os.path.join(args.things_dir)):
        if file in files:
            img_path = os.path.join(root, file)
            break

    # Load and transform the image
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
    images.append(img)

# Format the images to a numpy array
images = np.array(images)


# =============================================================================
# Generate the in silico EEG responses using BERG, and compute the ERPs
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Empty result dictionaries
insilico_erps = []
metadata = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the encoding model
    model = berg.get_encoding_model(args.encoding_model, subject=sub)

    # Generate the in silico EEG responses, and average them across repeats and
    # image conditions
    eeg, metadata_sub = berg.encode(model, images, return_metadata=True)
    insilico_erps.append(np.mean(eeg, (0, 1)))
    metadata.append(metadata_sub)
    del eeg, metadata_sub, model
    torch.cuda.empty_cache()
    gc.collect()

# Convert to numpy arrays
insilico_erps = np.array(insilico_erps)
 

# =============================================================================
# Load the in vivo EEG responses for the same images, and compute the ERPs
# =============================================================================
# The in vivo EEG responses reflect the same data preprocessing version as the
# one used to train and test BERG's THINGS EEG2 encoding models. The code for
# this preprocessing is available at:
# https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data/train_dataset-things_eeg_2

invivo_erps = []

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Load the preprocessed in vivo EEG responses for the test images, and
    # average them across repeats and image conditions
    eeg_dir = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_eeg_2', 'eeg_sub-'+format(sub,'02')+
        '_split-test.h5')
    invivo_erps.append(np.mean(h5py.File(eeg_dir, 'r')['eeg'][:], (0, 1)))

# Convert to numpy arrays
invivo_erps = np.array(invivo_erps)


# =============================================================================
# Average the ERPs across channels from the same channel group
# =============================================================================
insilico_erps_chan_avg = {}
invivo_erps_chan_avg = {}

# Loop across channel groups
chan_groups = ['O', 'P', 'T', 'C', 'F']
ch_names = metadata[0]['eeg']['ch_names']
for chan in chan_groups:

    # Loop across EEG channels, and select the ones from the channel group of
    # interest
    idx_chan = []
    for c, ch_name in enumerate(ch_names):
        if chan in ch_name:
            idx_chan.append(c)
    idx_chan = np.array(idx_chan)

    # Average the ERPs across the selected channels
    insilico_erps_chan_avg[chan] = np.mean(insilico_erps[:,idx_chan], 1)
    invivo_erps_chan_avg[chan] = np.mean(invivo_erps[:,idx_chan], 1)


# =============================================================================
# Bootstrap the ERP confidence intervals
# =============================================================================
# Empty result variables
times = metadata[0]['eeg']['times']
ci_insilico_erps_chan_avg = {}
ci_invivo_erps_chan_avg = {}
for chan in chan_groups:
    ci_insilico_erps_chan_avg[chan] = np.zeros((2, len(times)))
    ci_invivo_erps_chan_avg[chan] = np.zeros((2, len(times)))

# Empty bootstrap distribution arrays
insilico_dist = {}
invivo_dist = {}
for chan in chan_groups:
    insilico_dist[chan] = np.zeros((args.n_iter, len(times)))
    invivo_dist[chan] = np.zeros((args.n_iter, len(times)))

# Compute the bootstrap distributions
for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    for chan in chan_groups:
        insilico_dist[chan][i] = np.mean(insilico_erps_chan_avg[chan][idx], 0)
        invivo_dist[chan][i] = np.mean(invivo_erps_chan_avg[chan][idx], 0)

# Compute the confidence intervals
for chan in chan_groups:
    ci_insilico_erps_chan_avg[chan][0] = np.percentile(insilico_dist[chan],
        2.5, axis=0)
    ci_insilico_erps_chan_avg[chan][1] = np.percentile(insilico_dist[chan],
        97.5, axis=0)
    ci_invivo_erps_chan_avg[chan][0] = np.percentile(invivo_dist[chan], 2.5,
        axis=0)
    ci_invivo_erps_chan_avg[chan][1] = np.percentile(invivo_dist[chan], 97.5,
        axis=0)


# =============================================================================
# Correlate the in vivo and in silico ERPs
# =============================================================================
corr_erps_chan_avg = {}

# Loop across channel groups
for chan in chan_groups:

    # Loop across subjects
    corr = []
    for s in range(len(args.subjects)):

        # Compute the correlation
        corr.append(pearsonr(insilico_erps_chan_avg[chan][s],
            invivo_erps_chan_avg[chan][s])[0])

    # Store the correlation results
    corr_erps_chan_avg[chan] = np.array(corr)


# =============================================================================
# Test for significance
# =============================================================================
pval_corr_erps_chan_avg = {}

# Loop across channel groups
for chan in chan_groups:

    # Compute the significance
    pval_corr_erps_chan_avg[chan] = ttest_1samp(corr_erps_chan_avg[chan], 0,
        alternative='greater')


# =============================================================================
# Compute the mean squared error between the in vivo and in silico ERPs
# =============================================================================
# Compute the MSE
mse_erps = (insilico_erps - invivo_erps) ** 2

# Average the MSE across occipital and parietal channels
idx_chan = []
for c, ch_name in enumerate(ch_names):
    if 'O' in ch_name or 'P' in ch_name:
        idx_chan.append(c)
idx_chan = np.array(idx_chan)
mse_erps = np.mean(mse_erps[:,idx_chan], 1)

# Average the MSE across time points between 60ms and 600ms
idx_time = np.where((times >= 0.06) & (times <= 0.6))[0]
mse_erps = np.mean(mse_erps[:,idx_time], 1)


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
    'mse_erps': mse_erps,
    'metadata': metadata
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'erps', 'erps', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'erps.npy'

np.save(os.path.join(save_dir, file_name), results)