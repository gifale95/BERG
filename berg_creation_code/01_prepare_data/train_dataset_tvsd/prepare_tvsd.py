"""Preprocess the raw neural data from the THINGS Ventral Stream Spiking Dataset
(Papale et al., Neuron 2025):
 - split training and test data,
 - day-specific z-score normalization using pre-stimulus baseline,
 - create comprehensive metadata mapping.
After preprocessing, the neural data is saved as:
 - Training data: (Trials x Time points x Electrodes)
 - Test data: (Trials x Time points x Electrodes) and averaged version
The data is saved in HDF5 format for efficient loading during model training.

Parameters
----------
monkey : str
    Monkey identifier ('monkeyN' or 'monkeyF').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
tvsd_dir : str
    Directory of the TVSD dataset.
batch_size : int
    Batch size for chunked processing to manage memory usage.
"""

import argparse
from utils import split_tvsd_data, normalize_tvsd_data, create_tvsd_metadata
import os

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--monkey", required=True, choices=["monkeyN", "monkeyF"], 
                    help="Select which monkey's data to use.")
parser.add_argument('--berg_dir', required=True, type=str,
                    help="Directory of the BERG framework.")
parser.add_argument('--tvsd_dir', required=True, type=str,
                    help="Directory of the TVSD dataset.")
parser.add_argument('--batch_size', default=1000, type=int,
                    help="Batch size for chunked processing.")
args = parser.parse_args()

print('>>> TVSD Data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-tvsd_monkey')
os.makedirs(output_dir, exist_ok=True)

# Create input paths
monkey_path = os.path.join(args.tvsd_dir, args.monkey)
things_mua_trials = os.path.join(monkey_path, "THINGS_MUA_trials.mat")
things_mapping_file = os.path.join(monkey_path, "_logs/things_imgs.mat")

# =============================================================================
# Split training and test data
# =============================================================================
# Load raw neural data and split into training and test partitions based on
# stimulus type. Training data contains single presentations of 22,248 images,
# while test data contains 30 repetitions of 100 images for noise ceiling estimation.
split_tvsd_data(things_mua_trials, output_dir, args.monkey, args.batch_size)

# =============================================================================
# Normalize neural responses
# =============================================================================
# Apply day-specific z-score normalization using pre-stimulus baseline period
# (-100 to 0ms) to account for daily recording variations and electrode drift.
normalize_tvsd_data(things_mua_trials, output_dir, args.monkey, args.batch_size)

# =============================================================================
# Create dataset metadata
# =============================================================================
# Generate comprehensive metadata linking stimulus IDs to image files,
# object categories, and experimental conditions for both training and test sets.
create_tvsd_metadata(
    original_filepath=things_mua_trials,
    things_mapping_file=things_mapping_file, 
    output_dir=output_dir,
    monkey_id=args.monkey
)