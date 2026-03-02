"""Prepare the preprocessed neural data from the THINGS Ventral Stream Spiking
Dataset (Papale et al., Neuron 2025):
 - split training and test data,
 - create comprehensive metadata mapping,
 - optionally create random training splits.

After preparation, the neural data is saved as:
 - Training data: (Trials x Electrodes x Time points)
 - Test data: (Trials x Electrodes x Time points)
 - Test data (trial average): (100 Test conditions x Electrodes x Time points)
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
create_splits : bool
    Whether to create 4 random training splits (default: True).
    
    
Output Files Created (per monkey):
────────────────────────────────────────────────────────────────
tvsd_{monkey}_all_training_splits.h5       : (22,248, 1024, 300)
tvsd_{monkey}_split-test.h5                : (3,000, 1024, 300)
tvsd_{monkey}_split-test_averaged.h5       : (100, 1024, 300)

If create_splits=True, additionally:
tvsd_{monkey}_single_training_split_1.h5   : (5,562, 1024, 300)
tvsd_{monkey}_single_training_split_2.h5   : (5,562, 1024, 300)
tvsd_{monkey}_single_training_split_3.h5   : (5,562, 1024, 300)
tvsd_{monkey}_single_training_split_4.h5   : (5,562, 1024, 300)

tvsd_{monkey}_metadata.npy                 : 

    'utah-array':
        times                : (300,)   - Time points (-100 to 199ms)
        electrode_order      : (1024,)  - Electrode mapping order (0-based)
        monkey_id            : str      - Monkey identifier
        n_electrodes         : int      - Number of electrodes (1024)
        
    'roi':
        roi_assignments      : (1024,)  - ROI assignment per electrode (0=V1, 1=V4, 2=IT)
        roi_labels           : (3,)     - ROI label names ['V1', 'V4', 'IT']
    
    'encoding_model':
        all_training_splits:                   - Training data and encoding accuracy results for encoding models trained on all training splits
            train_img_ids        : (22248,) - Training stimulus IDs
            train_stimuli        : (22248,) - Training image filenames
            train_concepts       : (22248,) - Training object categories
            train_days           : (22248,) - Recording days for training
            train_sequence_pos   : (22248,) - Position in 4-image sequence
            correlation_results  : (1024, 300) - Prediction accuracy (Pearson's r) (added by 01_test_encoding.py)
            percent_noise_ceiling: (1024, 300) - Noise ceiling normalized prediction accuracy (% of noise ceiling) (added by 01_test_encoding.py)
        
        single_training_split_{N}:            - Training data and encoding accuracy results for encoding models trained on training split N
            train_img_ids        : (5562,)  - Training stimulus IDs
            train_stimuli        : (5562,)  - Training image filenames
            train_concepts       : (5562,)  - Training object categories
            train_days           : (5562,)  - Recording days for training
            train_sequence_pos   : (5562,)  - Position in 4-image sequence
            correlation_results  : (1024, 300) - Prediction accuracy (Pearson's r) (added by 01_test_encoding.py)
            percent_noise_ceiling: (1024, 300) - Noise ceiling normalized prediction accuracy (% of noise ceiling) (added by 01_test_encoding.py)
        
        test_img_ids         : (3000,)  - Test stimulus IDs (individual trials)
        test_stimuli         : (3000,)  - Test image filenames (individual)
        test_concepts        : (3000,)  - Test object categories (individual)
        test_days            : (3000,)  - Recording days for test
        test_sequence_pos    : (3000,)  - Position in sequence for test
        
        SNR                  : (4, 1024) - Signal-to-noise ratio per day per electrode
        SNR_max              : (1024,)  - Best SNR across all days per electrode
        ncsnr                : (1024, 300) - Neural signal-to-noise ratio per electrode/timepoint
        noise_ceiling        : (1024, 300) - Noise Ceiling per electrode for all timepoints
"""

import argparse
from utils import split_tvsd_data, create_tvsd_metadata
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
parser.add_argument('--create_splits', default=True, type=lambda x: str(x).lower() == 'true',
                    help="Create 4 random training splits (default: True).")
args = parser.parse_args()

print('>>> TVSD Data preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-tvsd')
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
print("")
print("Splitting training and testing data")
shuffled_indices = split_tvsd_data(things_mua_trials, output_dir, args.monkey, args.batch_size, args.create_splits)

# =============================================================================
# Create dataset metadata
# =============================================================================
# Generate comprehensive metadata linking stimulus IDs to image files,
# object categories, experimental conditions, and baseline normalization statistics.
print("")
print("Creating metadata")
create_tvsd_metadata(
    original_filepath=things_mua_trials,
    things_mapping_file=things_mapping_file, 
    output_dir=output_dir,
    monkey_id=args.monkey,
    create_splits=args.create_splits,
    shuffled_indices=shuffled_indices
)