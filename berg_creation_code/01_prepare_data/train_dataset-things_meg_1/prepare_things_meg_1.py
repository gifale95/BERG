"""Preprocess the THINGS-data MEG dataset (Hebart et al., 2023):
 - split training and test data based on trial type,
 - create comprehensive metadata mapping,
 - optionally create random training splits.

The MEG data is saved as:
 - Training data: (Trials x Time points x Sensors)
 - Test data: (Trials x Time points x Sensors) and averaged version

The data is saved in HDF5 format for efficient loading during model training.

Parameters
----------
subject : str
    Subject identifier ('P1', 'P2', 'P3', or 'P4').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
meg_data_dir : str
    Directory containing the preprocessed MEG .fif files.
batch_size : int
    Batch size for chunked processing to manage memory usage.
create_splits : bool
    Whether to create 4 random training splits (default: True).


Output Files Created (per subject):
────────────────────────────────────────────────────────────────
meg_{subject}_all_training_splits.h5        : (22248, 271, 281) - Non-normalized training data (all splits)
meg_{subject}_split-test.h5                 : (2400, 271, 281)  - Non-normalized test data
meg_{subject}_split-test_averaged.h5        : (200, 271, 281)   - Non-normalized averaged test data

If create_splits=True, additionally:
meg_{subject}_single_training_split_1.h5    : (5562, 271, 281)  - Training split 1
meg_{subject}_single_training_split_2.h5    : (5562, 271, 281)  - Training split 2
meg_{subject}_single_training_split_3.h5    : (5562, 271, 281)  - Training split 3
meg_{subject}_single_training_split_4.h5    : (5562, 271, 281)  - Training split 4

meg_{subject}_metadata.npy                  :

'meg':
    times                      : (281,)   - Time points (e.g., -0.1 to 1.3s relative to stimulus onset)
    subject_id                 : str      - Subject identifier
'sensors':
    sensor_names               : (271,)   - MEG sensor name strings
    sensor_prefixes            : (271,)   - Sensor prefixes (e.g., 'MLF', 'MRC', 'MZO')
    sensor_hemispheres         : (271,)   - Hemisphere labels ('Left', 'Right', 'Midline')
    sensor_regions             : (271,)   - Region labels ('Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital')
    n_sensors                  : int      - Number of MEG sensors (271)
    
'encoding_model':
    all_training_splits:                   - Training data and encoding accuracy results for encoding models trained on all training splits
        train_img_ids          : (22248,) - THINGS image IDs for train trials
        train_concepts         : (22248,) - Object category IDs for train trials
        train_stimuli          : (22248,) - Image filenames for train trials
        train_sessions         : (22248,) - Session numbers for train trials
        train_runs             : (22248,) - Run numbers for train trials
        train_img_files        : (22248,) - Full image paths for train trials
        correlation_results    : (271, 281) - Prediction accuracy (Pearson's r) (added by 01_test_encoding.py)
        percent_noise_ceiling  : (271, 281) - Noise ceiling normalized prediction accuracy (% of noise ceiling) (added by 01_test_encoding.py)
    
    single_training_split_{N}:            - Training data and encoding accuracy results for encoding models trained on training split N
        train_img_ids          : (5562,)  - THINGS image IDs for train trials
        train_concepts         : (5562,)  - Object category IDs for train trials
        train_stimuli          : (5562,)  - Image filenames for train trials
        train_sessions         : (5562,)  - Session numbers for train trials
        train_runs             : (5562,)  - Run numbers for train trials
        train_img_files        : (5562,)  - Full image paths for train trials
        correlation_results    : (271, 281) - Prediction accuracy (Pearson's r) (added by 01_test_encoding.py)
        percent_noise_ceiling  : (271, 281) - Noise ceiling normalized prediction accuracy (% of noise ceiling) (added by 01_test_encoding.py)

    test_img_ids               : (2400,)  - THINGS image IDs for test trials
    test_stimuli               : (2400,)  - Image filenames for test trials
    test_concepts              : (2400,)  - Object category IDs for test trials
    test_image_nr              : (2400,)  - Test image numbers (1–200, repeated over repetitions)
    test_sessions              : (2400,)  - Session numbers for test trials
    test_runs                  : (2400,)  - Run numbers for test trials
    test_img_files             : (2400,)  - Full image paths on disk for test images
    
    ncsnr                      : (281, 271) - Neural cross-validated signal-to-noise ratio per time point and sensor
    noise_ceiling              : (281, 271) - Noise ceiling per time point and sensor
"""

import argparse
import os
from utils_meg import split_meg_data, create_meg_metadata

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True, choices=["P1", "P2", "P3", "P4"],
                    help="Select which subject's data to use.")
parser.add_argument('--berg_dir', required=True, type=str,
                    help="Directory of the BERG framework.")
parser.add_argument('--meg_data_dir', required=True, type=str,
                    help="Directory containing preprocessed MEG .fif files.")
parser.add_argument('--batch_size', default=1000, type=int,
                    help="Batch size for chunked processing.")
parser.add_argument('--create_splits', default=True, type=lambda x: str(x).lower() == 'true',
                    help="Create 4 random training splits (default: True).")
args = parser.parse_args()

print('>>> MEG THINGS-data preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-things_meg_1')
os.makedirs(output_dir, exist_ok=True)

# Create input path
meg_file = os.path.join(args.meg_data_dir, f'preprocessed_{args.subject}-epo.fif')

if not os.path.exists(meg_file):
    raise FileNotFoundError(f"MEG file not found: {meg_file}")

# =============================================================================
# Split training and test data
# =============================================================================
print("")
print("Splitting training and testing data")
shuffled_indices = split_meg_data(meg_file, output_dir, args.subject, args.batch_size, args.create_splits)


# =============================================================================
# Create dataset metadata
# =============================================================================
print("")
print("Creating metadata")
create_meg_metadata(
    meg_filepath=meg_file,
    output_dir=output_dir,
    subject_id=args.subject,
    create_splits=args.create_splits,
    shuffled_indices=shuffled_indices)

print("\nPreparation complete!")