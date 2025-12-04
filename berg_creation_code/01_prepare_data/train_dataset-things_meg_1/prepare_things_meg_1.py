"""Preprocess the THINGS-data MEG dataset (Hebart et al., 2023):
 - split training and test data based on trial type,
 - create comprehensive metadata mapping.

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


Output Files Created (per subject):
────────────────────────────────────────────────────────────────
meg_{subject}_split-train.h5                : (22248, 271, 281) - Non-normalized training data
meg_{subject}_split-test.h5                 : (2400, 271, 281)  - Non-normalized test data
meg_{subject}_split-test_averaged.h5        : (200, 271, 281)   - Non-normalized averaged test data
meg_{subject}_metadata.npz                  :

'meg':
    times                      : (281,)   - Time points (e.g., -0.1 to 1.3s relative to stimulus onset)
    subject_id                 : str      - Subject identifier
'sensors:
    sensor_names               : (271,)   - MEG sensor name strings
    sensor_prefixes            : (271,)   - Sensor prefixes (e.g., 'MLF', 'MRC', 'MZO')
    sensor_hemispheres         : (271,)   - Hemisphere labels ('Left', 'Right', 'Midline')
    sensor_regions             : (271,)   - Region labels ('Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital')
    n_sensors                  : int      - Number of MEG sensors (271)
    
'encoding_model':
    train_img_ids              : (22248,) - THINGS image IDs for training (for ViT linking)
    train_concepts             : (22248,) - Object category IDs (1–1854)
    train_sessions             : (22248,) - Session numbers for training trials
    train_runs                 : (22248,) - Run numbers within each session
    train_img_files            : (22248,) - Full image paths on disk for training images
    
    test_things_img_ids        : (2400,)  - THINGS image IDs for test trials
    test_image_nr              : (2400,)  - Test image numbers (1–200, repeated over repetitions)
    test_concepts              : (2400,)  - Object category IDs for test trials
    test_sessions              : (2400,)  - Session numbers for test trials
    test_runs                  : (2400,)  - Run numbers for test trials
    test_img_files             : (2400,)  - Full image paths on disk for test images
    
    test_avg_things_img_ids    : (200,)   - Unique THINGS image IDs used in averaging
    test_avg_image_nr          : (200,)   - Test image numbers 1–200 (averaged over repetitions)
    test_avg_concepts          : (200,)   - Object category IDs for averaged test images
    test_avg_img_files         : (200,)   - Full image paths on disk for averaged test images
    
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
split_meg_data(meg_file, output_dir, args.subject, args.batch_size)


# =============================================================================
# Create dataset metadata
# =============================================================================
print("")
print("Creating metadata")
create_meg_metadata(
    meg_filepath=meg_file,
    output_dir=output_dir,
    subject_id=args.subject)

print("\nPreparation complete!")