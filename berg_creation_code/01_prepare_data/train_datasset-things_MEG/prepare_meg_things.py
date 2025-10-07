"""Preprocess the THINGS-data MEG dataset (Hebart et al., 2023):
 - split training and test data based on trial type,
 - session-specific z-score normalization using pre-stimulus baseline,
 - create comprehensive metadata mapping.

After preprocessing, the MEG data is saved as:
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


Usage
-----
python '/Users/domenicbersch/Documents/Repositories/NEST/berg_creation_code/01_prepare_data/train_datasset-things_MEG/prepare_meg_things.py' \
    --subject P1 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --meg_data_dir '/Volumes/Extreme SSD/Datasets/THINGS/LOCAL 2/ocontier/thingsmri/openneuro/THINGS-data/THINGS-MEG/ds004212/derivatives/preprocessed' \
    --batch_size 512


Output Files Created (per subject):
──────────────────────────────────────────────────────────────
meg_{subject}_split-train.h5           : (22248, 271, 281)
meg_{subject}_split-test.h5            : (2400, 271, 281)
meg_{subject}_split-train_normalized.h5: (22248, 271, 281)
meg_{subject}_split-test_normalized.h5 : (2400, 271, 281)
meg_{subject}_split-test_averaged.h5   : (200, 271, 281)
meg_{subject}_metadata.npz             :

    Training Data:
        train_things_img_ids  : (22248,) - THINGS image IDs for ViT linking
        train_categories      : (22248,) - Object category numbers (1-1854)
        train_exemplars       : (22248,) - Exemplar numbers (1-12)
        train_sessions        : (22248,) - Session numbers (1-12)
        train_runs            : (22248,) - Run numbers (1-10)
        train_image_paths     : (22248,) - Image file paths
        train_full_image_path : (22248,) - Relative paths (e.g., vest/vest_10s.jpg)
    
    
    Test Data (Individual Trials):
        test_things_img_ids   : (2400,)  - THINGS image IDs
        test_image_nr         : (2400,)  - Test image numbers (1-200)
        test_categories       : (2400,)  - Object categories
        test_exemplars        : (2400,)  - Exemplar numbers
        test_sessions         : (2400,)  - Session numbers
        test_runs             : (2400,)  - Run numbers
        test_image_paths      : (2400,)  - Image file paths
        test_full_image_path  : (2400,)  - Relative paths (e.g., limousine/limousine_15s.jpg)
    
    Test Data (Averaged Across 12 Repetitions):
        test_avg_things_img_ids : (200,)   - Unique THINGS image IDs
        test_avg_image_nr       : (200,)   - Test image numbers 1-200
        test_avg_categories     : (200,)   - Object categories
        test_avg_image_paths    : (200,)   - Image file paths
        test_avg_full_image_path: (200,)   - Relative paths (e.g., limousine/limousine_15s.jpg)
    
    Temporal Information:
        times                 : (281,)     - Time points (-0.1 to 1.3s)
    
    Sensor Information:
        sensor_names          : (271,)     - MEG sensor name strings
        sensor_prefixes       : (271,)     - Sensor prefixes (e.g., MLF, MRC, MZO)
        sensor_hemispheres    : (271,)     - Hemisphere labels (Left, Right, Midline)
        sensor_regions        : (271,)     - Region labels (Frontal, Central, Parietal, Temporal, Occipital)
        n_sensors             : int        - Number of sensors (271)
    
    Normalization Parameters:
        baseline_means        : (12, 271)  - Session-specific baseline means
        baseline_stds         : (12, 271)  - Session-specific baseline stds
        baseline_sessions     : (12,)      - Session numbers for baselines
        baseline_time_range   : (2,)       - Baseline period bounds [start, end]
        baseline_indices      : (20,)      - Time indices for baseline period
    
    Subject Metadata:
        subject_id            : str        - Subject identifier


Total: 6 files per subject
"""

import argparse
import os
from utils_meg import split_meg_data, normalize_meg_data, create_meg_metadata

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

print('>>> MEG THINGS-data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-things_meg')
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
# Normalize MEG responses
# =============================================================================
print("")
print("Normalizing training and testing data")
baseline_stats = normalize_meg_data(meg_file, output_dir, args.subject, args.batch_size)

# =============================================================================
# Create dataset metadata
# =============================================================================
print("")
print("Creating metadata")
create_meg_metadata(
    meg_filepath=meg_file,
    output_dir=output_dir,
    subject_id=args.subject,
    baseline_stats=baseline_stats
)

print("\nPreprocessing complete!")