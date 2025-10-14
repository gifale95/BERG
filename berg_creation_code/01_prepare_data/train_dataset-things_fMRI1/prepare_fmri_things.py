"""Preprocess the THINGS-fMRI dataset (Hebart et al., 2023):
 - split training and test data based on trial type,
 - normalize data using voxel-wise z-scoring,
 - create comprehensive metadata mapping,
 - generate averaged test data across repeated presentations.

After preprocessing, the fMRI data is saved as:
 - Training data: (Trials x Voxels) = (8640, 211339)   
 - Test data: (Trials x Voxels)    = (1200, 211339)    
 - Averaged test (unique images):  (100, 211339)
 - Normalized versions of all three datasets      

The data is saved in HDF5 format for efficient loading during model training.

Parameters
----------
subject : str
    Subject identifier (e.g., 'sub-01', 'sub-02', 'sub-03').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
fmri_data_dir : str
    Directory containing the preprocessed fMRI HDF5 and CSV files.
batch_size : int
    Batch size for chunked processing to manage memory usage.

Usage
-----
python berg_creation_code/01_prepare_data/train_dataset-things_fMRI/prepare_fmri_things.py \
    --subject sub-01 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --fmri_data_dir '/Volumes/Extreme SSD/Datasets/THINGS/betas_csv' \
    --batch_size 1000

Output Files Created (per subject):
────────────────────────────────────────────────────────────────
Original Data:
fmri_{subject}_split-train.h5           : (8640, 211339)           # Training data 
fmri_{subject}_split-test.h5            : (1200, 211339)           # Test data 
fmri_{subject}_split-test_averaged.h5   : (100, 211339)            # Averaged test data 

Normalized Data:
fmri_{subject}_split-train_normalized.h5           : (8640, 211339)  # Normalized training
fmri_{subject}_split-test_normalized.h5            : (1200, 211339)  # Normalized test
fmri_{subject}_split-test_averaged_normalized.h5   : (100, 211339)   # Normalized averaged test

Metadata:
fmri_{subject}_metadata.npz             :

    Training Data (sub-01):
        train_sessions            : (8640,)   int64
        train_runs                : (8640,)   int64
        train_stimuli             : (8640,)   object   e.g. ['dog_12s.jpg', 'mango_12s.jpg', ...]
        train_concepts            : (8640,)   object   e.g. ['dog', 'mango', ...]
        train_trial_ids           : (8640,)   int64

    Test Data – Individual Trials (sub-01):
        test_sessions             : (1200,)   int64
        test_runs                 : (1200,)   int64
        test_stimuli              : (1200,)   object
        test_concepts             : (1200,)   object
        test_trial_ids            : (1200,)   int64

    Test Data – Averaged Across Repetitions (sub-01):
        test_avg_stimuli          : (100,)    str
        test_avg_concepts         : (100,)    str

    Normalization Parameters:
        voxel_mean                : (211339,)     float64  # Mean per voxel from training
        voxel_std                 : (211339,)     float64  # Std per voxel from training

    Voxel Information (common shapes across subjects; values shown for sub-01):
        voxel_coords              : (211339, 3)   int64   # voxel indices
        noise_ceiling_singletrial : (211339,)     float64
        noise_ceiling_testset     : (211339,)     float64
        splithalf_corrected       : (211339,)     float64
        splithalf_uncorrected     : (211339,)     float64
        prf_eccentricity          : (211339,)     float64
        prf_polarangle            : (211339,)     float64
        prf_rsquared              : (211339,)     float64
        prf_size                  : (211339,)     float64
        n_voxels                  : int = 211339

    ROI Indices (Functional ROIs; counts shown for sub-01):
        roi_V1        : (1049,)   int64
        roi_V2        : (774,)    int64
        roi_V3        : (762,)    int64
        roi_hV4       : (613,)    int64
        roi_VO1       : (287,)    int64
        roi_VO2       : (149,)    int64
        roi_LO1_prf   : (349,)    int64
        roi_LO2_prf   : (348,)    int64
        roi_TO1       : (369,)    int64
        roi_TO2       : (316,)    int64
        roi_V3b       : (402,)    int64
        roi_V3a       : (642,)    int64
        roi_lFFA      : (154,)    int64
        roi_rFFA      : (399,)    int64
        roi_lOFA      : (69,)     int64
        roi_rOFA      : (250,)    int64
        roi_lEBA      : (563,)    int64
        roi_rEBA      : (640,)    int64
        roi_lPPA      : (395,)    int64
        roi_rPPA      : (414,)    int64
        roi_lRSC      : (420,)    int64
        roi_rRSC      : (558,)    int64
        roi_lTOS      : (38,)     int64
        roi_rTOS      : (121,)    int64
        roi_lLOC      : (1573,)   int64
        roi_rLOC      : (1127,)   int64
        roi_IT        : (4145,)   int64
        roi_lSTS      : (69,)     int64
        roi_rSTS      : (449,)    int64

    Subject Metadata:
        subject_id                : 'sub-01'

    ROI Summary (sub-01):
        Total functional ROIs     : 29
        Total voxels in ROIs      : 17,444

Total: 7 files per subject (6 HDF5 data files + 1 metadata)

Note: All HDF5 files use the key 'neural_data'.
      Data shape is (Trials x Voxels) after transposition from original (Voxels x Trials).
      Normalized data uses voxel-wise z-scoring: (data - mean) / std, computed from training data.
"""


import argparse
import os
from utils_fmri import split_fmri_data, normalize_fmri_data, create_fmri_metadata

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True, type=str,
                    help="Subject identifier (e.g., 'sub-01', 'sub-02').")
parser.add_argument('--berg_dir', required=True, type=str,
                    help="Directory of the BERG framework.")
parser.add_argument('--fmri_data_dir', required=True, type=str,
                    help="Directory containing preprocessed fMRI HDF5 and CSV files.")
parser.add_argument('--batch_size', default=1000, type=int,
                    help="Batch size for chunked processing.")
args = parser.parse_args()

print('>>> fMRI THINGS-data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-things_fmri')
os.makedirs(output_dir, exist_ok=True)

# Create input paths
response_file = os.path.join(args.fmri_data_dir, f'{args.subject}_ResponseData.h5')
stimulus_file = os.path.join(args.fmri_data_dir, f'{args.subject}_StimulusMetadata.csv')
voxel_file = os.path.join(args.fmri_data_dir, f'{args.subject}_VoxelMetadata.csv')

# Check if files exist
if not os.path.exists(response_file):
    raise FileNotFoundError(f"Response data file not found: {response_file}")
if not os.path.exists(stimulus_file):
    raise FileNotFoundError(f"Stimulus metadata file not found: {stimulus_file}")
if not os.path.exists(voxel_file):
    raise FileNotFoundError(f"Voxel metadata file not found: {voxel_file}")

# =============================================================================
# Split training and test data
# =============================================================================
print("")
print("Splitting training and testing data")
split_fmri_data(response_file, stimulus_file, output_dir, args.subject, args.batch_size)

# =============================================================================
# Normalize fMRI data
# =============================================================================
print("")
norm_stats = normalize_fmri_data(output_dir, args.subject)

# =============================================================================
# Create dataset metadata
# =============================================================================
print("")
print("Creating metadata")
create_fmri_metadata(
    stimulus_filepath=stimulus_file,
    voxel_filepath=voxel_file,
    output_dir=output_dir,
    subject_id=args.subject,
    norm_stats=norm_stats
)

print("\nPreprocessing complete!")