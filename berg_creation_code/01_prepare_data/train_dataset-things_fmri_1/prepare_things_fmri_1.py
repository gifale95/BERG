"""Prepare the THINGS-fMRI1 dataset (Hebart et al., 2023):
 - split training and test data based on trial type,
 - create comprehensive metadata mapping,
 - generate averaged test data across repeated presentations.

After preparation, the fMRI data is saved as:
 - Training data: (Trials x Voxels) = (8640, 211339)   
 - Test data: (Trials x Voxels)    = (1200, 211339)    
 - Averaged test (unique images):  (100, 211339)  

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
python berg_creation_code/01_prepare_data/train_dataset-things_fMRI_1/prepare_things_fMRI_1.py \
    --subject sub-02 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --fmri_data_dir '/Volumes/Extreme SSD/Datasets/THINGS/betas_csv' \
    --batch_size 1000

Output Files Created (per subject):
────────────────────────────────────────────────────────────────────────────
Neural Data:
fmri_{subject}_split-train.h5           : (8640, 211339)
fmri_{subject}_split-test.h5            : (1200, 211339)
fmri_{subject}_split-test_averaged.h5   : (100, 211339)

Metadata:
fmri_{subject}_metadata.npy             :
    'fmri':
        voxel_coords         : (211339, 3) - Voxel coordinates in volume space (x, y, z indices)
        n_voxels             : int         - Total number of voxels (scalar = 211339)
        subject_id           : str         - Subject identifier (e.g., 'sub-01')

    'encoding_model':
        train_stimuli        : (8640,)  - Stimulus filenames for training trials
        train_concepts       : (8640,)  - Concept labels for training trials
        test_stimuli         : (1200,)  - Stimulus filenames for individual test trials
        test_concepts        : (1200,)  - Concept labels for individual test trials
        test_avg_stimuli     : (100,)   - Stimulus filenames for unique test images (after averaging)
        test_avg_concepts    : (100,)   - Concept labels for unique test images (after averaging)
        noise_ceiling_singletrial : (211339,) - Max explainable variance per voxel based on single-trial repeat reliability
        noise_ceiling_testset     : (211339,) - Max explainable variance per voxel based on averaged test-set repeats
        splithalf_corrected       : (211339,) - Raw split-half voxel reliability without correction
        splithalf_uncorrected     : (211339,) - Split-half reliability corrected to estimate full-data consistency

    'prf':
        prf_eccentricity     : (211339,) - Distance of receptive field center from fixation (deg)
        prf_polarangle       : (211339,) - Angular position of receptive field center (0–360°)
        prf_rsquared         : (211339,) - Variance explained by pRF model (fit quality)
        prf_size             : (211339,) - Estimated receptive field size (deg)

    'roi':
        V1, V2, V3, hV4, VO1, VO2, LO1_prf, LO2_prf, TO1, TO2, V3b, V3a,
        lFFA, rFFA, lOFA, rOFA, lEBA, rEBA, lPPA, rPPA, lRSC, rRSC, 
        lTOS, rTOS, lLOC, rLOC, IT, lSTS, rSTS
            Each ROI entry contains voxel indices (variable length) for that functional region
"""


import argparse
import os
from utils_things_fMRI_1 import split_fmri_data, create_fmri_metadata, create_averaged_test_data
import pandas as pd

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

print('>>> fMRI THINGS-data preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_dir = os.path.join(args.berg_dir, 'model_training_datasets', 'train_dataset-things_fmri_1')
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
split_fmri_data(
    response_file, stimulus_file, output_dir, args.subject, args.batch_size
)


# =============================================================================
# Create averaged test data (from normalized individual trials)
# =============================================================================
print("")
print("Creating averaged test data from normalized trials")
test_file = os.path.join(output_dir, f'fmri_{args.subject}_split-test.h5')

# Need to recreate test_mask for create_averaged_test_data
stim_metadata = pd.read_csv(stimulus_file)
test_mask = stim_metadata['trial_type'] == 'test'

create_averaged_test_data(test_file, stimulus_file, output_dir, args.subject, test_mask)




# =============================================================================
# Create dataset metadata
# =============================================================================
print("")
print("Creating metadata")
create_fmri_metadata(
    stimulus_filepath=stimulus_file,
    voxel_filepath=voxel_file,
    output_dir=output_dir,
    subject_id=args.subject
)