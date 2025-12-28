"""Prepare MOSAIC fMRI datasets for encoding model training:
 - Download noise ceilings from MOSAIC HDF5 files,
 - Download metadata and create subject-wise metadata with stimulus mappings,
 - Add ROI binary masks for visual and full cortex spaces,
 - Map noise ceilings to prediction spaces.

After preparation, metadata files contain stimulus information, train/test
splits, ROI masks, and noise ceiling estimates for each subject.

Datasets processed: BOLD5000, deeprecon, GOD, NSD, THINGS, BMD, NOD, HAD

Parameters
----------
berg_dir : str
    Directory of the BERG framework.

Output Files Created (per subject):
────────────────────────────────────────────────────────────────
mosaic_metadata/{dataset}/sub-{id}.npy : Subject metadata

    'fmri':
        participant_id       : str      - Subject identifier
        age                  : int      - Subject age
        sex                  : str      - Subject sex
        filenames            : (70850,)   - All stimulus filenames
        alias                : (70850,)   - Stimulus aliases
        source               : (70850,)   - Stimulus sources
        train_idx            : (69566,)   - Indices of training trials
        test_idx             : (1284,)    - Indices of test trials
        train_filenames      : (69566,)   - Training stimulus filenames
        test_filenames       : (1284,)    - Test stimulus filenames
        reps                 : (70850,)   - Repetition count per stimulus for this subject
        roi_visual_vertices  : dict - ROI name → binary mask (7831,) for visual cortex
        roi_all_vertices     : dict - ROI name → binary mask (57051,) for full cortex
        
    'encoding_models':
        test_n-avg_noiseceiling_visual_vertices : (7831,) - Noise ceiling for visual cortex
        test_n-avg_noiseceiling_all_vertices    : (57051,) - Noise ceiling for full cortex

"""

import argparse
import os
from pathlib import Path
from utils import download_noise_ceilings, download_metadata, add_noise_ceilings_to_metadata, add_roi_masks_to_metadata

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', required=True, type=str,
                    help="Directory of the BERG framework.")
args = parser.parse_args()

print('>>> MOSAIC Data Preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directories
metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri', 'train_dataset-mosaic', 'metadata')
nc_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri', 'train_dataset-mosaic', 'noise_ceilings')
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(nc_dir, exist_ok=True)

# =============================================================================
# Download noise ceilings
# =============================================================================
download_noise_ceilings(nc_dir)

# =============================================================================
# Create subject metadata
# =============================================================================
# Generate metadata linking neural responses to stimulus information for
# each subject. Includes participant demographics, stimulus filenames,
# train/test splits, and subject-specific repetition counts.
print("Creating subject metadata")
download_metadata(metadata_dir)

# =============================================================================
# Add ROI masks to metadata
# =============================================================================
# Create binary masks indicating which vertices belong to each brain region
# for both visual cortex space (7831 vertices) and full cortex space (57051 vertices).
print("Adding ROI masks to metadata")
add_roi_masks_to_metadata(metadata_dir)

# =============================================================================
# Add noise ceilings to metadata
# =============================================================================
# Map full cortex noise ceilings (91282 vertices) to prediction spaces:
# visual cortex (7831 vertices) and full cortex (57051 vertices).
print("Adding noise ceilings to metadata")
add_noise_ceilings_to_metadata(metadata_dir, nc_dir)