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
    roi                  : dict - ROI name → vertex indices in full HCP grayordinate space (e.g., 'L_V1' → array([3319, 3320, ...]))

'encoding_models':
    vertex_mapping_visual         : (7831,)  - Indices mapping visual cortex model predictions (GlasserGroups 1-5) to full 91k HCP space. Usage: pred_HCP = np.full((batch, 91282), np.nan); pred_HCP[:, vertex_mapping_visual] = predictions_7831
    vertex_mapping_all            : (57051,) - Indices mapping full cortex model predictions (GlasserGroups 1-22) to full 91k HCP space. Usage: pred_HCP = np.full((batch, 91282), np.nan); pred_HCP[:, vertex_mapping_all] = predictions_57051
    test_n-avg_noiseceiling       : (91282,) - Vertex-wise noise ceiling computed on naturalistic test stimuli (real-world photographic images) using repeat-averaged beta estimates.
    test_n-1_noiseceiling         : (91282,) - Vertex-wise noise ceiling computed on naturalistic test stimuli (real-world photographic images) using single-trial beta estimates.
    train_n-avg_noiseceiling      : (91282,) - Vertex-wise noise ceiling computed on naturalistic training stimuli (real-world photographic images used for model fitting) using repeat-averaged beta estimates.
    train_n-1_noiseceiling        : (91282,) - Vertex-wise noise ceiling computed on naturalistic training stimuli (real-world photographic images used for model fitting) using single-trial beta estimates.
    artificial_n-avg_noiseceiling : (91282,) - Vertex-wise noise ceiling computed on artificial test stimuli (controlled non-naturalistic images such as gratings, noise patterns, and simple shapes) using repeat-averaged beta estimates.
    artificial_n-1_noiseceiling   : (91282,) - Vertex-wise noise ceiling computed on artificial test stimuli (controlled non-naturalistic images such as gratings, noise patterns, and simple shapes) using single-trial beta estimates.


"""

import argparse
import os
from pathlib import Path
from utils import download_noise_ceilings, download_metadata, add_noise_ceilings_to_metadata, add_roi_indices_to_metadata, add_vertex_mappings_to_metadata

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
metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri', 'train_dataset-mosaic', 'model-mosaic', 'metadata')
nc_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri', 'train_dataset-mosaic', 'model-mosaic', 'noise_ceilings')
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(nc_dir, exist_ok=True)

# =============================================================================
# Download noise ceilings
# =============================================================================
# Download HDF5 files and extract noise ceilings
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
# Download ROI masks from glasser
print("Adding ROI indices to metadata")
add_roi_indices_to_metadata(metadata_dir)

# =============================================================================
# Add noise ceilings to metadata
# =============================================================================
# Map full cortex noise ceilings (91282 vertices) to prediction spaces:
# visual cortex (7831 vertices) and full cortex (57051 vertices).
print("Adding noise ceilings to metadata")
add_noise_ceilings_to_metadata(metadata_dir, nc_dir)


# =============================================================================
# Add vertex mapping to metadata
# =============================================================================
# Model predictions are in reduced vertex spaces (visual or all cortex), while noise ceilings
# and ROI indices are defined in full HCP grayordinate space (91,282 vertices). These mappings
# allow expansion: predictions_91k[vertex_mapping] = predictions_model.
add_vertex_mappings_to_metadata(metadata_dir)