"""Prepare metadata for the Mouse Visual Cortex Foundation Model dataset
(Wang et al., Nature 2025).

This script extracts and organizes metadata for each (session, scan_idx) 
combination from the mouse calcium imaging dataset. The metadata includes
anatomical information, orientation/direction tuning properties, and model
performance metrics for all recorded units.

Parameters
----------
data_path : str
    Path to MOUSE dataset directory containing anatomy, performance, and
    ori_dir_tuning subdirectories.
output_path : str
    Path to output directory for metadata files.

Output Files Created (per session/scan combination):
────────────────────────────────────────────────────────────────
session{session}_scan{scan}_metadata.npy : Metadata for one recording session

    'calcium_2p':
        session              : int      - Session identifier
        scan                 : int      - Scan index within session
        animal_id            : int      - Animal identifier (17797)
        unit_id              : (N,)     - Unit identifiers (N = number of neurons)
        coordinates          : (N, 3)   - 3D spatial coordinates (x, y, z) for each unit
        OSI                  : (N,)     - Orientation Selectivity Index
        DSI                  : (N,)     - Direction Selectivity Index
        gOSI                 : (N,)     - Global Orientation Selectivity Index
        gDSI                 : (N,)     - Global Direction Selectivity Index
        pref_ori             : (N,)     - Preferred orientation (degrees)
        pref_dir             : (N,)     - Preferred direction (degrees)
        roi                  : dict     - Binary masks for brain regions
            V1               : (N,)     - Visual area 1 (1 = unit in V1, 0 = not in V1)
            LM               : (N,)     - Lateral medial area
            AL               : (N,)     - Anterolateral area
            RL               : (N,)     - Rostrolateral area
        field_masks          : dict     - Binary masks for imaging fields
            field_1          : (N,)     - Imaging field 1 (1 = unit in field, 0 = not)
            field_2          : (N,)     - Imaging field 2
            ...              : ...      - Additional fields as present in data
    'encoding_model':
        cc_abs               : (N,)     - Absolute correlation coefficient
        cc_max               : (N,)     - Maximum correlation coefficient (noise ceiling)
        cc_norm              : (N,)     - Normalized correlation coefficient
        
        
Per-session neuron counts:
    Session 4, Scan 7:  7,493 neurons 
    Session 5, Scan 6:  8,592 neurons 
    Session 5, Scan 7:  8,138 neurons 
    Session 6, Scan 2:  8,158 neurons 
    Session 6, Scan 4:  8,221 neurons 
    Session 6, Scan 6:  7,971 neurons
    Session 6, Scan 7:  7,887 neurons 
    Session 7, Scan 3:  8,618 neurons 
    Session 7, Scan 5:  8,194 neurons 
    Session 8, Scan 5:  9,941 neurons 
    Session 9, Scan 3:  7,973 neurons 
    Session 9, Scan 4:  7,855 neurons
    Session 9, Scan 6:  5,130 neurons 
    
python berg_creation_code/01_prepare_data/train_dataset-mouse/prepare_mouse.py --data_path '/Volumes/Extreme SSD/Datasets/MOUSE'
    
"""

import argparse
from utils import extract_mouse_metadata
from pathlib import Path

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, type=str,
                    help="Path to MOUSE dataset directory.")
parser.add_argument('--berg_dir', default='/Volumes/Extreme SSD/brain-encoding-response-generator', type=str,
                    help="Directory of the Brain Encoding Response Generator (BERG).")
args = parser.parse_args()

print('>>> Mouse Foundation Model Metadata Extraction <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Create output directory
output_path = Path(args.berg_dir) / 'encoding_models' / 'modality-calcium_2p' / 'train_dataset-wang_2025' / 'model-3DCNN' / 'metadata'
output_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Extract metadata for all sessions
# =============================================================================
# Process each (session, scan_idx) combination to create comprehensive metadata
# files linking neural units to their anatomical locations, tuning properties,
# and encoding model performance metrics.
print("")
print("Extracting metadata for all sessions")
extract_mouse_metadata(
    data_path=Path(args.data_path),
    output_path=output_path
)