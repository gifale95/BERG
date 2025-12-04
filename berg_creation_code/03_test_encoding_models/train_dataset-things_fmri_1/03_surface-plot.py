"""Plot encoding model prediction accuracy on cortical surfaces using pycortex.
Visualizes voxel-wise correlation results on inflated cortical surfaces.

Parameters
----------
subjects : list of str
    List of subjects to analyze (e.g., 'sub-01 sub-02 sub-03').
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
pycortex_filestore : str
    Path to pycortex filestore containing subject surfaces.
transform : str
    Name of the transform to use (default: 'align_auto').
    
    
python berg_creation_code/03_test_encoding_models/train_dataset-things_fmri_1/03_surface-plot.py \
    --subjects sub-01 sub-02 sub-03 \
    --berg_dir /Volumes/Extreme\ SSD/brain-encoding-response-generator \
    --pycortex_filestore /Volumes/Extreme\ SSD/brain-encoding-response-generator/pycortex_filestore \
    --transform align_auto
    
"""

import argparse
import os
import numpy as np
import cortex
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=str, nargs='+', required=True,
                   help="List of subject IDs (e.g., 'sub-01 sub-02 sub-03')")
parser.add_argument('--berg_dir', required=True, type=str)
parser.add_argument('--pycortex_filestore', required=True, type=str,
                   help="Path to pycortex filestore")
parser.add_argument('--transform', type=str, default='align_auto',
                   help="Transform name (default: align_auto)")
args = parser.parse_args()

n_subjects = len(args.subjects)
print(f"Processing {n_subjects} subjects: {', '.join(args.subjects)}")


# =============================================================================
# Configure pycortex
# =============================================================================
cortex.database.default_filestore = args.pycortex_filestore

# Subject mapping
subject_map = {'sub-01': 'S1', 'sub-02': 'S2', 'sub-03': 'S3'}


# =============================================================================
# Process each subject
# =============================================================================
for subject in args.subjects:
    print(f"\n{'='*80}")
    print(f"Processing {subject}")
    print('='*80)
    
    # Load encoding model metadata
    metadata_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
        'train_dataset-things_fmri_1', 'model-vit_b_32', 'metadata')
    file_name = f'metadata_{subject}.npy'
    metadata_file = os.path.join(metadata_dir, file_name)
    
    metadata = np.load(metadata_file, allow_pickle=True).item()
    correlations = metadata['encoding_model']['correlation_results']
    
    # Turn into explained_variance
    explained_variance = correlations ** 2

    coords = metadata['fmri']['voxel_coords']
    
    print(f"Explained variance shape: {explained_variance.shape}")
    print(f"Range before clipping: [{explained_variance.min():.4f}, {explained_variance.max():.4f}]")

    
    # Clip correlations between 0 and max value
    max_ev = explained_variance.max()
    explained_variance_clipped = np.clip(explained_variance, 0, max_ev)

    print(f"Range after clipping: [{explained_variance_clipped.min():.4f}, {explained_variance_clipped.max():.4f}]")

    
    # =============================================================================
    # Create 3D volume
    # =============================================================================
    # Calculate volume shape from coordinate ranges (Z, Y, X for pycortex)
    vol_shape = (coords[:, 2].max() + 1, 
                coords[:, 1].max() + 1, 
                coords[:, 0].max() + 1)
    print(f"Volume shape (Z, Y, X): {vol_shape}")

    # Create empty volume filled with zeros
    volume_3d = np.zeros(vol_shape)

    # Place clipped explained variance values at their coordinates
    volume_3d[coords[:, 2], coords[:, 1], coords[:, 0]] = explained_variance_clipped

    print(f"Non-zero voxels: {np.sum(volume_3d > 0)}")

    # =============================================================================
    # Create pycortex Volume and visualize
    # =============================================================================
    pycortex_subject = subject_map[subject]

    volume = cortex.Volume(volume_3d, pycortex_subject, args.transform,
                        vmin=0, vmax=max_ev, cmap='hot')

    fig = cortex.quickshow(volume, with_curvature=True, with_sulci=True)

    # Add text annotation
    fig.text(0.5, 0.98, f'Encoding model accuracy (explained variance, r²) - {subject}', 
         fontsize=16, ha='center', va='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.text(0.02, 0.02, 'Volume space data projected to cortical surface', 
            fontsize=12, ha='left', va='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # =============================================================================
    # Save figure
    # =============================================================================
    save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
        'train_dataset-things_fmri_1', 'model-vit_b_32', 'encoding_models_accuracy')
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    save_name = f'surface_correlation_{subject}'
    fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
    
    print(f"Saved: {save_dir}/{save_name}.png")
    
    plt.close(fig)