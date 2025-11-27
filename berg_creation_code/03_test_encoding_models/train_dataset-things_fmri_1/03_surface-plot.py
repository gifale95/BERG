"""Plot encoding model prediction accuracy on cortical surfaces using pycortex.
Visualizes voxel-wise correlation results on inflated cortical surfaces.

Parameters
----------
subjects : list of str
    List of subjects to analyze (e.g., 'sub-01 sub-02 sub-03').
model : str
    Name of the used encoding model.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
only_cls : str
    If we should only use CLS token or all patches ('True' or 'False').
regression : str
    Type of regression used ('ridge' or 'linear').
pycortex_filestore : str
    Path to pycortex filestore containing subject surfaces.
transform : str
    Name of the transform to use (default: 'align_auto').
vmin : float
    Minimum value for colormap (default: -0.5).
vmax : float
    Maximum value for colormap (default: 0.9).

Example usage:
python berg_creation_code/03_test_encoding_models/train_dataset-things_fmri_1/03_surface-plot.py \
    --subjects sub-01 sub-02 sub-03 \
    --berg_dir '/Volumes/Extreme SSD/brain-encoding-response-generator' \
    --pycortex_filestore '/Volumes/Extreme SSD/Datasets/THINGS/pycortex_filestore' \
    --only_cls True \
    --regression linear \
    --model vit_b_32 \
    --transform align_auto \
    --vmin -0.5 \
    --vmax 0.9
"""

import argparse
import os
import numpy as np
import cortex
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=str, nargs='+', required=True,
                   help="List of subject IDs (e.g., 'sub-01 sub-02 sub-03')")
parser.add_argument('--model', required=True, choices=["vit_b_32"],
                   help="Selecting which model to use")
parser.add_argument('--only_cls', required=True, choices=["True", "False"],
                    help='If we should only use CLS token or all patches')
parser.add_argument('--regression', required=True, choices=["ridge", "linear"],
                   help="Select type of regression")
parser.add_argument('--berg_dir', required=True, type=str)
parser.add_argument('--pycortex_filestore', required=True, type=str,
                   help="Path to pycortex filestore")
parser.add_argument('--transform', type=str, default='align_auto',
                   help="Transform name (default: align_auto)")
parser.add_argument('--vmin', type=float, default=-0.5,
                   help="Minimum value for colormap (default: -0.5)")
parser.add_argument('--vmax', type=float, default=0.9,
                   help="Maximum value for colormap (default: 0.9)")
args = parser.parse_args()

args.only_cls = args.only_cls == "True"
cls_suffix = 'cls' if args.only_cls else 'all'

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
        'train_dataset-things_fmri_1', f'model-{args.model}', 'metadata')
    file_name = f'metadata_{subject}.npy'
    metadata_file = os.path.join(metadata_dir, file_name)
    
    metadata = np.load(metadata_file, allow_pickle=True).item()
    correlations = metadata['encoding_model']['correlation_results']
    coords = metadata['fmri']['voxel_coords']
    
    print(f"Correlation shape: {correlations.shape}")
    print(f"Range: [{correlations.min():.4f}, {correlations.max():.4f}]")
    
    # =============================================================================
    # Create 3D volume
    # =============================================================================
    # Calculate volume shape from coordinate ranges (Z, Y, X for pycortex)
    vol_shape = (coords[:, 2].max() + 1, 
                 coords[:, 1].max() + 1, 
                 coords[:, 0].max() + 1)
    print(f"Volume shape (Z, Y, X): {vol_shape}")
    
    # Create empty volume filled with NaN
    volume_3d = np.full(vol_shape, np.nan)
    
    # Place correlation values at their coordinates
    volume_3d[coords[:, 2], coords[:, 1], coords[:, 0]] = correlations
    
    print(f"Non-NaN voxels: {np.sum(~np.isnan(volume_3d))}")
    
    # =============================================================================
    # Create pycortex Volume and visualize
    # =============================================================================
    pycortex_subject = subject_map[subject]
    
    volume = cortex.Volume(volume_3d, pycortex_subject, args.transform,
                           vmin=args.vmin, vmax=args.vmax, cmap='RdBu_r')
    
    fig = cortex.quickshow(volume, with_curvature=True, with_sulci=True)
    
    # Add text annotation
    fig.text(0.02, 0.98, 'Volume space data projected to cortical surface', 
             fontsize=14, ha='left', va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =============================================================================
    # Save figure
    # =============================================================================
    save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
        'train_dataset-things_fmri_1', f'model-{args.model}', 'encoding_models_accuracy')
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    save_name = f'surface_correlation_{args.regression}_{cls_suffix}_{args.model}_{subject}'
    fig.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
    
    print(f"Saved: {save_dir}/{save_name}.png")
    
    plt.close(fig)
