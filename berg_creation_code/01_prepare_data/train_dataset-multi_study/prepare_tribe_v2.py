"""Prepare metadata for TRIBE v2 fMRI encoding model in BERG.

This script downloads the Glasser HCP-MMP1.0 parcellation, resamples it
to fsaverage5 (20,484 cortical vertices), and creates the metadata .npy file
required by the BERG model implementation.

The Glasser parcellation provides 180 bilateral cortical regions (360 hemisphere-
specific), which are used for ROI-based selection of predicted brain responses.


Output Files Created
────────────────────
{berg_dir}/encoding_models/modality-fmri/train_dataset-multi_study/
    model-tribe_v2/metadata/metadata_average.npy

Metadata structure:
    'fmri':
        subject_id           : str      - Subject identifier ('average')
        n_vertices           : int      - Total cortical vertices (20484)
        n_vertices_lh        : int      - Left hemisphere vertices (10242)
        n_vertices_rh        : int      - Right hemisphere vertices (10242)
        surface_mesh         : str      - Surface mesh name ('fsaverage5')
        output_frequency_hz  : float    - Temporal resolution of predictions (1.0 Hz)

    'roi':
        parcellation         : str      - Parcellation name ('Glasser_HCP-MMP1.0')
        roi_labels           : (180,)   - Bilateral ROI names (e.g., 'V1', 'V2', 'FFC')
        roi_assignments      : (20484,) - ROI index per vertex (-1 = medial wall)
        roi_index            : dict     - Mapping from ROI name to integer index in roi_assignments

Requirements
------------
    nilearn, nibabel, scipy, numpy, requests
"""

import argparse
import os
import tempfile

import nibabel as nib
import numpy as np
import requests
from scipy.spatial import cKDTree
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

# Glasser HCP-MMP1.0 annot files projected onto fsaverage (from Figshare)
# Source: https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446
GLASSER_ANNOT_URLS = {
    "lh": "https://ndownloader.figshare.com/files/5528816",
    "rh": "https://ndownloader.figshare.com/files/5528819",
}

N_VERTICES_FSAVERAGE5 = 10242  # per hemisphere
N_VERTICES_TOTAL = 20484  # both hemispheres


# =============================================================================
# Download Glasser parcellation
# =============================================================================


def download_glasser_annot(cache_dir):
    """Download Glasser HCP-MMP1.0 annot files for fsaverage.

    Downloads the left and right hemisphere annotation files from Figshare
    if they are not already cached locally.

    Parameters
    ----------
    cache_dir : str
        Directory to cache downloaded files.

    Returns
    -------
    dict
        Paths to the downloaded annot files, keyed by 'lh' and 'rh'.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}

    for hemi, url in GLASSER_ANNOT_URLS.items():
        filename = f"{hemi}.HCPMMP1.annot"
        filepath = os.path.join(cache_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"  Using cached {filename}")
        else:
            print(f"  Downloading {filename} from Figshare...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=128 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"  Saved to {filepath}")

        paths[hemi] = filepath

    return paths


# =============================================================================
# Resample parcellation from fsaverage to fsaverage5
# =============================================================================


def load_sphere_coordinates(mesh_name, hemi):
    """Load sphere coordinates for a FreeSurfer surface mesh.

    Uses nilearn to fetch the fsaverage mesh data, then loads the sphere
    surface coordinates using nibabel.

    Parameters
    ----------
    mesh_name : str
        FreeSurfer mesh name ('fsaverage' or 'fsaverage5').
    hemi : str
        Hemisphere ('left' or 'right').

    Returns
    -------
    np.ndarray
        Vertex coordinates on the sphere, shape (n_vertices, 3).
    """
    from nilearn import datasets

    mesh = datasets.fetch_surf_fsaverage(mesh_name)
    sphere_path = str(mesh[f"sphere_{hemi}"])

    # Handle .gii, .gii.gz, and FreeSurfer binary formats
    if sphere_path.endswith((".gii", ".gii.gz")):
        img = nib.load(sphere_path)
        coords = img.darrays[0].data
    else:
        coords, _ = nib.freesurfer.read_geometry(sphere_path)

    return coords


def resample_labels_to_fsaverage5(labels_fsaverage, hemi):
    """Resample vertex labels from fsaverage to fsaverage5 via nearest-neighbor.

    For each vertex on the fsaverage5 sphere, finds the nearest vertex on
    the fsaverage sphere and assigns its label. This is the standard approach
    for resampling surface parcellations across FreeSurfer mesh resolutions.

    Parameters
    ----------
    labels_fsaverage : np.ndarray
        Integer label array for fsaverage vertices, shape (163842,).
    hemi : str
        Hemisphere ('left' or 'right').

    Returns
    -------
    np.ndarray
        Integer label array for fsaverage5 vertices, shape (10242,).

    Example
    -------
    If fsaverage vertex 50321 has label 5 (corresponding to 'V2'), and
    fsaverage5 vertex 312 is nearest to fsaverage vertex 50321 on the
    sphere, then the output assigns label 5 to fsaverage5 vertex 312.
    """
    # Load sphere coordinates for source (fsaverage) and target (fsaverage5)
    coords_fs = load_sphere_coordinates("fsaverage", hemi)
    coords_fs5 = load_sphere_coordinates("fsaverage5", hemi)

    # Build KD-tree on fsaverage sphere and query with fsaverage5 vertices
    tree = cKDTree(coords_fs)
    _, nearest_indices = tree.query(coords_fs5)

    # Map labels via nearest neighbor
    labels_fs5 = labels_fsaverage[nearest_indices]

    return labels_fs5


def get_glasser_parcellation_fsaverage5(cache_dir):
    """Get the Glasser HCP-MMP1.0 parcellation on fsaverage5.

    Downloads the Glasser annot files for fsaverage, reads them with nibabel,
    resamples to fsaverage5 using nearest-neighbor on the sphere, and returns
    bilateral ROI labels and assignments.

    The annot files contain 181 labels per hemisphere (180 parcels + 1 for the
    medial wall labeled '???'). The medial wall is assigned index -1 in the
    output. ROI names are made bilateral by stripping hemisphere prefixes,
    giving 180 unique region names.

    Parameters
    ----------
    cache_dir : str
        Directory to cache downloaded annot files.

    Returns
    -------
    roi_labels : np.ndarray
        Array of 180 bilateral ROI name strings, e.g. ['V1', 'MST', 'V6', ...].
    roi_assignments : np.ndarray
        Integer array of shape (20484,) mapping each vertex to an index in
        roi_labels. Vertices on the medial wall are assigned -1.
        First 10242 entries are left hemisphere, last 10242 are right.
    """
    annot_paths = download_glasser_annot(cache_dir)

    all_labels_fs5 = []
    roi_names_bilateral = None

    for hemi_short, hemi_long in [("lh", "left"), ("rh", "right")]:
        # Read fsaverage annot file
        # labels: (163842,) int array of label IDs
        # ctab: color table
        # names: list of label name bytes
        labels_fs, ctab, names_bytes = nib.freesurfer.read_annot(
            annot_paths[hemi_short]
        )

        # Decode label names and strip hemisphere prefix if present
        # Names come as e.g. b'L_V1_ROI' or b'R_V1_ROI' depending on hemisphere
        # We strip the prefix and suffix to get bilateral names
        names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names_bytes]

        # Identify the medial wall label (usually '???' or 'Unknown')
        medial_wall_names = {"???", "Unknown", "unknown"}
        parcel_names = []
        medial_wall_idx = None
        for i, name in enumerate(names):
            if name in medial_wall_names:
                medial_wall_idx = i
            else:
                # Strip hemisphere prefix: 'L_V1_ROI' -> 'V1'
                clean = name
                if clean.startswith(("L_", "R_")):
                    clean = clean[2:]
                if clean.endswith("_ROI"):
                    clean = clean[:-4]
                parcel_names.append(clean)

        if roi_names_bilateral is None:
            roi_names_bilateral = np.array(parcel_names)
        # Verify both hemispheres have the same parcels
        assert np.array_equal(
            roi_names_bilateral, np.array(parcel_names)
        ), "Left and right hemisphere parcel names do not match"

        # Resample labels from fsaverage to fsaverage5
        labels_fs5 = resample_labels_to_fsaverage5(labels_fs, hemi_long)

        # Remap: annot index 0 = medial wall ('???') -> -1
        # annot indices 1..180 = parcels -> 0..179
        remapped = np.full(labels_fs5.shape, -1, dtype=np.int32)
        parcel_counter = 0
        for i, name in enumerate(names):
            name_str = name.decode("utf-8") if isinstance(name, bytes) else name
            if name_str in medial_wall_names:
                continue
            mask = labels_fs5 == i
            remapped[mask] = parcel_counter
            parcel_counter += 1

        all_labels_fs5.append(remapped)

    # Concatenate hemispheres: left first, then right
    roi_assignments = np.concatenate(all_labels_fs5)

    return roi_names_bilateral, roi_assignments


# =============================================================================
# Create metadata
# =============================================================================


def create_tribe_v2_metadata(berg_dir):
    """Create and save the TRIBE v2 metadata file.

    Generates a comprehensive metadata dictionary containing surface mesh info,
    ROI parcellation data (Glasser HCP-MMP1.0), and encoding model details.
    Saves the result as a .npy file in the standard BERG directory structure.

    Parameters
    ----------
    berg_dir : str
        Root directory of the BERG framework.
    """
    print("=" * 60)
    print("TRIBE v2 Metadata Preparation")
    print("=" * 60)

    # Output directory
    output_dir = os.path.join(
        berg_dir,
        "encoding_models",
        "modality-fmri",
        "train_dataset-multi_study",
        "model-tribe_v2",
        "metadata",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Cache directory for downloaded files
    cache_dir = os.path.join(berg_dir, "cache", "tribe_v2", "parcellation")

    # -------------------------------------------------------------------------
    # Step 1: Get Glasser parcellation on fsaverage5
    # -------------------------------------------------------------------------
    print("\nStep 1: Obtaining Glasser HCP-MMP1.0 parcellation on fsaverage5")
    roi_labels, roi_assignments = get_glasser_parcellation_fsaverage5(cache_dir)

    # Build roi_index: mapping from ROI name to its integer index in roi_assignments
    roi_index = {label: int(i) for i, label in enumerate(roi_labels)}

    n_assigned = int(np.sum(roi_assignments >= 0))
    n_medial_wall = int(np.sum(roi_assignments == -1))
    print(f"  Total vertices: {N_VERTICES_TOTAL}")
    print(f"  Assigned to ROIs: {n_assigned}")
    print(f"  Medial wall (unassigned): {n_medial_wall}")
    print(f"  Number of bilateral ROIs: {len(roi_labels)}")

    # -------------------------------------------------------------------------
    # Step 2: Build metadata dictionary
    # -------------------------------------------------------------------------
    print("\nStep 2: Building metadata dictionary")

    metadata = {
        "fmri": {
            "subject_id": "average",
            "n_vertices": N_VERTICES_TOTAL,
            "n_vertices_lh": N_VERTICES_FSAVERAGE5,
            "n_vertices_rh": N_VERTICES_FSAVERAGE5,
            "surface_mesh": "fsaverage5",
            "output_frequency_hz": 1.0,
        },
        "roi": {
            "parcellation": "Glasser_HCP-MMP1.0",
            "roi_labels": roi_labels,
            "roi_assignments": roi_assignments,
            "roi_index": roi_index,
        },
    }

    # -------------------------------------------------------------------------
    # Step 3: Save metadata
    # -------------------------------------------------------------------------
    output_path = os.path.join(output_dir, "metadata_average.npy")
    np.save(output_path, metadata)
    print(f"\nMetadata saved to: {output_path}")

    # Print summary of ROI indices
    print("\nROI index mapping (first 20):")
    for name, idx in list(roi_index.items())[:20]:
        n_verts = int(np.sum(roi_assignments == idx))
        print(f"  {name:12s}: index {idx:3d}  ({n_verts:5d} vertices)")
    if len(roi_index) > 20:
        print(f"  ... and {len(roi_index) - 20} more ROIs")

    print("\nDone.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare metadata for TRIBE v2 encoding model in BERG."
    )
    parser.add_argument(
        "--berg_dir",
        required=True,
        type=str,
        help="Root directory of the BERG framework.",
    )
    args = parser.parse_args()

    print(">>> TRIBE v2 Metadata Preparation <<<")
    print("\nInput arguments:")
    for key, val in vars(args).items():
        print(f"  {key:16s} {val}")

    create_tribe_v2_metadata(args.berg_dir)