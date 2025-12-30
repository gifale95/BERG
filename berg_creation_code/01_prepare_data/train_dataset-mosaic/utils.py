"""
MOSAIC Dataset Metadata Preparation
====================================
Downloads and processes metadata for MOSAIC fMRI datasets, including:
- Stimulus information and train/test splits
- Noise ceiling values mapped to model vertex spaces
- ROI masks for brain region analysis
"""

import h5py
import numpy as np
import mosaic
import os
from pathlib import Path
from mosaic.stiminfo import get_stiminfo
from mosaic.participantinfo import get_participantinfo
from mosaic.models.transforms import SelectROIs
from mosaic.constants import region_of_interest_labels
from mosaic.constants import subject_id_to_file_mapping


# Dataset configuration
DATASETS = {
    "BOLD5000": {"long_name": "BOLD5000", "n_subjects": 4},
    "deeprecon": {"long_name": "deeprecon", "n_subjects": 3},
    "GOD": {"long_name": "GenericObjectDecoding", "n_subjects": 5},
    "NSD": {"long_name": "NaturalScenesDataset", "n_subjects": 8},
    "THINGS": {"long_name": "THINGS", "n_subjects": 3},
    "BMD": {"long_name": "BOLDMomentsDataset", "n_subjects": 10},
    "NOD": {"long_name": "NaturalObjectDataset", "n_subjects": 30},
    "HAD": {"long_name": "HumanActionsDataset", "n_subjects": 30},
}

def _sub_num(sub_str: str) -> int:
    """Extract subject number for sorting."""
    return int(sub_str.split("-")[1])

def download_metadata(save_path):
    """
    Download and structure subject-wise metadata from MOSAIC datasets.
    
    Creates per-subject metadata files containing stimulus info, train/test splits,
    and participant information.
    
    Steps:
    1. Query MOSAIC for stimulus and participant metadata
    2. Extract dataset-wide stimulus information (filenames, sources, aliases)
    3. Determine train/test splits from stimulus metadata
    4. Save subject-specific metadata as .npy files
    
    Output structure:
        {save_path}/
        ├── BOLD5000/
        │   ├── sub-01.npy
        │   └── ...
        └── NSD/
            └── ...
    
    Each file contains:
        {
            "fmri": {
                "participant_id": "sub-01",
                "age": ...,
                "filenames": array([...]),       # All stimulus filenames
                "train_idx": array([...]),       # Train trial indices
                "test_idx": array([...]),        # Test trial indices
                "reps": array([...]),            # Subject repetition counts
                ...
            }
        }
    """
    out_root = Path(save_path)
    out_root.mkdir(exist_ok=True)

    for short_id, info in DATASETS.items():
        if short_id == "THINGS":
            long_name = "THINGS_fmri"  # Inconsistency in MOSAIC
        else:
            long_name = info["long_name"]
        print(f"[{short_id}] Exporting subject-wise metadata...")
        
        # Get dataset metadata
        stim = get_stiminfo(dataset_name=short_id)
        part = get_participantinfo(dataset_name=long_name)
        subject_ids = sorted(part["participant_id"].astype(str).unique(), key=_sub_num)

        # Extract stimulus information
        filenames = stim["filename"].astype(str).to_numpy()
        alias = stim["alias"].astype(str).to_numpy()
        source = stim["source"].astype(str).to_numpy()

        # Get train/test split if available
        if "test_train" in stim.columns:
            split = stim["test_train"].astype(str).values
            train_idx = np.where(split == "train")[0]
            test_idx = np.where(split == "test")[0]
            train_filenames = filenames[train_idx]
            test_filenames = filenames[test_idx]
        else:
            train_idx = test_idx = train_filenames = test_filenames = None

        # Save per-subject metadata
        for subj in subject_ids:
            subj_row = part.loc[part["participant_id"] == subj].to_dict(orient="records")[0]
            reps_col = f"{subj}_reps"
            reps = stim[reps_col].to_numpy() if reps_col in stim.columns else None

            meta = {
                "fmri": {
                    **subj_row,
                    "filenames": filenames,
                    "alias": alias,
                    "source": source,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "train_filenames": train_filenames,
                    "test_filenames": test_filenames,
                    "reps": reps,
                }
            }

            ds_dir = out_root / short_id
            ds_dir.mkdir(exist_ok=True)
            np.save(ds_dir / f"{subj}.npy", meta, allow_pickle=True)

        print(f"Saved {len(subject_ids)} subjects to {ds_dir}/")


def download_noise_ceilings(output_dir):
    """
    Download all available noise ceiling variants from MOSAIC datasets.
    
    Noise ceilings represent the maximum predictable variance in brain responses
    and are used to evaluate encoding model performance.
    
    Available variants by dataset:
    - All datasets: test_n-avg, test_n-1
    - NSD, BMD, NOD, deeprecon: train_n-avg, train_n-1
    - NSD, deeprecon: artificial_n-avg, artificial_n-1
    
    Output:
        {output_dir}/
        ├── BOLD5000_sub-01_test_n-avg_noise_ceiling.npy
        ├── BOLD5000_sub-01_test_n-1_noise_ceiling.npy
        ├── NSD_sub-01_test_n-avg_noise_ceiling.npy
        ├── NSD_sub-01_train_n-avg_noise_ceiling.npy
        └── ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for short_id, info in DATASETS.items():
        dataset_name = info["long_name"]
        n_subjects = info["n_subjects"]
        print(f"{short_id}:")

        for subj in range(1, n_subjects + 1):
            try:
                # Load subject data
                dataset = mosaic.load(
                    names_and_subjects={dataset_name: [subj]},
                    folder="./MOSAIC",
                    parse_betas=False
                )

                # Get HDF5 file path
                hdf5_filename = subject_id_to_file_mapping[dataset_name][subj]
                hdf5_path = f"./MOSAIC/{hdf5_filename}"

                # Extract all available noise ceiling variants
                with h5py.File(hdf5_path, "r") as f:
                    if 'noiseceilings' not in f:
                        print(f"  sub-{subj:02d}: No noise ceilings group")
                        continue
                    
                    saved_count = 0
                    
                    # Try all possible variants
                    for phase in ["test", "train", "artificial"]:
                        for method in ["n-avg", "n-1"]:
                            nc_key = f"sub-{subj:02d}_{short_id}_phase-{phase}_{method}_noiseceiling"
                            
                            if nc_key in f['noiseceilings']:
                                nc_data = f['noiseceilings'][nc_key][:].astype(np.float32)
                                output_path = output_dir / f"{short_id}_sub-{subj:02d}_{phase}_{method}_noise_ceiling.npy"
                                np.save(output_path, nc_data)
                                saved_count += 1
                    
                    if saved_count > 0:
                        print(f"  sub-{subj:02d}: Saved {saved_count} noise ceiling variant(s)")
                    else:
                        print(f"  sub-{subj:02d}: No noise ceiling variants found")

                # Clean up HDF5 file
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)

            except Exception as e:
                print(f"  sub-{subj:02d}: Failed ({type(e).__name__}: {str(e)})")

    print(f"All files saved to: {output_dir}/")


def add_noise_ceilings_to_metadata(metadata_dir, noise_ceiling_dir):
    """
    Add noise ceiling values to metadata in their original space.
    
    Stores all available noise ceiling variants for each subject.
    Users can map to model prediction spaces (visual/all) when needed using vertex mappings.
    
    Available noise ceiling variants:
    - test_n-avg (all datasets)
    - test_n-1 (all datasets)
    - train_n-avg (only NSD, BMD, NOD, deeprecon)
    - train_n-1 (only NSD, BMD, NOD, deeprecon)
    - artificial_n-avg (only NSD, deeprecon)
    - artificial_n-1 (only NSD, deeprecon)
    
    Each noise ceiling is stored as a float32 array of shape (91282,) in full fsLR32k space.
    Only variants that exist for a given subject are stored.
    """
    metadata_dir = Path(metadata_dir)
    noise_ceiling_dir = Path(noise_ceiling_dir)
    
    # Define all possible noise ceiling variants
    nc_variants = [
        "test_n-avg",
        "test_n-1",
        "train_n-avg",
        "train_n-1",
        "artificial_n-avg",
        "artificial_n-1"
    ]
    
    # Process all subjects
    for dataset_dir in sorted(metadata_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        
        for meta_file in sorted(dataset_dir.glob("sub-*.npy")):
            subject_id = meta_file.stem
            subj_num = int(subject_id.split('-')[1])
            meta = np.load(meta_file, allow_pickle=True).item()
            
            # Initialize encoding_models section
            if "encoding_models" not in meta:
                meta["encoding_models"] = {}
            
            # Try to load noise ceiling from intermediate files
            nc_found = False
            
            for variant in nc_variants:
                # Look for intermediate noise ceiling file (from download step)
                nc_file = noise_ceiling_dir / f"{dataset_name}_{subject_id}_{variant}_noise_ceiling.npy"
                
                if nc_file.exists():
                    nc_data = np.load(nc_file).astype(np.float32)
                    meta["encoding_models"][f"{variant}_noiseceiling"] = nc_data
                    nc_found = True
            
            if nc_found:
                print(f"Added noise ceilings: {dataset_name:8s}, {subject_id}")
            else:
                print(f"No noise ceilings found: {dataset_name:8s}, {subject_id}")
            
            np.save(meta_file, meta, allow_pickle=True)


def add_roi_indices_to_metadata(metadata_dir):
    """
    Add ROI vertex indices from the Glasser atlas to metadata.
    
    Stores original vertex indices (in full fsLR32k space) for each ROI,
    allowing users to slice model predictions after expanding to full brain space.
    
    Steps:
    1. Get all cortical ROIs from Glasser atlas (360 regions)
    2. For each ROI, extract vertex indices in full 91k space
    3. Store indices in metadata under 'fmri'/'roi'
    
    Note: These are the raw indices from MOSAIC's Glasser atlas. To use with
    model predictions, expand predictions to 91k space first using the vertex
    mapping (GlasserGroups 1-22 or 1-5).
    """
    
    # Get all cortical ROIs
    all_rois = sorted([roi for roi in region_of_interest_labels.keys() 
                       if roi and (roi.startswith('L_') or roi.startswith('R_'))])
    
    # Build dictionary of ROI indices
    roi_indices_dict = {}
    
    for roi in all_rois:
        try:
            roi_selector = SelectROIs(selected_rois=[roi])
            roi_indices = np.array(roi_selector.selected_roi_indices)
            roi_indices_dict[roi] = roi_indices
                    
        except Exception as e:
            print(f"Error processing {roi}: {e}")
    
    print(f"Total ROIs extracted: {len(roi_indices_dict)}")
    
    # Add to all metadata files
    metadata_dir = Path(metadata_dir)
    total_updated = 0
    
    print("Updating metadata files...")
    for dataset_dir in sorted(metadata_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        for meta_file in sorted(dataset_dir.glob("sub-*.npy")):
            meta = np.load(meta_file, allow_pickle=True).item()
            
            meta["fmri"]["roi"] = roi_indices_dict
            
            np.save(meta_file, meta, allow_pickle=True)
            total_updated += 1
    
    print(f"Updated {total_updated} metadata files")


def add_vertex_mappings_to_metadata(metadata_dir):
    """
    Add vertex mapping arrays to metadata for expanding model predictions to full HCP space.
    
    Model predictions are in reduced vertex spaces (visual or all cortex), while noise ceilings
    and ROI indices are defined in full HCP grayordinate space (91,282 vertices). These mappings
    allow expansion: predictions_91k[vertex_mapping] = predictions_model.
    
    Adds two mappings to metadata under 'encoding_models':
    - vertex_mapping_visual: (7,831,) array mapping visual cortex predictions to 91k space
    - vertex_mapping_all: (57,051,) array mapping full cortex predictions to 91k space
    
    These correspond to:
    - Visual: GlasserGroups 1-5 (visual cortex)
    - All: GlasserGroups 1-22 (visual + sensorimotor + auditory + association cortex)
    """
    
    # Get vertex mappings for both model variants
    visual_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 6)])
    all_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 23)])
    
    vertex_mapping_visual = np.array(visual_selector.selected_roi_indices, dtype=np.int32)
    vertex_mapping_all = np.array(all_selector.selected_roi_indices, dtype=np.int32)
    
    print(f"Visual mapping: {len(vertex_mapping_visual)} vertices (range: {vertex_mapping_visual.min()}-{vertex_mapping_visual.max()})")
    print(f"All mapping: {len(vertex_mapping_all)} vertices (range: {vertex_mapping_all.min()}-{vertex_mapping_all.max()})")
    
    # Add to all metadata files
    metadata_dir = Path(metadata_dir)
    total_updated = 0
    
    print("Updating metadata files...")
    for dataset_dir in sorted(metadata_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        for meta_file in sorted(dataset_dir.glob("sub-*.npy")):
            meta = np.load(meta_file, allow_pickle=True).item()
            
            # Initialize encoding_models section if needed
            if "encoding_models" not in meta:
                meta["encoding_models"] = {}
            
            meta["encoding_models"]["vertex_mapping_visual"] = vertex_mapping_visual
            meta["encoding_models"]["vertex_mapping_all"] = vertex_mapping_all
            
            np.save(meta_file, meta, allow_pickle=True)
            total_updated += 1
    
    print(f"Updated {total_updated} metadata files")