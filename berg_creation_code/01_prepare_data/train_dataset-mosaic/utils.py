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


# Dataset configuration
DATASETS = {
    "BOLD5000": {"long_name": "BOLD5000", "n_subjects": 4},
    "deeprecon": {"long_name": "deeprecon", "n_subjects": 3},
    "GOD": {"long_name": "GenericObjectDecoding", "n_subjects": 5},
    "NSD": {"long_name": "NaturalScenesDataset", "n_subjects": 8},
    "THINGS": {"long_name": "THINGS_fmri", "n_subjects": 3},
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


def download_noise_ceilings(output_dir="./noise_ceilings"):
    """
    Download test noise ceiling values from MOSAIC datasets.
    
    Noise ceilings represent the maximum predictable variance in brain responses
    and are used to evaluate encoding model performance.
    
    Steps:
    1. Load each subject's data from MOSAIC
    2. Extract test-phase averaged noise ceiling from HDF5 file
    3. Save as float32 .npy file and clean up temporary HDF5 file
    4. Skip subjects without noise ceiling data
    
    Output:
        {output_dir}/
        ├── BOLD5000_sub-01_test_n-avg_noise_ceiling.npy
        ├── NSD_sub-01_test_n-avg_noise_ceiling.npy
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
                from mosaic.constants import subject_id_to_file_mapping
                hdf5_filename = subject_id_to_file_mapping[dataset_name][subj]
                hdf5_path = f"./MOSAIC/{hdf5_filename}"

                # Extract noise ceiling
                with h5py.File(hdf5_path, "r") as f:
                    if 'noiseceilings' in f:
                        nc_key = f"sub-{subj:02d}_{short_id}_phase-test_n-avg_noiseceiling"
                        
                        if nc_key in f['noiseceilings']:
                            nc_data = f['noiseceilings'][nc_key][:].astype(np.float32)
                            output_path = output_dir / f"{short_id}_sub-{subj:02d}_test_n-avg_noise_ceiling.npy"
                            np.save(output_path, nc_data)
                            print(f"Saved: {output_path.name}")
                        else:
                            print(f"Skipped sub-{subj:02d}: no test n-avg")

                # Clean up HDF5 file
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)

            except Exception as e:
                print(f"Failed sub-{subj:02d}: {type(e).__name__}")

    print(f"All files saved to: {output_dir}/")


def add_noise_ceilings_to_metadata(metadata_dir, noise_ceiling_dir):
    """
    Map noise ceilings from full brain space (91k) to model prediction spaces.
    
    MOSAIC models predict subsets of brain vertices:
    - Visual space: 7,831 vertices (GlasserGroups 1-5)
    - All space: 57,051 vertices (GlasserGroups 1-22)
    
    Steps:
    1. Get vertex indices for visual and all spaces using SelectROIs
    2. For each subject, load noise ceiling in full 91k space
    3. Extract values at model prediction vertices only
    4. Save to metadata under 'encoding_models' key
    5. For missing noise ceilings, fill with NaN 
    
    Why: Model predictions are sized (7831,) or (57051,), so noise ceilings
         must match to enable direct comparison.
    """
    # Get model vertex spaces
    visual_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 6)])
    all_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 23)])
    
    visual_indices = np.array(visual_selector.selected_roi_indices)
    all_indices = np.array(all_selector.selected_roi_indices)
    
    metadata_dir = Path(metadata_dir)
    noise_ceiling_dir = Path(noise_ceiling_dir)
    
    # Process all subjects
    for dataset_dir in sorted(metadata_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        
        for meta_file in sorted(dataset_dir.glob("sub-*.npy")):
            subject_id = meta_file.stem
            meta = np.load(meta_file, allow_pickle=True).item()
            
            # Look for noise ceiling file
            nc_file = noise_ceiling_dir / f"{dataset_name}_{subject_id}_test_n-avg_noise_ceiling.npy"
            
            if nc_file.exists():
                # Map to model spaces
                nc_full = np.load(nc_file).astype(np.float32)
                nc_visual = nc_full[visual_indices]
                nc_all = nc_full[all_indices]
                
                print(f"Dataset: {dataset_name:8s}, Subject: {subject_id}, "
                      f"NC: {nc_full.shape[0]:5d} → Visual: {nc_visual.shape[0]:5d}, All: {nc_all.shape[0]:5d}")
            else:
                # No noise ceiling - use NaN
                nc_visual = np.full(len(visual_indices), np.nan, dtype=np.float32)
                nc_all = np.full(len(all_indices), np.nan, dtype=np.float32)
                
                print(f"Skipped: {dataset_name:8s}, Subject: {subject_id} (no noise ceiling)")
            
            # Add to metadata
            if "encoding_models" not in meta:
                meta["encoding_models"] = {}
            
            meta["encoding_models"]["test_n-avg_noiseceiling_visual_vertices"] = nc_visual
            meta["encoding_models"]["test_n-avg_noiseceiling_all_vertices"] = nc_all
            
            np.save(meta_file, meta, allow_pickle=True)


def add_roi_masks_to_metadata(metadata_dir):
    """
    Add binary ROI masks for extracting region-specific predictions.
    
    Creates masks for each brain region (ROI) in the Glasser atlas, allowing
    extraction of predictions for specific areas.
    
    Steps:
    1. Get all cortical ROIs from Glasser atlas (360 regions)
    2. For each ROI, create binary mask in visual (7831,) and all (57051,) spaces
    3. Mask[i] = True if vertex i belongs to this ROI
    4. Store masks in metadata under 'fmri'/'roi_visual_vertices' and 'roi_all_vertices'
    5. Skip ROIs not present in visual space (only add to all space)
    
    """
    
    # Get model vertex spaces
    visual_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 6)])
    all_selector = SelectROIs(selected_rois=[f"GlasserGroup_{i}" for i in range(1, 23)])
    
    visual_space_indices = np.array(visual_selector.selected_roi_indices)
    all_space_indices = np.array(all_selector.selected_roi_indices)
    
    # Get all cortical ROIs
    all_rois = sorted([roi for roi in region_of_interest_labels.keys() 
                       if roi and (roi.startswith('L_') or roi.startswith('R_'))])
    
    # Build masks for each ROI
    roi_masks_visual = {}
    roi_masks_all = {}
    rois_in_all_not_visual = []
    
    for roi in all_rois:
        try:
            roi_selector = SelectROIs(selected_rois=[roi])
            roi_indices = np.array(roi_selector.selected_roi_indices)
            
            # Create mask for visual space
            mask_visual = np.isin(visual_space_indices, roi_indices)
            if mask_visual.any():
                roi_masks_visual[roi] = mask_visual
            
            # Create mask for all space
            mask_all = np.isin(all_space_indices, roi_indices)
            if mask_all.any():
                roi_masks_all[roi] = mask_all
                
                if not mask_visual.any():
                    rois_in_all_not_visual.append(roi)
                    
        except Exception as e:
            print(f"Error processing {roi}: {e}")
    
    print(f"ROIs in visual space: {len(roi_masks_visual)}")
    print(f"ROIs in all space: {len(roi_masks_all)}")
    print(f"ROIs in 'all' but NOT in 'visual': {len(rois_in_all_not_visual)}")
    if rois_in_all_not_visual:
        print("  " + ", ".join(rois_in_all_not_visual[:10]))
        if len(rois_in_all_not_visual) > 10:
            print(f"  ... and {len(rois_in_all_not_visual) - 10} more")
    
    # Add to all metadata files
    metadata_dir = Path(metadata_dir)
    total_updated = 0
    
    print("Updating metadata files...")
    for dataset_dir in sorted(metadata_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        for meta_file in sorted(dataset_dir.glob("sub-*.npy")):
            meta = np.load(meta_file, allow_pickle=True).item()
            
            meta["fmri"]["roi_visual_vertices"] = roi_masks_visual
            meta["fmri"]["roi_all_vertices"] = roi_masks_all
            
            np.save(meta_file, meta, allow_pickle=True)
            total_updated += 1
    