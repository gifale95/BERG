import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# Extract Mouse Metadata
# =============================================================================

def extract_mouse_metadata(data_path, output_path):
    """Extract metadata for each (session, scan_idx) combination.
    
    Process mouse calcium imaging dataset to create metadata files linking
    neural units to their anatomical properties, orientation/direction tuning
    characteristics, and encoding model performance metrics. Each metadata file
    corresponds to one recording session and contains information for all units
    recorded in that session.

    
    Parameters
    ----------
    data_path : Path
        Path to MOUSE dataset directory containing subdirectories:
        - anatomy/units.csv : Anatomical information per unit
        - anatomy/metadata.csv : Session-level metadata
        - performance/units.csv : Model performance metrics
        - ori_dir_tuning/units.csv : Orientation/direction tuning properties
    output_path : Path
        Path to output metadata directory.
        
    Output Files
    ------------
    session{session}_scan{scan}_metadata.npy : Metadata for one recording session
        Contains 'calcium' and 'encoding_model' dictionaries with all unit
        properties for that session.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all data files
    anatomy_df = pd.read_csv(data_path / 'anatomy' / 'units.csv')
    metadata_df = pd.read_csv(data_path / 'anatomy' / 'metadata.csv')
    performance_df = pd.read_csv(data_path / 'performance' / 'units.csv')
    ori_dir_df = pd.read_csv(data_path / 'ori_dir_tuning' / 'units.csv')
    
    # Get all unique (session, scan_idx) combinations
    combinations = anatomy_df[['session', 'scan_idx']].drop_duplicates()
    
    for _, row in tqdm(combinations.iterrows(), total=len(combinations), desc="Processing sessions"):
        session = row['session']
        scan_idx = row['scan_idx']
        
        # Filter data for this combination
        anat_subset = anatomy_df[(anatomy_df['session'] == session) & 
                                  (anatomy_df['scan_idx'] == scan_idx)]
        perf_subset = performance_df[(performance_df['session'] == session) & 
                                      (performance_df['scan_idx'] == scan_idx)]
        ori_subset = ori_dir_df[(ori_dir_df['session'] == session) & 
                                 (ori_dir_df['scan_idx'] == scan_idx)]
        meta_subset = metadata_df[(metadata_df['session'] == session) & 
                                   (metadata_df['scan_idx'] == scan_idx)]
        
        # Check consistency
        unit_ids_anat = set(anat_subset['unit_id'])
        unit_ids_perf = set(perf_subset['unit_id'])
        unit_ids_ori = set(ori_subset['unit_id'])
        
        if unit_ids_anat != unit_ids_perf or unit_ids_anat != unit_ids_ori:
            raise ValueError(f"Session {session}, scan {scan_idx}: Mismatch in unit_ids across files")
        
        # Sort by unit_id for consistent ordering
        anat_subset = anat_subset.sort_values('unit_id').reset_index(drop=True)
        perf_subset = perf_subset.sort_values('unit_id').reset_index(drop=True)
        ori_subset = ori_subset.sort_values('unit_id').reset_index(drop=True)
        
        # Get animal_id
        animal_id = meta_subset['animal_id'].iloc[0]
        
        # Extract coordinates
        coordinates = np.column_stack([
            anat_subset['unit_x'].values,
            anat_subset['unit_y'].values,
            anat_subset['unit_z'].values
        ]).astype(np.float32)
        
        # Create ROI binary masks
        unique_rois = anat_subset['brain_area'].unique()
        roi_masks = {}
        for roi in unique_rois:
            roi_masks[roi] = (anat_subset['brain_area'] == roi).values.astype(np.int8)
        
        # Create field binary masks
        unique_fields = anat_subset['field'].unique()
        field_masks = {}
        for field in sorted(unique_fields):
            field_masks[f'field_{field}'] = (anat_subset['field'] == field).values.astype(np.int8)
        
        # Build metadata dictionary
        metadata = {
            'calcium_2p': {
                'session': int(session),
                'scan': int(scan_idx),
                'animal_id': int(animal_id),
                'unit_id': anat_subset['unit_id'].values.astype(np.int32),
                'coordinates': coordinates,
                'roi': roi_masks,
                'field_masks': field_masks,
                'OSI': ori_subset['OSI'].values.astype(np.float32),
                'DSI': ori_subset['DSI'].values.astype(np.float32),
                'gOSI': ori_subset['gOSI'].values.astype(np.float32),
                'gDSI': ori_subset['gDSI'].values.astype(np.float32),
                'pref_ori': ori_subset['pref_ori'].values.astype(np.float32),
                'pref_dir': ori_subset['pref_dir'].values.astype(np.float32),
            },
            'encoding_model': {
                'cc_abs': perf_subset['cc_abs'].values.astype(np.float32),
                'cc_max': perf_subset['cc_max'].values.astype(np.float32),
                'cc_norm': perf_subset['cc_norm'].values.astype(np.float32),
            }
        }
        
        # Save metadata
        output_file = output_path / f'session{session}_scan{scan_idx}_metadata.npy'
        np.save(output_file, metadata, allow_pickle=True)
    
    print(f"Saved {len(combinations)} metadata files to {output_path}")