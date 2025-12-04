import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# =============================================================================
# Split training and test data
# =============================================================================

def split_fmri_data(response_filepath, stimulus_filepath, output_dir, subject_id, batch_size):
    """Split fMRI data, compute session statistics, normalize, and save."""
    
    print(f"Loading stimulus metadata from: {stimulus_filepath}")
    stim_metadata = pd.read_csv(stimulus_filepath)
    
    all_sessions = stim_metadata['session'].values
    unique_sessions = np.unique(all_sessions)
    n_sessions = len(unique_sessions)
    
    print(f"Loading fMRI response data from: {response_filepath}")
    
    train_mask = stim_metadata['trial_type'] == 'train'
    test_mask = stim_metadata['trial_type'] == 'test'
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    train_sessions = stim_metadata[train_mask]['session'].values
    test_sessions = stim_metadata[test_mask]['session'].values
    
    print(f"Training trials: {len(train_indices)}")
    print(f"Test trials: {len(test_indices)}")
    
    with h5py.File(response_filepath, 'r') as f:
        response_data = f['ResponseData/block0_values']
        n_voxels, n_trials = response_data.shape
        
        print(f"Original data shape: ({n_voxels} voxels, {n_trials} trials)")
        print("Loading and transposing data...")
        all_data = response_data[:, :].T  # (trials, voxels)
        
        # Compute session statistics
        print(f"Computing session statistics ({n_sessions} sessions)...")
        session_means = np.zeros((n_sessions, n_voxels), dtype=np.float32)
        session_stds = np.zeros((n_sessions, n_voxels), dtype=np.float32)
        
        for i, session in enumerate(unique_sessions):
            session_mask = all_sessions == session
            session_data = all_data[session_mask, :]
            
            print(f"  Session {session}: {len(session_data)} trials")
            
            session_mean = np.mean(session_data, axis=0)
            session_std = np.std(session_data, axis=0)
            session_std = np.maximum(session_std, 1e-8)
            
            session_means[i, :] = session_mean
            session_stds[i, :] = session_std
        
        # Split data
        print("Splitting data...")
        train_data = all_data[train_indices, :].astype(np.float32)
        test_data = all_data[test_indices, :].astype(np.float32)
        
        # Normalize train data
        for i, session in enumerate(tqdm(unique_sessions, desc="Normalizing train")):
            session_mask = train_sessions == session
            session_indices = np.where(session_mask)[0]
            if len(session_indices) > 0:
                train_data[session_indices, :] = (
                    (train_data[session_indices, :] - session_means[i, :]) / session_stds[i, :]
                )
        
        # Normalize test data
        for i, session in enumerate(tqdm(unique_sessions, desc="Normalizing test")):
            session_mask = test_sessions == session
            session_indices = np.where(session_mask)[0]
            if len(session_indices) > 0:
                test_data[session_indices, :] = (
                    (test_data[session_indices, :] - session_means[i, :]) / session_stds[i, :]
                )
        
        # Save normalized data
        train_file = os.path.join(output_dir, f'fmri_{subject_id}_split-train.h5')
        test_file = os.path.join(output_dir, f'fmri_{subject_id}_split-test.h5')
        
        with h5py.File(train_file, 'w') as f_train:
            f_train.create_dataset('neural_data', data=train_data, dtype='float32')
        
        with h5py.File(test_file, 'w') as f_test:
            f_test.create_dataset('neural_data', data=test_data, dtype='float32')
    
    print(f"Training shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    return session_means, session_stds, unique_sessions

def create_averaged_test_data(test_filepath, stimulus_filepath, output_dir, subject_id, test_mask):
    """Create averaged test data across repeated presentations of the same stimulus.
    
    Parameters
    ----------
    test_filepath : str
        Path to the individual test trials HDF5 file.
    stimulus_filepath : str
        Path to the stimulus metadata CSV file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    test_mask : np.ndarray
        Boolean mask indicating test trials.
        
    Output Files
    ------------
    fmri_{subject}_split-test_averaged.h5 : (n_unique_test, 211339)
    """
    # Load stimulus metadata for test trials
    stim_metadata = pd.read_csv(stimulus_filepath)
    test_metadata = stim_metadata[test_mask]
    test_stimuli = test_metadata['stimulus'].values
    
    # Find unique stimuli
    unique_stimuli = np.unique(test_stimuli)
    n_unique = len(unique_stimuli)
    
    print(f"Unique test stimuli: {n_unique}")
    
    # Load test data
    with h5py.File(test_filepath, 'r') as f:
        test_data = f['neural_data']
        n_test, n_voxels = test_data.shape
        
        # Create output file
        averaged_file = os.path.join(output_dir, f'fmri_{subject_id}_split-test_averaged.h5')
        
        with h5py.File(averaged_file, 'w') as f_out:
            averaged_dataset = f_out.create_dataset(
                'neural_data',
                shape=(n_unique, n_voxels),
                dtype='float32'
            )
            
            # Average across repetitions for each unique stimulus
            for i, stimulus in enumerate(tqdm(unique_stimuli, desc="Averaging test stimuli")):
                stimulus_mask = test_stimuli == stimulus
                stimulus_indices = np.where(stimulus_mask)[0]
                
                # Load data for this stimulus across all repetitions
                stimulus_data = test_data[stimulus_indices, :]
                
                # Average across repetitions
                averaged_data = np.mean(stimulus_data, axis=0)
                
                # Write to output
                averaged_dataset[i, :] = averaged_data
    
    print(f"Averaged test shape: ({n_unique}, {n_voxels})")



# =============================================================================
# Create dataset metadata
# =============================================================================

def extract_roi_indices(voxel_df):
    """Extract voxel indices for each functional ROI.
    
    Parameters
    ----------
    voxel_df : pd.DataFrame
        Voxel metadata dataframe.
        
    Returns
    -------
    dict
        Dictionary mapping ROI names to arrays of voxel indices.
    """
    # Define functional ROIs (exclude Glasser parcels)
    functional_rois = [
        'V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2',
        'LO1 (prf)', 'LO2 (prf)', 'TO1', 'TO2', 'V3b', 'V3a',
        'lFFA', 'rFFA', 'lOFA', 'rOFA',
        'lEBA', 'rEBA',
        'lPPA', 'rPPA', 'lRSC', 'rRSC', 'lTOS', 'rTOS',
        'lLOC', 'rLOC', 'IT',
        'lSTS', 'rSTS'
    ]
    
    roi_indices = {}
    
    for roi_name in functional_rois:
        if roi_name in voxel_df.columns:
            # Get voxel indices where ROI == 1
            roi_mask = voxel_df[roi_name] == 1
            indices = np.where(roi_mask)[0]
            
            # Create clean ROI name for metadata key
            # Replace spaces and parentheses: 'LO1 (prf)' -> 'LO1_prf'
            clean_name = roi_name.replace(' (', '_').replace(')', '').replace(' ', '_')
            roi_indices[f'{clean_name}'] = indices
            
            print(f"  {roi_name}: {len(indices)} voxels")
    
    return roi_indices


def create_fmri_metadata(stimulus_filepath, voxel_filepath, output_dir, subject_id,
                        session_means=None, session_stds=None, unique_sessions=None):
    """Create comprehensive metadata file for fMRI dataset.
    
    Generate metadata linking neural responses to stimulus images and voxel properties.
    Includes experimental conditions, voxel anatomical/functional information, and
    ROI indices for both training and test sets.
    
    Parameters
    ----------
    stimulus_filepath : str
        Path to the stimulus metadata CSV file.
    voxel_filepath : str
        Path to the voxel metadata CSV file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
        
    Output Files
    ------------
    fmri_{subject}_metadata.npz : Complete dataset metadata including stimulus
                                 mappings, experimental conditions, voxel properties,
                                 ROI indices, and normalization statistics (if provided)
    """
    print("Creating dataset metadata...")
    
    # Load metadata files
    print(f"Loading stimulus metadata from: {stimulus_filepath}")
    stim_metadata = pd.read_csv(stimulus_filepath)
    
    print(f"Loading voxel metadata from: {voxel_filepath}")
    voxel_metadata = pd.read_csv(voxel_filepath)
    
    # Split masks
    train_mask = stim_metadata['trial_type'] == 'train'
    test_mask = stim_metadata['trial_type'] == 'test'
    
    # Extract training metadata
    train_data = stim_metadata[train_mask]
    train_stimuli = train_data['stimulus'].values
    train_concepts = train_data['concept'].values
    
    # Extract test metadata (individual trials)
    test_data = stim_metadata[test_mask]
    test_stimuli = test_data['stimulus'].values
    test_concepts = test_data['concept'].values
    
    
    # Extract voxel information
    print("Extracting voxel information...")
    voxel_coords = voxel_metadata[['voxel_x', 'voxel_y', 'voxel_z']].values
    noise_ceiling_singletrial = voxel_metadata['nc_singletrial'].values
    noise_ceiling_testset = voxel_metadata['nc_testset'].values
    splithalf_corrected = voxel_metadata['splithalf_corrected'].values
    splithalf_uncorrected = voxel_metadata['splithalf_uncorrected'].values
    prf_eccentricity = voxel_metadata['prf-eccentricity'].values
    prf_polarangle = voxel_metadata['prf-polarangle'].values
    prf_rsquared = voxel_metadata['prf-rsquared'].values
    prf_size = voxel_metadata['prf-size'].values
    
    # Extract ROI indices
    print("Extracting ROI indices...")
    roi_indices = extract_roi_indices(voxel_metadata)
    
    # Compile metadata dictionary
    metadata_dict = {
        'fmri': {
            'voxel_coords': voxel_coords,
            'n_voxels': len(voxel_metadata),
            'subject_id': subject_id},
        'encoding_model':{
            'train_stimuli': train_stimuli,
            'train_concepts': train_concepts,
            'test_stimuli': test_stimuli,
            'test_concepts': test_concepts,
            'noise_ceiling_singletrial': noise_ceiling_singletrial,
            'noise_ceiling_testset': noise_ceiling_testset,
            'splithalf_corrected': splithalf_corrected,
            'splithalf_uncorrected': splithalf_uncorrected,
        },
        'prf' : {            
            'prf_eccentricity': prf_eccentricity,
            'prf_polarangle': prf_polarangle,
            'prf_rsquared': prf_rsquared,
            'prf_size': prf_size},
        'roi': {}
    }
    
    
    # Add ROI indices to metadata
    metadata_dict['roi'].update(roi_indices)
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'fmri_{subject_id}_metadata.npy')
    np.save(metadata_file, metadata_dict, allow_pickle=True)