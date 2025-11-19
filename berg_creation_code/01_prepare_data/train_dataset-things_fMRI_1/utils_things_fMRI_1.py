import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# =============================================================================
# Split training and test data
# =============================================================================

def split_fmri_data(response_filepath, stimulus_filepath, output_dir, subject_id, batch_size):
    """Split fMRI neural data into training and test partitions.
    
    Load preprocessed fMRI response data and separate based on trial_type using
    chunked processing to minimize memory usage. Data is transposed from original
    (Voxels x Trials) to (Trials x Voxels) format and processed in batches.
    
    Parameters
    ----------
    response_filepath : str
        Path to the fMRI response data HDF5 file.
    stimulus_filepath : str
        Path to the stimulus metadata CSV file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    batch_size : int
        Batch size for chunked processing (number of trials per batch).
        
    Output Files
    ------------
    fmri_{subject}_split-train.h5 : (n_train, 211339)
    fmri_{subject}_split-test.h5  : (n_test, 211339)
    """
    print(f"Loading stimulus metadata from: {stimulus_filepath}")
    stim_metadata = pd.read_csv(stimulus_filepath)
    
    print(f"Loading fMRI response data from: {response_filepath}")
    
    # Split based on trial_type
    train_mask = stim_metadata['trial_type'] == 'train'
    test_mask = stim_metadata['trial_type'] == 'test'
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    n_train = len(train_indices)
    n_test = len(test_indices)
    
    print(f"Training trials: {n_train}")
    print(f"Test trials: {n_test}")
    
    # Open response data to get shape information
    with h5py.File(response_filepath, 'r') as f:
        # Original shape: (211339 voxels, 9840 trials)
        response_data = f['ResponseData/block0_values']
        n_voxels, n_trials = response_data.shape
        
        print(f"Original data shape: ({n_voxels} voxels, {n_trials} trials)")
        print(f"Transposing to: ({n_trials} trials, {n_voxels} voxels)")
        
        # Create output files with pre-allocated datasets
        train_file = os.path.join(output_dir, f'fmri_{subject_id}_split-train.h5')
        test_file = os.path.join(output_dir, f'fmri_{subject_id}_split-test.h5')
        
        with h5py.File(train_file, 'w') as f_train:
            train_dataset = f_train.create_dataset(
                'neural_data', 
                shape=(n_train, n_voxels),
                dtype='float32'
            )
            
            with h5py.File(test_file, 'w') as f_test:
                test_dataset = f_test.create_dataset(
                    'neural_data',
                    shape=(n_test, n_voxels), 
                    dtype='float32'
                )
                
                print("Processing data in batches...")
                
                # Process voxels in batches to manage memory
                n_batches = int(np.ceil(n_voxels / batch_size))
                
                for batch_idx in tqdm(range(n_batches), desc="Processing voxel batches"):
                    voxel_start = batch_idx * batch_size
                    voxel_end = min(voxel_start + batch_size, n_voxels)
                    
                    # Load batch of voxels across all trials
                    # Shape: (n_voxels_in_batch, n_trials)
                    batch_data = response_data[voxel_start:voxel_end, :]
                    
                    # Transpose to (n_trials, n_voxels_in_batch)
                    batch_data_transposed = batch_data.T
                    
                    # Split into train and test
                    train_batch = batch_data_transposed[train_indices, :]
                    test_batch = batch_data_transposed[test_indices, :]
                    
                    # Write to output files
                    train_dataset[:, voxel_start:voxel_end] = train_batch
                    test_dataset[:, voxel_start:voxel_end] = test_batch
    
    print(f"Training shape: ({n_train}, {n_voxels})")
    print(f"Test shape: ({n_test}, {n_voxels})")


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
# Normalize data
# =============================================================================



def normalize_fmri_data(train_filepath, test_filepath, stimulus_filepath, subject_id):
    """Normalize fMRI data using session-wise z-score normalization.
    
    Computes mean and std per voxel per session from training data, then applies
    normalization to train and test data in-place. Test data uses training statistics.
    
    Parameters
    ----------
    train_filepath : str
        Path to training data HDF5 file
    test_filepath : str
        Path to test data HDF5 file
    stimulus_filepath : str
        Path to stimulus metadata CSV
    subject_id : str
        Subject identifier
        
    Returns
    -------
    tuple
        (train_session_means, train_session_stds, unique_sessions)
        Arrays of shape (n_sessions, n_voxels) and (n_sessions,)
    """
    print("Loading stimulus metadata for session info...")
    stim_metadata = pd.read_csv(stimulus_filepath)
    
    train_mask = stim_metadata['trial_type'] == 'train'
    test_mask = stim_metadata['trial_type'] == 'test'
    
    train_sessions = stim_metadata[train_mask]['session'].values
    test_sessions = stim_metadata[test_mask]['session'].values
    
    unique_sessions = np.unique(train_sessions)
    n_sessions = len(unique_sessions)
    
    print(f"Found {n_sessions} unique sessions")
    
    # Load training data and compute statistics on them
    print("Computing session-wise statistics on training data...")
    with h5py.File(train_filepath, 'r+') as f_train:
        train_data = f_train['neural_data']
        n_train, n_voxels = train_data.shape
        
        # Initialize statistics arrays
        train_session_means = np.zeros((n_sessions, n_voxels), dtype=np.float32)
        train_session_stds = np.zeros((n_sessions, n_voxels), dtype=np.float32)
        
        # Iterate through sessions and compute mean and std
        for i, session in enumerate(unique_sessions):
            session_mask = train_sessions == session
            session_indices = np.where(session_mask)[0]
            
            print(f"  Session {session}: {len(session_indices)} trials")
            
            # Load session data
            session_data = train_data[session_indices, :]
            
            # Compute mean and std per voxel
            session_mean = np.mean(session_data, axis=0)
            session_std = np.std(session_data, axis=0)
            
            # Handle zero std
            session_std[session_std == 0] = 1
            
            train_session_means[i, :] = session_mean
            train_session_stds[i, :] = session_std
        
        # Normalize training data in-place
        print("Normalizing training data...")
        for i, session in enumerate(tqdm(unique_sessions, desc="Normalizing train sessions")):
            session_mask = train_sessions == session
            session_indices = np.where(session_mask)[0]
            
            session_data = train_data[session_indices, :]
            session_mean = train_session_means[i, :]
            session_std = train_session_stds[i, :]
            normalized_data = (session_data - session_mean) / session_std
            train_data[session_indices, :] = normalized_data
            
    
    # Normalize test data using training statistics
    print("Normalizing test data...")
    with h5py.File(test_filepath, 'r+') as f_test:
        test_data = f_test['neural_data']
        
        for i, session in enumerate(tqdm(unique_sessions, desc="Normalizing test sessions")):
            session_mask = test_sessions == session
            session_indices = np.where(session_mask)[0]
            
            if len(session_indices) > 0:
                session_data = test_data[session_indices, :]
                session_mean = train_session_means[i, :]
                session_std = train_session_stds[i, :]
                normalized_data = (session_data - session_mean) / session_std
                test_data[session_indices, :] = normalized_data
    
    
    return train_session_means, train_session_stds, unique_sessions




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
            roi_indices[f'roi_{clean_name}'] = indices
            
            print(f"  {roi_name}: {len(indices)} voxels")
    
    return roi_indices


def create_fmri_metadata(stimulus_filepath, voxel_filepath, output_dir, subject_id,
                        train_means=None, train_stds=None, unique_sessions=None):
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
    train_sessions = train_data['session'].values
    train_runs = train_data['run'].values
    train_stimuli = train_data['stimulus'].values
    train_concepts = train_data['concept'].values
    train_trial_ids = train_data['trial_id'].values
    
    # Extract test metadata (individual trials)
    test_data = stim_metadata[test_mask]
    test_sessions = test_data['session'].values
    test_runs = test_data['run'].values
    test_stimuli = test_data['stimulus'].values
    test_concepts = test_data['concept'].values
    test_trial_ids = test_data['trial_id'].values
    
    

    # Create averaged test metadata (one entry per unique stimulus)
    unique_test_stimuli = np.unique(test_stimuli)
    test_avg_stimuli = []
    test_avg_concepts = []
    
    for stimulus in unique_test_stimuli:
        stimulus_mask = test_stimuli == stimulus
        # Take the first occurrence for each unique stimulus
        idx = np.where(stimulus_mask)[0][0]
        test_avg_stimuli.append(test_data.iloc[idx]['stimulus'])
        test_avg_concepts.append(test_data.iloc[idx]['concept'])
    
    test_avg_stimuli = np.array(test_avg_stimuli)
    test_avg_concepts = np.array(test_avg_concepts)
    
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
    
    # In the 'fmri' section:

    
    # Compile metadata dictionary
    metadata_dict = {
        'fmri': {
            # Training data
            'train_sessions': train_sessions,
            'train_runs': train_runs,
            'train_stimuli': train_stimuli,
            'train_concepts': train_concepts,
            'train_trial_ids': train_trial_ids,
            
            # Test data (individual trials)
            'test_sessions': test_sessions,
            'test_runs': test_runs,
            'test_stimuli': test_stimuli,
            'test_concepts': test_concepts,
            'test_trial_ids': test_trial_ids,
            
            # Test averaged data
            'test_avg_stimuli': test_avg_stimuli,
            'test_avg_concepts': test_avg_concepts,
            
            # Voxel information
            'voxel_coords': voxel_coords,
            'prf_eccentricity': prf_eccentricity,
            'prf_polarangle': prf_polarangle,
            'prf_rsquared': prf_rsquared,
            'prf_size': prf_size,
            'n_voxels': len(voxel_metadata),
            
            # Subject metadata
            'subject_id': subject_id},
        'encoding_model':{
            'noise_ceiling_singletrial': noise_ceiling_singletrial,
            'noise_ceiling_testset': noise_ceiling_testset,
            'splithalf_corrected': splithalf_corrected,
            'splithalf_uncorrected': splithalf_uncorrected,
        }
    }
    
    if train_means is not None:
        metadata_dict['fmri']['train_session_means'] = train_means
        metadata_dict['fmri']['train_session_stds'] = train_stds
        metadata_dict['fmri']['unique_sessions'] = unique_sessions
    
    # Add ROI indices to metadata
    metadata_dict['fmri'].update(roi_indices)
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'fmri_{subject_id}_metadata.npy')
    np.save(metadata_file, metadata_dict, allow_pickle=True)
    
    print(f"Training trials: {len(train_sessions)}")
    print(f"Test trials: {len(test_sessions)}")
    print(f"Unique test stimuli: {len(test_avg_stimuli)}")
    print(f"Voxels: {len(voxel_metadata)}")
    print(f"Functional ROIs: {len(roi_indices)}")
    print(f"Metadata saved to: {metadata_file}")