import h5py
import numpy as np
import mne
import os
from tqdm import tqdm


# =============================================================================
# Split training and test data
# =============================================================================

def split_meg_data(meg_filepath, output_dir, subject_id, batch_size):
    """Split MEG neural data into training and test partitions.
    
    Load preprocessed MNE epochs and separate based on trial_type using
    chunked processing to minimize memory usage. Data is never fully loaded
    into memory - instead processed in batches and written directly to disk.
    
    Parameters
    ----------
    meg_filepath : str
        Path to the preprocessed MNE epochs .fif file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    batch_size : int
        Batch size for chunked processing.
        
    Output Files
    ------------
    meg_{subject}_split-train.h5 : (22248, 271, 281)
    meg_{subject}_split-test.h5  : (2400, 271, 281)
    """
    print(f"Loading MNE epochs metadata from: {meg_filepath}")
    epochs = mne.read_epochs(meg_filepath, preload=False, verbose=False)
    
    # Get metadata without loading data
    metadata = epochs.metadata
    
    # Split based on trial_type
    train_mask = metadata['trial_type'] == 'exp'
    test_mask = metadata['trial_type'] == 'test'
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    n_train = len(train_indices)
    n_test = len(test_indices)
    
    print(f"Training trials: {n_train}")
    print(f"Test trials: {n_test}")
    
    # Get data shape info
    n_channels = len(epochs.info['ch_names'])
    n_times = len(epochs.times)
    
    # Create output files with pre-allocated datasets
    train_file = os.path.join(output_dir, f'meg_{subject_id}_split-train.h5')
    test_file = os.path.join(output_dir, f'meg_{subject_id}_split-test.h5')
    
    with h5py.File(train_file, 'w') as f_train:
        train_dataset = f_train.create_dataset(
            'neural_data', 
            shape=(n_train, n_channels, n_times),
            dtype='float64'
        )
        
        with h5py.File(test_file, 'w') as f_test:
            test_dataset = f_test.create_dataset(
                'neural_data',
                shape=(n_test, n_channels, n_times), 
                dtype='float64'
            )
            
            print("Processing data in batches...")
            train_write_idx = 0
            test_write_idx = 0
            
            # Process in batches to control memory usage
            n_batches = int(np.ceil(len(epochs) / batch_size))
            
            for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(epochs))
                
                # Load only this batch
                batch_epochs = epochs[start_idx:end_idx]
                batch_data = batch_epochs.get_data()  # (batch_size, channels, times)
                
                # Separate train and test within this batch
                batch_train_mask = train_mask[start_idx:end_idx]
                batch_test_mask = test_mask[start_idx:end_idx]
                
                if np.any(batch_train_mask):
                    train_chunk = batch_data[batch_train_mask]
                    n_train_chunk = train_chunk.shape[0]
                    train_dataset[train_write_idx:train_write_idx + n_train_chunk] = train_chunk
                    train_write_idx += n_train_chunk
                
                if np.any(batch_test_mask):
                    test_chunk = batch_data[batch_test_mask]
                    n_test_chunk = test_chunk.shape[0]
                    test_dataset[test_write_idx:test_write_idx + n_test_chunk] = test_chunk
                    test_write_idx += n_test_chunk
    
    print(f"Training shape: ({n_train}, {n_channels}, {n_times})")
    print(f"Test shape: ({n_test}, {n_channels}, {n_times})")


# =============================================================================
# Normalize MEG responses
# =============================================================================

def load_metadata_and_baseline_info(meg_filepath):
    """Load MEG epochs and compute full time window information for normalization."""
    epochs = mne.read_epochs(meg_filepath, preload=False, verbose=False)
    
    metadata = epochs.metadata
    times = epochs.times  # (281,) in seconds
    
    # Identify training trials and their sessions
    train_mask = metadata['trial_type'] == 'exp'
    train_sessions = metadata.loc[train_mask, 'session_nr'].values
    
    # Use full time window for normalization (all 281 timepoints)
    # This accounts for actual response variance, not just baseline noise
    full_time_indices = np.arange(len(times))
    
    return metadata, times, train_mask, train_sessions, full_time_indices


def compute_session_specific_baseline_stats(output_dir, subject_id, train_sessions, baseline_indices):
    """Compute baseline statistics separately for each recording session."""
    unique_sessions = np.unique(train_sessions)
    baseline_stats = {}
    
    print("Computing session-specific baseline statistics...")
    
    # Load training data in read-only mode
    train_file = os.path.join(output_dir, f'meg_{subject_id}_split-train.h5')
    with h5py.File(train_file, 'r') as f:
        train_data = f['neural_data']  # (22248, 271, 281)
        
        for session in tqdm(unique_sessions, desc="Processing sessions"):
            session_mask = train_sessions == session
            session_indices = np.where(session_mask)[0]
            
            # Load only this session's data
            session_data = train_data[session_indices]  # (N_session, 271, 281)
            
            # Extract baseline period
            session_baseline = session_data[:, :, baseline_indices]  # (N_session, 271, 20)
            
            # Compute mean and std across trials and time
            session_mean = np.mean(session_baseline, axis=(0, 2))  # (271,)
            session_std = np.std(session_baseline, axis=(0, 2))    # (271,)
            session_std = np.maximum(session_std, 1e-8)  # Avoid division by zero
            
            baseline_stats[session] = {
                'mean': session_mean,
                'std': session_std
            }
    
    return baseline_stats


def apply_normalization_to_training_data(output_dir, subject_id, train_sessions, 
                                        baseline_stats, batch_size):
    """Apply session-specific z-score normalization to training data."""
    train_file = os.path.join(output_dir, f'meg_{subject_id}_split-train.h5')
    normalized_file = os.path.join(output_dir, f'meg_{subject_id}_split-train_normalized.h5')
    
    with h5py.File(train_file, 'r') as f_in:
        train_data = f_in['neural_data']
        n_trials, n_sensors, n_timepoints = train_data.shape
        
        with h5py.File(normalized_file, 'w') as f_out:
            normalized_dataset = f_out.create_dataset(
                'neural_data',
                shape=(n_trials, n_sensors, n_timepoints),
                dtype='float32'
            )
            
            print("Normalizing training data...")
            
            for start_idx in tqdm(range(0, n_trials, batch_size), desc="Normalizing batches"):
                end_idx = min(start_idx + batch_size, n_trials)
                
                chunk_data = train_data[start_idx:end_idx]  # (batch, 271, 281)
                chunk_sessions = train_sessions[start_idx:end_idx]
                
                normalized_chunk = np.zeros_like(chunk_data, dtype='float32')
                
                for i, session in enumerate(chunk_sessions):
                    trial_data = chunk_data[i]  # (271, 281)
                    session_mean = baseline_stats[session]['mean']  # (271,)
                    session_std = baseline_stats[session]['std']    # (271,)
                    
                    # Z-score: subtract mean and divide by std
                    normalized_chunk[i] = (trial_data - session_mean[:, None]) / session_std[:, None]
                
                normalized_dataset[start_idx:end_idx] = normalized_chunk


def create_averaged_test_data(output_dir, subject_id, test_image_nrs, test_data):
    """Create averaged test data across 12 repetitions.
    
    Parameters
    ----------
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    test_image_nrs : np.ndarray
        Test image numbers for each trial.
    test_data : np.ndarray
        Test data to average (2400, 271, 281).
        
    Returns
    -------
    np.ndarray
        Averaged test data (200, 271, 281).
    """
    unique_test_images = np.unique(test_image_nrs)
    n_unique = len(unique_test_images)
    test_averaged = np.zeros((n_unique, test_data.shape[1], test_data.shape[2]), dtype='float32')
    
    for i, img_nr in enumerate(unique_test_images):
        img_mask = test_image_nrs == img_nr
        test_averaged[i] = np.mean(test_data[img_mask], axis=0)
    
    return test_averaged


def apply_normalization_to_test_data(meg_filepath, output_dir, subject_id, baseline_stats):
    """Apply session-specific normalization to test data and create averaged versions."""
    # Load metadata to get test sessions
    epochs = mne.read_epochs(meg_filepath, preload=False, verbose=False)
    metadata = epochs.metadata
    
    test_mask = metadata['trial_type'] == 'test'
    test_sessions = metadata.loc[test_mask, 'session_nr'].values
    test_image_nrs = metadata.loc[test_mask, 'test_image_nr'].values
    
    # Load test data
    test_file = os.path.join(output_dir, f'meg_{subject_id}_split-test.h5')
    with h5py.File(test_file, 'r') as f:
        test_data = f['neural_data'][:]  # (2400, 271, 281)
    
    # Create non-normalized averaged test data
    print("Creating non-normalized averaged test data...")
    test_averaged = create_averaged_test_data(output_dir, subject_id, test_image_nrs, test_data)
    
    # Save non-normalized averaged test data
    averaged_test_file = os.path.join(output_dir, f'meg_{subject_id}_split-test_averaged.h5')
    with h5py.File(averaged_test_file, 'w') as f:
        f.create_dataset('neural_data', data=test_averaged)
    
    print(f"Non-normalized averaged test shape: {test_averaged.shape}")
    
    # Normalize test data
    normalized_test = np.zeros_like(test_data, dtype='float32')
    
    print("Normalizing test data...")
    for i, session in enumerate(tqdm(test_sessions, desc="Processing test trials")):
        if session in baseline_stats:
            session_mean = baseline_stats[session]['mean']
            session_std = baseline_stats[session]['std']
        else:
            # Fallback to average across all sessions
            all_means = np.array([baseline_stats[s]['mean'] for s in baseline_stats.keys()])
            all_stds = np.array([baseline_stats[s]['std'] for s in baseline_stats.keys()])
            session_mean = np.mean(all_means, axis=0)
            session_std = np.mean(all_stds, axis=0)
        
        normalized_test[i] = (test_data[i] - session_mean[:, None]) / session_std[:, None]
    
    # Save normalized test data
    normalized_test_file = os.path.join(output_dir, f'meg_{subject_id}_split-test_normalized.h5')
    with h5py.File(normalized_test_file, 'w') as f:
        f.create_dataset('neural_data', data=normalized_test)
    
    # Create normalized averaged test data
    print("Creating normalized averaged test data...")
    test_averaged_normalized = create_averaged_test_data(output_dir, subject_id, test_image_nrs, normalized_test)
    
    # Save normalized averaged test data
    averaged_test_normalized_file = os.path.join(output_dir, f'meg_{subject_id}_split-test_averaged_normalized.h5')
    with h5py.File(averaged_test_normalized_file, 'w') as f:
        f.create_dataset('neural_data', data=test_averaged_normalized)
    
    print(f"Normalized test shape: {normalized_test.shape}")
    print(f"Normalized averaged test shape: {test_averaged_normalized.shape}")


def normalize_meg_data(meg_filepath, output_dir, subject_id, batch_size):
    """Apply session-specific z-score normalization to MEG data.
    
    Performs normalization using baseline period (-100 to 0ms) statistics
    computed separately for each recording session to account for session 
    variability. Training data normalization uses own statistics to prevent
    data leakage, while test data uses training-derived parameters.
    
    Parameters
    ----------
    meg_filepath : str
        Path to the preprocessed MNE epochs .fif file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    batch_size : int
        Batch size for chunked processing.
        
    Returns
    -------
    dict
        Baseline normalization statistics for metadata inclusion.
        
    Output Files
    ------------
    meg_{subject}_split-train_normalized.h5 : (22248, 271, 281)
    meg_{subject}_split-test_normalized.h5  : (2400, 271, 281)
    meg_{subject}_split-test_averaged.h5    : (200, 271, 281) - Non-normalized
    meg_{subject}_split-test_averaged_normalized.h5 : (200, 271, 281) - Normalized
    """
    # Load metadata and baseline information
    metadata, times, train_mask, train_sessions, baseline_indices = load_metadata_and_baseline_info(meg_filepath)
    
    print(f"Baseline period: {times[baseline_indices[0]]:.3f} to {times[baseline_indices[-1]]:.3f} s")
    print(f"Baseline indices: {baseline_indices[0]} to {baseline_indices[-1]}")
    
    # Compute session-specific baseline statistics
    baseline_stats = compute_session_specific_baseline_stats(
        output_dir, subject_id, train_sessions, baseline_indices
    )
    
    # Normalize training data
    apply_normalization_to_training_data(
        output_dir, subject_id, train_sessions, baseline_stats, batch_size
    )
    
    # Normalize test data and create both averaged versions
    apply_normalization_to_test_data(
        meg_filepath, output_dir, subject_id, baseline_stats
    )
    
    # Prepare baseline statistics for metadata
    unique_sessions = sorted(baseline_stats.keys())
    baseline_means = np.array([baseline_stats[s]['mean'] for s in unique_sessions])
    baseline_stds = np.array([baseline_stats[s]['std'] for s in unique_sessions])
    
    baseline_metadata = {
        'baseline_means': baseline_means,
        'baseline_stds': baseline_stds,
        'baseline_sessions': np.array(unique_sessions),
        'baseline_time_range': np.array([times[baseline_indices[0]], times[baseline_indices[-1]]]),
        'baseline_indices': baseline_indices
    }
    
    return baseline_metadata

# =============================================================================
# Create dataset metadata
# =============================================================================

def create_meg_metadata(meg_filepath, output_dir, subject_id, baseline_stats):
    """Create comprehensive metadata file for MEG dataset.
    
    Generate metadata linking neural responses to THINGS database images through
    things_image_nr. Includes experimental conditions, baseline normalization
    parameters, and sensor information for both training and test sets.
    
    Parameters
    ----------
    meg_filepath : str
        Path to the preprocessed MNE epochs .fif file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    baseline_stats : dict
        Baseline normalization statistics from normalize_meg_data.
        
    Output Files
    ------------
    meg_{subject}_metadata.npz : Complete dataset metadata including stimulus
                                mappings, experimental conditions, baseline stats,
                                and sensor information
    """
    print("Creating dataset metadata...")
    
    # Load MNE epochs
    epochs = mne.read_epochs(meg_filepath, preload=False, verbose=False)
    metadata = epochs.metadata
    times = epochs.times
    
    # Get sensor information
    sensor_names = np.array(epochs.info['ch_names'])
    
    # Extract sensor region information from channel names
    sensor_prefixes = []
    sensor_hemispheres = []
    sensor_regions = []
    
    hemisphere_map = {'L': 'Left', 'R': 'Right', 'Z': 'Midline'}
    region_map = {'F': 'Frontal', 'C': 'Central', 'P': 'Parietal', 
                  'T': 'Temporal', 'O': 'Occipital'}
    
    for name in sensor_names:
        # Extract prefix (e.g., 'MLT23-1609' -> 'MLT')
        prefix = name.split('-')[0][:3]
        sensor_prefixes.append(prefix)
        
        # Parse hemisphere (second character: L/R/Z)
        hemisphere_code = prefix[1]
        if hemisphere_code not in hemisphere_map:
            raise ValueError(f"Unknown hemisphere code '{hemisphere_code}' in sensor '{name}'. "
                           f"Expected L, R, or Z.")
        sensor_hemispheres.append(hemisphere_map[hemisphere_code])
        
        # Parse region (third character: F/C/P/T/O)
        region_code = prefix[2]
        if region_code not in region_map:
            raise ValueError(f"Unknown region code '{region_code}' in sensor '{name}'. "
                           f"Expected F, C, P, T, or O.")
        sensor_regions.append(region_map[region_code])
    
    sensor_prefixes = np.array(sensor_prefixes)
    sensor_hemispheres = np.array(sensor_hemispheres)
    sensor_regions = np.array(sensor_regions)
    
    # Split masks
    train_mask = metadata['trial_type'] == 'exp'
    test_mask = metadata['trial_type'] == 'test'
    
    # Extract training metadata
    train_metadata = metadata[train_mask]
    train_things_img_ids = train_metadata['things_image_nr'].values
    train_categories = train_metadata['category_nr'].values
    train_exemplars = train_metadata['exemplar_nr'].values
    train_sessions = train_metadata['session_nr'].values
    train_runs = train_metadata['run_nr'].values
    train_image_paths = train_metadata['image_path'].values
    
    # Create full image paths for training (strip 'images_meg/' prefix)
    train_full_image_paths = []
    for path in train_image_paths:
        if path.startswith('images_meg/'):
            train_full_image_paths.append(path.replace('images_meg/', '', 1))
        else:
            train_full_image_paths.append(path)
    train_full_image_paths = np.array(train_full_image_paths)
    
    # Extract test metadata (individual trials)
    test_metadata = metadata[test_mask]
    test_things_img_ids = test_metadata['things_image_nr'].values
    test_image_nr = test_metadata['test_image_nr'].values
    test_categories = test_metadata['category_nr'].values
    test_exemplars = test_metadata['exemplar_nr'].values
    test_sessions = test_metadata['session_nr'].values
    test_runs = test_metadata['run_nr'].values
    test_image_paths = test_metadata['image_path'].values
    
    # Create full image paths for test (reconstruct with concept from filename)
    test_full_image_paths = []
    for path in test_image_paths:
        if path.startswith('images_test_meg/'):
            # Extract filename: images_test_meg/coat_rack_13s.jpg -> coat_rack_13s.jpg
            filename = path.replace('images_test_meg/', '', 1)
            
            # Extract concept from filename by removing numeric suffix
            # coat_rack_13s.jpg -> coat_rack
            # limousine_15s.jpg -> limousine
            name_without_ext = filename.replace('.jpg', '')
            parts = name_without_ext.split('_')
            
            # Find where the numeric suffix starts (iterate backwards)
            concept = name_without_ext  # Fallback
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] and parts[i][0].isdigit():
                    concept = '_'.join(parts[:i])
                    break
            
            # Reconstruct: coat_rack/coat_rack_13s.jpg
            test_full_image_paths.append(f"{concept}/{filename}")
        else:
            test_full_image_paths.append(path)
    test_full_image_paths = np.array(test_full_image_paths)
    
    # Create averaged test metadata (one entry per unique test image)
    unique_test_images = np.unique(test_image_nr)
    test_avg_things_img_ids = []
    test_avg_categories = []
    test_avg_image_paths = []
    test_avg_full_image_paths = []
    
    for img_nr in unique_test_images:
        img_mask = test_image_nr == img_nr
        # Take the first occurrence for each unique test image
        idx = np.where(img_mask)[0][0]
        test_avg_things_img_ids.append(test_metadata.iloc[idx]['things_image_nr'])
        test_avg_categories.append(test_metadata.iloc[idx]['category_nr'])
        test_avg_image_paths.append(test_metadata.iloc[idx]['image_path'])
        
        # Create full image path for this averaged test image
        path = test_metadata.iloc[idx]['image_path']
        if path.startswith('images_test_meg/'):
            filename = path.replace('images_test_meg/', '', 1)
            
            # Extract concept by removing numeric suffix
            name_without_ext = filename.replace('.jpg', '')
            parts = name_without_ext.split('_')
            
            concept = name_without_ext  # Fallback
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] and parts[i][0].isdigit():
                    concept = '_'.join(parts[:i])
                    break
            
            test_avg_full_image_paths.append(f"{concept}/{filename}")
        else:
            test_avg_full_image_paths.append(path)
    
    test_avg_full_image_paths = np.array(test_avg_full_image_paths)
    
    # Compile metadata dictionary
    metadata_dict = {
        # Training data
        'train_things_img_ids': train_things_img_ids,
        'train_categories': train_categories,
        'train_exemplars': train_exemplars,
        'train_sessions': train_sessions,
        'train_runs': train_runs,
        'train_image_paths': train_image_paths,
        'train_full_image_path': train_full_image_paths,
        
        # Test data (individual trials)
        'test_things_img_ids': test_things_img_ids,
        'test_image_nr': test_image_nr,
        'test_categories': test_categories,
        'test_exemplars': test_exemplars,
        'test_sessions': test_sessions,
        'test_runs': test_runs,
        'test_image_paths': test_image_paths,
        'test_full_image_path': test_full_image_paths,
        
        # Test averaged data
        'test_avg_things_img_ids': np.array(test_avg_things_img_ids),
        'test_avg_image_nr': unique_test_images,
        'test_avg_categories': np.array(test_avg_categories),
        'test_avg_image_paths': np.array(test_avg_image_paths),
        'test_avg_full_image_path': test_avg_full_image_paths,
        
        # Temporal information
        'times': times,
        
        # Sensor information
        'sensor_names': sensor_names,
        'sensor_prefixes': sensor_prefixes,
        'sensor_hemispheres': sensor_hemispheres,
        'sensor_regions': sensor_regions,
        'n_sensors': len(sensor_names),
        
        # Normalization parameters
        'baseline_means': baseline_stats['baseline_means'],
        'baseline_stds': baseline_stats['baseline_stds'],
        'baseline_sessions': baseline_stats['baseline_sessions'],
        'baseline_time_range': baseline_stats['baseline_time_range'],
        'baseline_indices': baseline_stats['baseline_indices'],
        
        # Subject metadata
        'subject_id': subject_id
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'meg_{subject_id}_metadata.npz')
    np.savez(metadata_file, **metadata_dict)
    
    print(f"Training trials: {len(train_things_img_ids)}")
    print(f"Test trials: {len(test_things_img_ids)}")
    print(f"Unique test images: {len(unique_test_images)}")
    print(f"Time points: {len(times)} ({times[0]:.3f} to {times[-1]:.3f} s)")
    print(f"Sensors: {len(sensor_names)}")
    print(f"Metadata saved to: {metadata_file}")