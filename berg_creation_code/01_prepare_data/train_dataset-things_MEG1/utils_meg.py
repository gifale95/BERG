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
# Create dataset metadata
# =============================================================================

def create_meg_metadata(meg_filepath, output_dir, subject_id):
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
        'meg': {
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
        
        # Subject metadata
        'subject_id': subject_id}
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'meg_{subject_id}_metadata.npy')
    np.save(metadata_file, metadata, allow_pickle=True)
    
    print(f"Training trials: {len(train_things_img_ids)}")
    print(f"Test trials: {len(test_things_img_ids)}")
    print(f"Unique test images: {len(unique_test_images)}")
    print(f"Time points: {len(times)} ({times[0]:.3f} to {times[-1]:.3f} s)")
    print(f"Sensors: {len(sensor_names)}")
    print(f"Metadata saved to: {metadata_file}")