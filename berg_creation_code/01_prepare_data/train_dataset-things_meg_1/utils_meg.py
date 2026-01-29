import h5py
import numpy as np
import mne
import os
from tqdm import tqdm


# =============================================================================
# Split training and test data
# =============================================================================

def split_meg_data(meg_filepath, output_dir, subject_id, batch_size, create_splits=True):
    """Split MEG neural data into training and test partitions.
    
    Load preprocessed MNE epochs and separate based on trial_type using
    chunked processing to minimize memory usage. Data is never fully loaded
    into memory - instead processed in batches and written directly to disk.
    
    Optionally creates 4 random training splits by shuffling training indices.
    
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
    create_splits : bool, optional
        Whether to create 4 random training splits (default: True).
        
    Output Files
    ------------
    meg_{subject}_all_training_splits.h5 : (22248, 271, 281)
    meg_{subject}_split-test.h5  : (2400, 271, 281)
    meg_{subject}_split-test_averaged.h5 : (200, 271, 281)
    
    If create_splits=True, additionally:
    meg_{subject}_single_training_split_1.h5 : (5562, 271, 281)
    meg_{subject}_single_training_split_2.h5 : (5562, 271, 281)
    meg_{subject}_single_training_split_3.h5 : (5562, 271, 281)
    meg_{subject}_single_training_split_4.h5 : (5562, 271, 281)
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
    
    # Get test image numbers for averaging
    test_metadata = metadata[test_mask]
    test_image_nrs = test_metadata['test_image_nr'].values
    
    # Create output files with pre-allocated datasets
    train_file = os.path.join(output_dir, f'meg_{subject_id}_all_training_splits.h5')
    test_file = os.path.join(output_dir, f'meg_{subject_id}_split-test.h5')
    
    with h5py.File(train_file, 'w') as f_train:
        train_dataset = f_train.create_dataset(
            'neural_data', 
            shape=(n_train, n_channels, n_times),
            dtype='float32'
        )
        
        with h5py.File(test_file, 'w') as f_test:
            test_dataset = f_test.create_dataset(
                'neural_data',
                shape=(n_test, n_channels, n_times), 
                dtype='float32'
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
    
    # Process test data averaged
    print("Processing test data averaged...")
    
    # Load the test data we just saved
    with h5py.File(test_file, 'r') as f:
        test_data = f['neural_data'][:]  # (2400, 271, 281)
    
    unique_test_images = np.unique(test_image_nrs)
    test_averaged = np.zeros((len(unique_test_images), n_channels, n_times), dtype='float32')
    
    for i, img_nr in enumerate(tqdm(unique_test_images, desc="Averaging test data")):
        mask = test_image_nrs == img_nr
        test_averaged[i] = np.mean(test_data[mask], axis=0)
    
    averaged_test_file = os.path.join(output_dir, f'meg_{subject_id}_split-test_averaged.h5')
    
    with h5py.File(averaged_test_file, 'w') as f_out:
        f_out.create_dataset('neural_data', data=test_averaged)
    
    print(f"Averaged test shape: {test_averaged.shape}")
    
    if create_splits:
        print("")
        print("Creating 4 random training splits...")
        
        seed = 20200220
        np.random.seed(seed)
        
        shuffled_indices = np.random.permutation(n_train)
        
        repeat_size = n_train // 4
        
        with h5py.File(train_file, 'r') as f_train:
            train_data = f_train['neural_data'][:]
        
        for split_idx in range(1, 5):
            start_idx = (split_idx - 1) * repeat_size
            end_idx = split_idx * repeat_size
            
            split_indices = shuffled_indices[start_idx:end_idx]
            split_data = train_data[split_indices]
            
            split_file = os.path.join(output_dir, f'meg_{subject_id}_single_training_split_{split_idx}.h5')
            
            with h5py.File(split_file, 'w') as f_split:
                f_split.create_dataset('neural_data', data=split_data)
            
            print(f"Split {split_idx} shape: {split_data.shape}")
        
        return shuffled_indices
    else:
        return None


# =============================================================================
# Compute Noise Ceiling
# =============================================================================


def compute_noise_ceiling(meg_filepath, test_filepath, subject_id):
    """Compute ncsnr and noise ceiling from test data with repeated presentations.
    
    Estimates noise ceiling using the variance across 12 repeated presentations
    of 200 test images. The noise ceiling represents the maximum achievable
    prediction accuracy given measurement noise.
    
    Parameters
    ----------
    meg_filepath : str
        Path to the preprocessed MNE epochs .fif file to extract metadata.
    test_filepath : str
        Path to the processed test HDF5 file (2400, 271, 281).
    subject_id : str
        Subject identifier for saving results.
        
    Returns
    -------
    dict
        'ncsnr': (271, 281) - Neural signal-to-noise ratio per sensor/timepoint
        'noise_ceiling': (271, 281) - Noise ceiling in r² percentage units (0-100)
    """
    # =============================================================================
    # Load the THINGS MEG responses for the test images
    # =============================================================================
    # Load test stimulus IDs from metadata
    data = mne.read_epochs(meg_filepath, preload=False, verbose=False)
    metadata = data.metadata
    test_mask = metadata['trial_type'] == 'test'
    test_metadata = metadata[test_mask]
    stimulus_ids = test_metadata['test_image_nr'].values
    
    # Load test neural data
    with h5py.File(test_filepath, 'r') as f:
        meg_data = f['neural_data'][:].astype(np.float32)
    
    # Get the unique image number
    unique_test_images = np.unique(stimulus_ids)
    
    # Reshape the MEG data to (samples, features)
    n_sensors = meg_data.shape[1]
    n_timepoints = meg_data.shape[2]
    n_features = n_sensors * n_timepoints
    neural_data = meg_data.reshape(meg_data.shape[0], n_features)
    
    # =============================================================================
    # Compute the ncsnr and noise ceiling
    # =============================================================================
    # Estimate the noise standard deviation (calculate the variance of the
    # responses across the 30 presentations of each test image).
    var = []
    for img in unique_test_images:
        idx = np.where(stimulus_ids == img)[0]
        var.append(np.nanvar(neural_data[idx], axis=0, ddof=1))
    # Average the variance across images and compute the square root of the
    # result
    sigma_noise = np.sqrt(np.nanmean(var, 0))

    # Estimate the signal standard deviation (total variance - noise variance)
    tot_var_data = np.nanvar(neural_data, axis=0, ddof=1)
    sigma_signal = tot_var_data - (sigma_noise ** 2)
    sigma_signal[sigma_signal<0] = 0
    sigma_signal = np.sqrt(sigma_signal)

    # Compute the ncsnr
    ncsnr = sigma_signal / sigma_noise

    # Convert the ncsnr to noise ceiling (the noise ceiling is in r² explained
    # variance units)
    img_reps = 12
    noise_ceiling = 100 * (ncsnr ** 2) / ((ncsnr ** 2) + (1 / img_reps))

    # Reshape the scores to (n_sensors, n_timepoints)
    ncsnr = ncsnr.reshape(n_sensors, n_timepoints)
    noise_ceiling = noise_ceiling.reshape(n_sensors, n_timepoints)

    # =============================================================================
    # Return the ncsnr and noise ceiling
    # =============================================================================
    results = {
        'ncsnr': ncsnr,
        'noise_ceiling': noise_ceiling
    }
    
    return results


# =============================================================================
# Create dataset metadata
# =============================================================================

def create_meg_metadata(meg_filepath, output_dir, subject_id, create_splits=True, shuffled_indices=None):
    """Create comprehensive metadata file for MEG dataset.
    
    Generate metadata linking neural responses to THINGS database images through
    things_image_nr. Includes experimental conditions and sensor information for 
    both training and test sets.
    
    Parameters
    ----------
    meg_filepath : str
        Path to the preprocessed MNE epochs .fif file.
    output_dir : str
        Output directory for processed data files.
    subject_id : str
        Subject identifier for file naming.
    create_splits : bool, optional
        Whether to include split metadata (default: True).
    shuffled_indices : ndarray, optional
        Shuffled training indices for creating splits.
        
    Output Files
    ------------
    meg_{subject}_metadata.npy : Complete dataset metadata
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
    
    # Turn into category and image
    train_stimuli = []
    train_concepts = []
    for path in train_full_image_paths:
        conc, stim = path.split("/")
        train_concepts.append(conc)
        train_stimuli.append(stim)
        
    
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
    
    # Turn into category and image
    test_stimuli = []
    test_concepts = []
    for path in test_full_image_paths:
        conc, stim = path.split("/")
        test_concepts.append(conc)
        test_stimuli.append(stim)
    
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
    
    
    # Turn into category and image
    test_avg_stimuli = []
    test_avg_concepts = []
    for path in test_avg_full_image_paths:
        conc, stim = path.split("/")
        test_avg_concepts.append(conc)
        test_avg_stimuli.append(stim)
    
    # Compute noise ceilings
    test_filepath = os.path.join(output_dir, f'meg_{subject_id}_split-test.h5')
    nc_data = compute_noise_ceiling(meg_filepath, test_filepath, subject_id)
    ncsnr = nc_data["ncsnr"]
    noise_ceiling = nc_data["noise_ceiling"]
    
    # Compile metadata dictionary
    metadata_dict = {
        'meg': {
        'times': times,
        'subject_id': subject_id},
        
        'sensors': {
        'sensor_names': sensor_names,
        'sensor_prefixes': sensor_prefixes,
        'sensor_hemispheres': sensor_hemispheres,
        'sensor_regions': sensor_regions,
        'n_sensors': len(sensor_names),
        },
        
        'encoding_model': {
            'all_training_splits': {
                'train_img_ids': train_things_img_ids,
                'train_stimuli': train_stimuli,
                'train_concepts': train_concepts,
                'train_sessions': train_sessions,
                'train_runs': train_runs,
                'train_img_files': train_full_image_paths,
            },
            
            'test_img_ids': test_things_img_ids,
            'test_stimuli': test_stimuli,
            'test_concepts': test_concepts,
            'test_image_nr': test_image_nr,
            'test_sessions': test_sessions,
            'test_runs': test_runs,
            'test_img_files': test_full_image_paths,
            
            'ncsnr': ncsnr,
            'noise_ceiling': noise_ceiling}
    }
    
    if create_splits:
        if shuffled_indices is None:
            raise ValueError("shuffled_indices must be provided when create_splits=True")
        
        n_train = len(train_things_img_ids)
        split_size = n_train // 4
        
        for split_idx in range(1, 5):
            start_idx = (split_idx - 1) * split_size
            end_idx = split_idx * split_size
            
            split_indices = shuffled_indices[start_idx:end_idx]
            
            metadata_dict['encoding_model'][f'single_training_split_{split_idx}'] = {
                'train_img_ids': train_things_img_ids[split_indices],
                'train_stimuli': [train_stimuli[i] for i in split_indices],
                'train_concepts': [train_concepts[i] for i in split_indices],
                'train_sessions': train_sessions[split_indices],
                'train_runs': train_runs[split_indices],
                'train_img_files': train_full_image_paths[split_indices]
            }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'meg_{subject_id}_metadata.npy')
    np.save(metadata_file, metadata_dict, allow_pickle=True)
    
    print(f"Training trials: {len(train_things_img_ids)}")
    print(f"Test trials: {len(test_things_img_ids)}")
    print(f"Unique test images: {len(unique_test_images)}")
    print(f"Time points: {len(times)} ({times[0]:.3f} to {times[-1]:.3f} s)")
    print(f"Sensors: {len(sensor_names)}")
    print(f"Metadata saved to: {metadata_file}")