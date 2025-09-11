import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# =============================================================================
# Split training and test data
# =============================================================================
# Load raw neural data and split into training and test partitions based on
# stimulus type. Training data contains single presentations of 22,248 images,
# while test data contains 30 repetitions of 100 images for noise ceiling estimation.


def split_tvsd_data(filepath, output_dir, monkey_id, batch_size):
    """Split TVSD neural data into training and test partitions.
    
    Load raw neural recordings (25,248 trials) and separate based on stimulus type.
    Training trials (test_idx=0) contain single presentations of THINGS images,
    while test trials (test_idx≠0) contain 30 repetitions of 100 images for 
    noise ceiling estimation. Uses chunked processing for memory efficiency.
    
    Parameters
    ----------
    filepath : str
        Path to the raw THINGS_MUA_trials.mat file.
    output_dir : str
        Output directory for processed data files.
    monkey_id : str
        Monkey identifier for file naming.
    batch_size : int
        Batch size for chunked processing.
        
    Output Files
    ------------
    tvsd_{monkey}_split-train.h5 : (22,248, 300, 1024)
    tvsd_{monkey}_split-test.h5  : (3,000, 300, 1024)
    """
    with h5py.File(filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
        test_idx = ALLMAT[2, :]
        train_mask = test_idx == 0
        test_mask = test_idx != 0
        
        n_train = np.sum(train_mask)
        n_test = np.sum(test_mask)
        
        print(f"Training trials: {n_train}")
        print(f"Test trials: {n_test}")
        
        allmua_dataset = f['ALLMUA']
        total_trials = allmua_dataset.shape[1]
        
        # Process training data
        print("Processing training data...")
        train_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-train.h5')
        
        with h5py.File(train_file, 'w') as train_h5:
            train_dataset = train_h5.create_dataset('neural_data', 
                                                   shape=(n_train, 300, 1024), 
                                                   dtype='float64')
            
            train_idx = 0
            for start_idx in tqdm(range(0, total_trials, batch_size), desc="Training batches"):
                end_idx = min(start_idx + batch_size, total_trials)
                
                chunk_data = allmua_dataset[:, start_idx:end_idx, :]
                chunk_train_mask = train_mask[start_idx:end_idx]
                
                if np.any(chunk_train_mask):
                    train_chunk = chunk_data[:, chunk_train_mask, :].transpose(1, 0, 2)
                    n_train_chunk = train_chunk.shape[0]
                    train_dataset[train_idx:train_idx + n_train_chunk] = train_chunk
                    train_idx += n_train_chunk
        
        # Process test data
        print("Processing test data...")
        test_indices = np.where(test_mask)[0]
        test_data = allmua_dataset[:, test_indices, :].transpose(1, 0, 2)
        
        test_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test.h5')
        with h5py.File(test_file, 'w') as test_h5:
            test_h5.create_dataset('neural_data', data=test_data)
        
        print(f"Training shape: ({n_train}, 300, 1024)")
        print(f"Test shape: {test_data.shape}")
        
        
        
# =============================================================================
# Normalize neural responses
# =============================================================================
# Apply day-specific z-score normalization using pre-stimulus baseline period
# (-100 to 0ms) to account for daily recording variations and electrode drift.


def load_metadata_and_baseline_info(original_filepath, output_dir, monkey_id):
    """Load metadata and compute baseline period information."""
    with h5py.File(original_filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
        tb = f['tb'][:].flatten()
    
    test_idx = ALLMAT[2, :]
    train_mask = test_idx == 0
    train_days = ALLMAT[5, train_mask]
    
    train_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-train.h5')
    with h5py.File(train_file, 'r') as f:
        train_shape = f['neural_data'].shape
    
    baseline_mask = (tb >= -100) & (tb <= 0)
    baseline_indices = np.where(baseline_mask)[0]
    
    return train_days, tb, train_shape, train_file, baseline_indices


def compute_day_specific_baseline_stats(train_days, train_file, baseline_indices, batch_size):
    """Compute baseline statistics separately for each recording day."""
    unique_days = np.unique(train_days)
    baseline_stats = {}
    
    print("Computing day-specific baseline statistics...")
    
    with h5py.File(train_file, 'r') as f:
        train_data = f['neural_data']
        
        for day in tqdm(unique_days, desc="Processing days"):
            day_mask = train_days == day
            day_indices = np.where(day_mask)[0]
            
            day_data = train_data[day_indices, :, :]
            day_baseline = day_data[:, baseline_indices, :]
            
            day_mean = np.mean(day_baseline, axis=(0, 1))
            day_std = np.std(day_baseline, axis=(0, 1))
            day_std = np.maximum(day_std, 1e-8)
            
            baseline_stats[day] = {'mean': day_mean, 'std': day_std}
    
    return baseline_stats


def apply_normalization_to_training_data(train_days, train_file, baseline_stats, 
                                        output_dir, monkey_id, batch_size):
    """Apply day-specific z-score normalization to training data."""
    normalized_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-train_normalized.h5')
    
    with h5py.File(train_file, 'r') as f_in:
        train_data = f_in['neural_data']
        n_trials, n_timepoints, n_electrodes = train_data.shape
        
        with h5py.File(normalized_file, 'w') as f_out:
            normalized_dataset = f_out.create_dataset(
                'neural_data_normalized', 
                shape=(0, n_timepoints, n_electrodes), 
                maxshape=(None, n_timepoints, n_electrodes),
                dtype='float32',
                chunks=True
            )
            
            print("Normalizing training data...")
            
            for start_idx in tqdm(range(0, n_trials, batch_size), desc="Normalizing batches"):
                end_idx = min(start_idx + batch_size, n_trials)
                
                chunk_data = train_data[start_idx:end_idx, :, :]
                chunk_days = train_days[start_idx:end_idx]
                chunk_size_actual = chunk_data.shape[0]
                
                normalized_chunk = np.zeros_like(chunk_data, dtype='float32')
                
                for i, day in enumerate(chunk_days):
                    trial_data = chunk_data[i, :, :]
                    day_mean = baseline_stats[day]['mean']
                    day_std = baseline_stats[day]['std']
                    
                    normalized_chunk[i, :, :] = (trial_data - day_mean[None, :]) / day_std[None, :]
                
                current_size = normalized_dataset.shape[0]
                normalized_dataset.resize((current_size + chunk_size_actual, n_timepoints, n_electrodes))
                normalized_dataset[current_size:current_size + chunk_size_actual, :, :] = normalized_chunk


def apply_normalization_to_test_data(baseline_stats, original_filepath, output_dir, monkey_id):
    """Apply day-specific normalization to test data and create averaged version."""
    test_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test.h5')
    normalized_test_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test_normalized.h5')
    averaged_test_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test_averaged.h5')
    
    with h5py.File(original_filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
    
    test_idx = ALLMAT[2, :]
    test_mask = test_idx != 0
    test_days = ALLMAT[5, test_mask]
    test_stimuli = test_idx[test_mask]
    
    with h5py.File(test_file, 'r') as f_in:
        test_data = f_in['neural_data'][:]
        
        normalized_test = np.zeros_like(test_data, dtype='float32')
        
        print("Normalizing test data...")
        for i, day in enumerate(tqdm(test_days, desc="Processing test trials")):
            if day in baseline_stats:
                day_mean = baseline_stats[day]['mean']
                day_std = baseline_stats[day]['std']
            else:
                all_means = np.array([baseline_stats[d]['mean'] for d in baseline_stats.keys()])
                all_stds = np.array([baseline_stats[d]['std'] for d in baseline_stats.keys()])
                day_mean = np.mean(all_means, axis=0)
                day_std = np.mean(all_stds, axis=0)
            
            normalized_test[i, :, :] = (test_data[i, :, :] - day_mean[None, :]) / day_std[None, :]
        
        with h5py.File(normalized_test_file, 'w') as f_out:
            f_out.create_dataset('neural_data_normalized', data=normalized_test)
        
        # Create averaged test data
        unique_test_ids = np.unique(test_stimuli)
        test_averaged = np.zeros((len(unique_test_ids), test_data.shape[1], test_data.shape[2]), dtype='float32')
        
        for i, stimulus_id in enumerate(tqdm(unique_test_ids, desc="Averaging test data")):
            mask = test_stimuli == stimulus_id
            test_averaged[i] = np.mean(normalized_test[mask], axis=0)
        
        with h5py.File(averaged_test_file, 'w') as f_out:
            f_out.create_dataset('neural_data_averaged', data=test_averaged)
    
    print(f"Normalized test shape: {normalized_test.shape}")
    print(f"Averaged test shape: {test_averaged.shape}")


def save_baseline_statistics(baseline_stats, tb, baseline_indices, output_dir, monkey_id):
    """Prepare baseline normalization statistics for metadata inclusion."""
    unique_days = sorted(baseline_stats.keys())
    baseline_means = np.array([baseline_stats[day]['mean'] for day in unique_days])
    baseline_stds = np.array([baseline_stats[day]['std'] for day in unique_days])
    
    return {
        'baseline_means': baseline_means,
        'baseline_stds': baseline_stds,
        'baseline_days': np.array(unique_days),
        'baseline_time_range': [tb[baseline_indices[0]], tb[baseline_indices[-1]]],
        'baseline_indices': baseline_indices
    }


def normalize_tvsd_data(original_filepath, output_dir, monkey_id, batch_size):
    """Apply day-specific z-score normalization to TVSD neural data.
    
    Performs 6-step normalization using baseline period (-100 to 0ms) statistics
    computed separately for each recording day to account for session variability
    and electrode drift. Training data normalization uses own statistics to prevent
    data leakage, while test data uses training-derived parameters.
    
    Steps: (1) Load metadata, (2) Identify baseline indices, (3) Compute day-specific
    baseline stats from training data, (4) Normalize training data with chunked
    processing, (5) Apply same normalization to test data and average repetitions,
    (6) Return baseline parameters for metadata inclusion.
    
    Parameters
    ----------
    original_filepath : str
        Path to the raw THINGS_MUA_trials.mat file.
    output_dir : str
        Output directory for processed data files.
    monkey_id : str
        Monkey identifier for file naming.
    batch_size : int
        Batch size for chunked processing.
        
    Returns
    -------
    dict
        Baseline normalization statistics for metadata inclusion.
        
    Output Files
    ------------
    tvsd_{monkey}_split-train_normalized.h5 : (22,248, 300, 1024)
    tvsd_{monkey}_split-test_normalized.h5  : (3,000, 300, 1024) 
    tvsd_{monkey}_split-test_averaged.h5    : (100, 300, 1024)
    """
    # Load metadata and baseline information
    train_days, tb, train_shape, train_file, baseline_indices = load_metadata_and_baseline_info(
        original_filepath, output_dir, monkey_id)
    
    print(f"Baseline period: {tb[baseline_indices[0]]:.1f} to {tb[baseline_indices[-1]]:.1f} ms")
    
    # Compute day-specific baseline statistics
    baseline_stats = compute_day_specific_baseline_stats(
        train_days, train_file, baseline_indices, batch_size)
    
    # Normalize training data
    apply_normalization_to_training_data(
        train_days, train_file, baseline_stats, output_dir, monkey_id, batch_size)
    
    # Normalize test data
    apply_normalization_to_test_data(
        baseline_stats, original_filepath, output_dir, monkey_id)
    
    # Save baseline statistics
    baseline_metadata = save_baseline_statistics(baseline_stats, tb, baseline_indices, output_dir, monkey_id)
    
    return baseline_metadata

def load_things_mapping(mat_file_path):
    """Load THINGS image mapping from MATLAB file and return as DataFrames."""
    
    def decode_matlab_string(arr):
        if isinstance(arr, np.ndarray) and arr.dtype == np.uint16:
            return ''.join([chr(c) for c in arr.flatten()])
        return arr

    def extract_group_data(file_handle, group_name):
        group = file_handle[group_name]
        data = {}
        
        for field_name in group.keys():
            references = group[field_name][()]
            values = []
            
            for ref in references.flatten():
                obj = file_handle['#refs#'][ref]
                if isinstance(obj, h5py.Dataset):
                    value = obj[()]
                    value = decode_matlab_string(value)
                    values.append(value)
                else:
                    values.append(None)
            
            data[field_name] = values
        
        return pd.DataFrame(data)

    with h5py.File(mat_file_path, 'r') as f:
        train_df = extract_group_data(f, 'train_imgs')
        test_df = extract_group_data(f, 'test_imgs')
    
    return train_df, test_df



# =============================================================================
# Create dataset metadata
# =============================================================================
# Generate comprehensive metadata linking stimulus IDs to image files,
# object categories, and experimental conditions for both training and test sets.


def create_tvsd_metadata(original_filepath, things_mapping_file, output_dir, monkey_id, baseline_stats):
    """Create comprehensive metadata file for TVSD dataset.
    
    Generate metadata linking neural responses to THINGS database images through
    stimulus ID mapping. Converts MATLAB 1-based indices to Python 0-based indices
    to map trial-by-trial neural responses to specific image files and categories.
    Includes experimental conditions, baseline normalization parameters, and electrode
    quality metrics from THINGS_normMUA.mat.
    
    Mapping Process: Extract stimulus IDs from ALLMAT → Convert to 0-based indices
    → Lookup image info in THINGS DataFrames → Create aligned metadata arrays
    for training (22,248 trials), test individual (3,000 trials), and test 
    averaged (100 stimuli) datasets.
    
    Parameters
    ----------
    original_filepath : str
        Path to the raw THINGS_MUA_trials.mat file.
    things_mapping_file : str
        Path to the THINGS image mapping file.
    output_dir : str
        Output directory for processed data files.
    monkey_id : str
        Monkey identifier for file naming.
    baseline_stats : dict
        Baseline normalization statistics from normalize_tvsd_data.
        
    Output Files
    ------------
    tvsd_{monkey}_metadata.npz : Complete dataset metadata including stimulus
                                mappings, experimental conditions, baseline stats,
                                and electrode quality metrics
    """
    print("Creating dataset metadata...")
    
    with h5py.File(original_filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
        tb = f['tb'][:].flatten()
    
    # Load electrode quality metrics from THINGS_normMUA.mat
    norm_mua_filepath = os.path.join(os.path.dirname(original_filepath), "THINGS_normMUA.mat")
    if not os.path.exists(norm_mua_filepath):
        raise FileNotFoundError(f"THINGS_normMUA.mat not found at: {norm_mua_filepath}")
    
    with h5py.File(norm_mua_filepath, 'r') as f:
        SNR = f['SNR'][:]
        SNR_max = f['SNR_max'][:]
        oracle = f['oracle'][:]
    
    train_df, test_df = load_things_mapping(things_mapping_file)
    
    train_idx = ALLMAT[1, :]
    test_idx = ALLMAT[2, :]
    train_mask = test_idx == 0
    test_mask = test_idx != 0
    
    # Training metadata
    train_stimulus_ids = train_idx[train_mask]
    train_days = ALLMAT[5, train_mask]
    train_sequence_pos = ALLMAT[4, train_mask]
    
    # Test metadata
    test_stimulus_ids = test_idx[test_mask]
    test_days = ALLMAT[5, test_mask]
    test_sequence_pos = ALLMAT[4, test_mask]
    
    # Map training stimuli to image info
    train_img_files = []
    train_img_concepts = []
    
    for stim_id in train_stimulus_ids:
        stim_id = int(stim_id)
        row = train_df.iloc[stim_id - 1]
        train_img_files.append(row['things_path'].split('\\')[-1])
        train_img_concepts.append(row['class'])
    
    # Map test stimuli to image info
    test_img_files = []
    test_img_concepts = []
    
    for stim_id in test_stimulus_ids:
        stim_id = int(stim_id)
        row = test_df.iloc[stim_id - 1]
        test_img_files.append(row['things_path'].split('\\')[-1])
        test_img_concepts.append(row['class'])
    
    # Create averaged test metadata
    unique_test_ids = np.unique(test_stimulus_ids)
    test_avg_img_files = []
    test_avg_img_concepts = []
    
    for stim_id in unique_test_ids:
        stim_id = int(stim_id)
        row = test_df.iloc[stim_id - 1]
        test_avg_img_files.append(row['things_path'].split('\\')[-1])
        test_avg_img_concepts.append(row['class'])
    
    metadata = {
        'train_img_ids': train_stimulus_ids,
        'train_img_files': np.array(train_img_files),
        'train_img_concepts': np.array(train_img_concepts),
        'train_days': train_days,
        'train_sequence_pos': train_sequence_pos,
        'test_img_ids': test_stimulus_ids,
        'test_img_files': np.array(test_img_files),
        'test_img_concepts': np.array(test_img_concepts),
        'test_days': test_days,
        'test_sequence_pos': test_sequence_pos,
        'test_avg_img_ids': unique_test_ids,
        'test_avg_img_files': np.array(test_avg_img_files),
        'test_avg_img_concepts': np.array(test_avg_img_concepts),
        'times': tb,
        'monkey_id': monkey_id,
        'n_electrodes': 1024,
        'baseline_means': baseline_stats['baseline_means'],
        'baseline_stds': baseline_stats['baseline_stds'], 
        'baseline_days': baseline_stats['baseline_days'],
        'baseline_time_range': baseline_stats['baseline_time_range'],
        'baseline_indices': baseline_stats['baseline_indices'],
        'SNR': SNR,
        'SNR_max': SNR_max,
        'oracle': oracle
    }
    
    metadata_file = os.path.join(output_dir, f'tvsd_{monkey_id}_metadata.npz')
    np.savez(metadata_file, **metadata)
    
    print(f"Training trials: {len(train_stimulus_ids)}")
    print(f"Test trials: {len(test_stimulus_ids)}")
    print(f"Unique test stimuli: {len(unique_test_ids)}")
    print(f"Time points: {len(tb)} ({tb[0]:.1f} to {tb[-1]:.1f} ms)")