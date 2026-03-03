import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import scipy.io


# =============================================================================
# Split training and test data
# =============================================================================
# Load raw neural data and split into training and test partitions based on
# stimulus type. Training data contains single presentations of 22,248 images,
# while test data contains 30 repetitions of 100 images for noise ceiling estimation.


def split_tvsd_data(filepath, output_dir, monkey_id, batch_size, create_splits=True):
    """Split TVSD neural data into training and test partitions.
    
    Load raw neural recordings (25,248 trials) and separate based on stimulus type.
    Training trials (test_idx=0) contain single presentations of THINGS images,
    while test trials (test_idx≠0) contain 30 repetitions of 100 images for 
    noise ceiling estimation. Uses chunked processing for memory efficiency.
    
    Optionally creates 4 random training splits by shuffling training indices.
    
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
    create_splits : bool, optional
        Whether to create 4 random training splits (default: True).
        
    Returns
    -------
    shuffled_indices : ndarray or None
        Shuffled training indices if create_splits=True, otherwise None.
        
    Output Files
    ------------
    tvsd_{monkey}_all_training_splits.h5 : (22,248, 1024, 300)
    tvsd_{monkey}_split-test.h5  : (3,000, 1024, 300)
    tvsd_{monkey}_split-test_averaged.h5   : (100, 1024, 300)
    
    If create_splits=True, additionally:
    tvsd_{monkey}_single_training_split_1.h5 : (5,562, 1024, 300)
    tvsd_{monkey}_single_training_split_2.h5 : (5,562, 1024, 300)
    tvsd_{monkey}_single_training_split_3.h5 : (5,562, 1024, 300)
    tvsd_{monkey}_single_training_split_4.h5 : (5,562, 1024, 300)
    """
    with h5py.File(filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
        test_idx = ALLMAT[2, :]
        train_mask = test_idx == 0
        test_mask = test_idx != 0
        test_stimuli = test_idx[test_mask]
        
        n_train = np.sum(train_mask)
        n_test = np.sum(test_mask)
        
        print(f"Training trials: {n_train}")
        print(f"Test trials: {n_test}")
        
        allmua_dataset = f['ALLMUA']
        total_trials = allmua_dataset.shape[1]
        
        # Process training data
        print("Processing training data...")
        train_file = os.path.join(output_dir, f'tvsd_{monkey_id}_all_training_splits.h5')
        
        with h5py.File(train_file, 'w') as train_h5:
            train_dataset = train_h5.create_dataset('neural_data', 
                                                   shape=(n_train, 1024, 300), 
                                                   dtype='float32')
            
            train_idx = 0
            for start_idx in tqdm(range(0, total_trials, batch_size), desc="Training batches"):
                end_idx = min(start_idx + batch_size, total_trials)
                
                chunk_data = allmua_dataset[:, start_idx:end_idx, :]
                chunk_train_mask = train_mask[start_idx:end_idx]
                
                if np.any(chunk_train_mask):
                    train_chunk = chunk_data[:, chunk_train_mask, :].transpose(1, 2, 0)
                    n_train_chunk = train_chunk.shape[0]
                    train_dataset[train_idx:train_idx + n_train_chunk] = train_chunk
                    train_idx += n_train_chunk
        
        # Process test data
        print("Processing test data...")
        test_indices = np.where(test_mask)[0]
        test_data = allmua_dataset[:, test_indices, :].transpose(1, 2, 0)
        
        test_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test.h5')
        with h5py.File(test_file, 'w') as test_h5:
            test_h5.create_dataset('neural_data', data=test_data)
                
        
        # Process test data averaged
        print("Processing test data averaged...")
        unique_test_ids = np.unique(test_stimuli)
        test_averaged = np.zeros((len(unique_test_ids), test_data.shape[1], test_data.shape[2]), dtype='float32')
        
        for i, stimulus_id in enumerate(tqdm(unique_test_ids, desc="Averaging test data")):
            mask = test_stimuli == stimulus_id
            test_averaged[i] = np.mean(test_data[mask], axis=0)
            
        averaged_test_file = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test_averaged.h5')
            
        with h5py.File(averaged_test_file, 'w') as f_out:
            f_out.create_dataset('neural_data', data=test_averaged)
        
        print(f"Training shape: ({n_train}, 1024, 300)")
        print(f"Test shape: {test_data.shape}")
        print(f"Averaged test shape: {test_averaged.shape}")
        
        if create_splits:
            print("")
            print("Creating 4 random training splits...")
            
            # Set random seed for reproducibility
            seed = 20200220
            np.random.seed(seed)
            
            # Shuffle training indices
            shuffled_indices = np.random.permutation(n_train)
            
            split_size = n_train // 4
            
            # Load all training data
            with h5py.File(train_file, 'r') as f_train:
                train_data = f_train['neural_data'][:]
            
            # Create 4 individual split files
            for split_idx in range(1, 5):
                start_idx = (split_idx - 1) * split_size
                end_idx = split_idx * split_size
                
                split_indices = shuffled_indices[start_idx:end_idx]
                split_data = train_data[split_indices]
                
                split_file = os.path.join(output_dir, f'tvsd_{monkey_id}_single_training_split_{split_idx}.h5')
                
                with h5py.File(split_file, 'w') as f_split:
                    f_split.create_dataset('neural_data', data=split_data)
                
                print(f"Split {split_idx} shape: {split_data.shape}")
            
            return shuffled_indices
        else:
            return None
        
        
        
# =============================================================================
# Compute Noise Ceiling
# =============================================================================


def compute_noise_ceiling(original_filepath, test_filepath, monkey_id):
    """Compute ncsnr and noise ceiling from test data with repeated presentations.
    
    Estimates noise ceiling using the variance across 30 repeated presentations
    of 100 test images. The noise ceiling represents the maximum achievable
    prediction accuracy given measurement noise.
    
    Parameters
    ----------
    original_filepath : str
        Path to THINGS_MUA_trials.mat to extract stimulus IDs from ALLMAT.
    test_filepath : str
        Path to the processed test HDF5 file (3,000, 1024, 300).
    monkey_id : str
        Monkey identifier for saving results.
        
    Returns
    -------
    dict
        'ncsnr': (1024, 300) - Neural signal-to-noise ratio per electrode/timepoint
        'noise_ceiling': (1024, 300) - Noise ceiling in r² percentage units (0-100)
    """
    # =============================================================================
    # Load the TVSD neural responses for the test images
    # =============================================================================
    # Load test stimulus IDs from ALLMAT
    with h5py.File(original_filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
        trial_type = ALLMAT[2, :]
        test_mask = trial_type != 0
        stimulus_ids = trial_type[test_mask].astype(int)
    
    # Load test neural data
    with h5py.File(test_filepath, 'r') as f:
        neural_data = f['neural_data'][:].astype(np.float32)
    
    unique_test_images = np.unique(stimulus_ids)
    
    # Reshape the data to (samples, features)
    n_electrodes = neural_data.shape[1]
    n_timepoints = neural_data.shape[2]
    n_features = n_electrodes * n_timepoints
    neural_data = neural_data.reshape(neural_data.shape[0], n_features)
    
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
    img_reps = 30
    noise_ceiling = 100 * (ncsnr ** 2) / ((ncsnr ** 2) + (1 / img_reps))
    
    # Reshape the scores to (n_electrodes, n_timepoints)
    ncsnr = ncsnr.reshape(n_electrodes, n_timepoints)
    noise_ceiling = noise_ceiling.reshape(n_electrodes, n_timepoints)
    
    # =============================================================================
    # Return the ncsnr and noise ceiling
    # =============================================================================
    results = {
        'ncsnr': ncsnr,
        'noise_ceiling': noise_ceiling
    }
    
    return results
        


# =============================================================================
# Create Metadata
# =============================================================================


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


def create_tvsd_metadata(original_filepath, things_mapping_file, output_dir, monkey_id, create_splits=True, shuffled_indices=None):
    """Create comprehensive metadata file for TVSD dataset.
    
    Generate metadata linking neural responses to THINGS database images through
    stimulus ID mapping. Converts MATLAB 1-based indices to Python 0-based indices
    to map trial-by-trial neural responses to specific image files and categories.
    Includes experimental conditions, electrode
    quality metrics, and electrode-to-ROI mapping information.
    
    Optionally creates metadata for 4 random training splits using provided shuffled indices.
    
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
    create_splits : bool, optional
        Whether to create metadata for 4 random training splits (default: True).
    shuffled_indices : ndarray or None
        Shuffled training indices from split_tvsd_data. Required if create_splits=True.
        
    Output Files
    ------------
    tvsd_{monkey}_metadata.npz : Complete dataset metadata including stimulus
                                mappings, experimental conditions, baseline stats,
                                electrode quality metrics, and electrode mapping
    """
    
    print("Creating dataset metadata...")
    
    with h5py.File(original_filepath, 'r') as f:
        ALLMAT = f['ALLMAT'][:]
        tb = f['tb'][:].flatten()
    
    # Load electrode mapping file
    mapping_file = os.path.join(os.path.dirname(original_filepath), "_logs", "1024chns_mapping_20220105.mat")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Electrode mapping file not found at: {mapping_file}")
    
    mapping_data = scipy.io.loadmat(mapping_file)
    electrode_order = mapping_data["mapping"][0] - 1  # Convert to 0-based indexing
    
    # Create ROI assignment array (1024 elements with values 0, 1, 2)
    roi_assignments = np.zeros(1024, dtype=int)
    
    if monkey_id == 'monkeyN':
        roi_assignments[0:512] = 0    # V1
        roi_assignments[512:768] = 1  # V4  
        roi_assignments[768:1024] = 2 # IT
    elif monkey_id == 'monkeyF':
        roi_assignments[0:512] = 0    # V1
        roi_assignments[512:832] = 2  # IT
        roi_assignments[832:1024] = 1 # V4

    # ROI labels 
    roi_labels = np.array(['V1', 'V4', 'IT'])
    
    # Load electrode quality metrics from THINGS_normMUA.mat
    norm_mua_filepath = os.path.join(os.path.dirname(original_filepath), "THINGS_normMUA.mat")
    if not os.path.exists(norm_mua_filepath):
        raise FileNotFoundError(f"THINGS_normMUA.mat not found at: {norm_mua_filepath}")
    
    with h5py.File(norm_mua_filepath, 'r') as f:
        SNR = f['SNR'][:]
        SNR_max = f['SNR_max'][:]
    
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
        
    # Compute noise ceilings
    test_filepath = os.path.join(output_dir, f'tvsd_{monkey_id}_split-test.h5')
    nc_data = compute_noise_ceiling(original_filepath, test_filepath, monkey_id)
    ncsnr = nc_data["ncsnr"]
    noise_ceiling = nc_data["noise_ceiling"]
    
    metadata = {
        'utah_array': {
            'times': tb,
            'monkey_id': monkey_id,
            'n_electrodes': 1024,
            'electrode_order': electrode_order},
        'roi': {
            'roi_assignments': roi_assignments,
            'roi_labels': roi_labels},
        'encoding_model': {
            'all_training_splits': {
                'train_img_ids': train_stimulus_ids,
                'train_stimuli': np.array(train_img_files),
                'train_concepts': np.array(train_img_concepts),
                'train_days': train_days,
                'train_sequence_pos': train_sequence_pos
            },
            
            'test_img_ids': test_stimulus_ids,
            'test_stimuli': np.array(test_img_files),
            'test_concepts': np.array(test_img_concepts),
            'test_days': test_days,
            'test_sequence_pos': test_sequence_pos,
            
            'SNR': SNR,
            'SNR_max': SNR_max,
            'ncsnr': ncsnr,
            'noise_ceiling': noise_ceiling}
    }
    
    if create_splits:
        if shuffled_indices is None:
            raise ValueError("shuffled_indices must be provided when create_splits=True")
        
        n_train = len(train_stimulus_ids)
        split_size = n_train // 4
        
        for split_idx in range(1, 5):
            start_idx = (split_idx - 1) * split_size
            end_idx = split_idx * split_size
            
            split_indices = shuffled_indices[start_idx:end_idx]
            
            metadata['encoding_model'][f'single_training_split_{split_idx}'] = {
                'train_img_ids': train_stimulus_ids[split_indices],
                'train_stimuli': np.array(train_img_files)[split_indices],
                'train_concepts': np.array(train_img_concepts)[split_indices],
                'train_days': train_days[split_indices],
                'train_sequence_pos': train_sequence_pos[split_indices]
            }
    
    metadata_file = os.path.join(output_dir, f'tvsd_{monkey_id}_metadata.npy')
    np.save(metadata_file, metadata, allow_pickle=True)