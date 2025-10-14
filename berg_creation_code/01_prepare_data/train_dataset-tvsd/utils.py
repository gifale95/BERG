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
# Create dataset metadata
# =============================================================================
# Generate comprehensive metadata linking stimulus IDs to image files,
# object categories, and experimental conditions for both training and test sets.


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


def create_tvsd_metadata(original_filepath, things_mapping_file, output_dir, monkey_id, baseline_stats):
    """Create comprehensive metadata file for TVSD dataset.
    
    Generate metadata linking neural responses to THINGS database images through
    stimulus ID mapping. Converts MATLAB 1-based indices to Python 0-based indices
    to map trial-by-trial neural responses to specific image files and categories.
    Includes experimental conditions, baseline normalization parameters, electrode
    quality metrics, and electrode-to-ROI mapping information.
    
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
    else:
        raise ValueError(f"Unknown monkey_id: {monkey_id}")
    
    # ROI labels (consistent across monkeys)
    roi_labels = np.array(['V1', 'V4', 'IT'])
    
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
        'electrode_order': electrode_order,
        'roi_assignments': roi_assignments,
        'roi_labels': roi_labels,
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
    print(f"ROI assignments - V1: {np.sum(roi_assignments == 0)}, V4: {np.sum(roi_assignments == 1)}, IT: {np.sum(roi_assignments == 2)}")