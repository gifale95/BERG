import numpy as np
import pandas as pd

# Load CSV files provided by Greta Tuckute
targets_csv = "tuckute_2024/lang_lh_netw_train_mean_beta-control-neural-T_items1-1000.csv"
stimset_csv = "tuckute_2024/beta-control-neural_stimset_D_light.csv"

df_targets = pd.read_csv(targets_csv)
df_stimset = pd.read_csv(stimset_csv)

# Extract training data 
train_stimulus_ids = df_targets.iloc[:, 0].to_numpy() 
train_targets = df_targets.iloc[:, 1].to_numpy()     

# Create a mapping from stimsetid to sentence
stimset_dict = dict(zip(df_stimset['stimsetid'], df_stimset['sentence']))

# Get training sentences in the same order as train_stimulus_ids
train_sentences = np.array([stimset_dict[stim_id] for stim_id in train_stimulus_ids])

# Extract Drive sentences (cond == 'D')
drive_data = df_stimset[df_stimset['cond'] == 'D']
drive_stimulus_ids = drive_data['stimsetid'].to_numpy()
drive_sentences = drive_data['sentence'].to_numpy()

# Extract Suppress sentences (cond == 'S')
suppress_data = df_stimset[df_stimset['cond'] == 'B']
suppress_stimulus_ids = suppress_data['stimsetid'].to_numpy()
suppress_sentences = suppress_data['sentence'].to_numpy()

# ROI information with noise ceilings
rois = [
    'lang_LH_AntTemp', 
    'lang_LH_IFG', 
    'lang_LH_IFGorb',
    'lang_LH_MFG', 
    'lang_LH_PostTemp', 
    'lang_LH_netw'
]

noise_ceiling = np.array([
    0.457712231,  # lang_LH_AntTemp
    0.452484144,  # lang_LH_IFG
    0.452201886,  # lang_LH_IFGorb
    0.515445403,  # lang_LH_MFG
    0.57890406,   # lang_LH_PostTemp
    0.559154577   # lang_LH_netw
])

noise_ceiling_snr = np.array([
    0.230227243,  # lang_LH_AntTemp
    0.226915578,  # lang_LH_IFG
    0.22673763,   # lang_LH_IFGorb
    0.269002404,  # lang_LH_MFG
    0.31750692,   # lang_LH_PostTemp
    0.3016193     # lang_LH_netw
])

# Build metadata dictionary
metadata = {
    # ROI information
    'rois': rois,
    'noise_ceiling': noise_ceiling,
    'noise_ceiling_snr': noise_ceiling_snr,
    
    # Model information
    'optimal_layer': 22,
    
    # Training data (n=1000 sentences, ordered by targets CSV)
    'train_stimulus_ids': train_stimulus_ids,
    'train_sentences': train_sentences,
    'train_targets': train_targets,
    
    # Test data - Drive (n=250)
    'drive_stimulus_ids': drive_stimulus_ids,
    'drive_sentences': drive_sentences,
    
    # Test data - Suppress (n=250)
    'suppress_stimulus_ids': suppress_stimulus_ids,
    'suppress_sentences': suppress_sentences,
}

# Save metadata
np.save(
    '/Volumes/Extreme SSD/brain-encoding-response-generator/encoding_models/'
    'modality-fmri/train_dataset-tuckute_2024/model-GPT2_XL/metadata/metadata.npy',
    metadata,
    allow_pickle=True
)
