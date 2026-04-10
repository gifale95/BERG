"""Prepare the preprocessed neural data from the Podcast ECoG dataset
(Zada et al., Scientific Data 2025):
 - Load GPT-2 XL features and align sub-word tokens to word level,
 - Load subject's preprocessed high-gamma ECoG,
 - Epoch around word onsets, downsample,
 - Save neural data, aligned features, and metadata.

Parameters
----------
subject : str
    Subject identifier (e.g., 'sub-01').
dataset_dir : str
    Root directory of the Podcast ECoG dataset (OpenNeuro ds005574).
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
tmin : float
    Epoch start relative to word onset in seconds (default: -2.0).
tmax : float
    Epoch end relative to word onset in seconds (default: 2.0).
sfreq_resample : float
    Target sampling frequency after downsampling (default: 32.0).

Output Files Created (per subject):
────────────────────────────────────────────────────────────────
zada2025_{subject}_neural.h5    : (n_epochs, n_electrodes, n_lags)
zada2025_{subject}_features.npy : (n_epochs, 1600)
zada2025_{subject}_metadata.npy :

    'ecog':
        subject_id        : str              - Subject identifier
        n_electrodes      : int              - Number of electrodes
        n_lags            : int              - Number of time lags
        sfreq             : float            - Sampling frequency (32 Hz)
        tmin              : float            - Epoch start (-2.0s)
        tmax              : float            - Epoch end (2.0s)
        times             : (n_lags,)        - Time points in seconds
        ch_names          : (n_electrodes,)  - Electrode names
        ch_coords         : (n_electrodes,3) - Electrode coordinates

    'encoding_model':
        epoch_selection   : (n_epochs,)      - Word indices of surviving epochs
"""

import argparse
import os
import numpy as np
import pandas as pd
import h5py
import mne


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True, type=str,
                    help="Subject identifier (e.g., 'sub-01').")
parser.add_argument("--dataset_dir", required=True, type=str,
                    help="Root directory of the Podcast ECoG dataset.")
parser.add_argument("--berg_dir", required=True, type=str,
                    help="Directory of the BERG framework.")
parser.add_argument("--tmin", default=-2.0, type=float,
                    help="Epoch start relative to word onset (s).")
parser.add_argument("--tmax", default=2.0, type=float,
                    help="Epoch end relative to word onset (s).")
parser.add_argument("--sfreq_resample", default=32.0, type=float,
                    help="Target sampling frequency after downsampling (Hz).")
args = parser.parse_args()

print('>>> Podcast ECoG Data Preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:20} {}'.format(key, val))


# =============================================================================
# Output directory
# =============================================================================
output_dir = os.path.join(args.berg_dir, 'model_training_datasets',
                          'train_dataset-zada2025')
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# Load GPT-2 XL features and transcript
# =============================================================================
# The dataset provides pre-extracted GPT-2 XL features for every token in the
# transcript. We load layer 24 embeddings (1600-dim) and the token-level
# transcript which contains word_idx for mapping tokens back to words.
print("\nLoading GPT-2 XL features...")

feature_layer = 24
embedding_path = os.path.join(args.dataset_dir, 'stimuli', 'gpt2-xl',
                              'features.hdf5')
transcript_path = os.path.join(args.dataset_dir, 'stimuli', 'gpt2-xl',
                               'transcript.tsv')

with h5py.File(embedding_path, 'r') as f:
    token_embeddings = f[f'layer-{feature_layer}'][...]
print(f"  Token embeddings shape: {token_embeddings.shape}")

df_tokens = pd.read_csv(transcript_path, sep='\t', index_col=0)
print(f"  Token transcript rows:  {len(df_tokens)}")

assert len(df_tokens) == token_embeddings.shape[0], \
    "Mismatch between transcript rows and embedding rows"


# =============================================================================
# Average sub-word token features to word level
# =============================================================================
# GPT-2's tokenizer splits some words into multiple sub-word tokens.
# We average the embeddings of all tokens belonging to the same word,
# identified by the word_idx column.
print("\nAveraging token embeddings to word level...")

word_embeddings = []
for _, group in df_tokens.groupby('word_idx'):
    indices = group.index.to_numpy()
    word_embeddings.append(token_embeddings[indices].mean(0))
word_embeddings = np.stack(word_embeddings)

# Build word-level dataframe with onset/offset times
df_words = df_tokens.groupby('word_idx').agg(
    dict(word='first', start='first', end='last'))

n_words_total = len(df_words)
print(f"  Word embeddings shape:  {word_embeddings.shape}")
print(f"  Total words:            {n_words_total}")

assert word_embeddings.shape == (n_words_total, token_embeddings.shape[1])


# =============================================================================
# Load subject's preprocessed high-gamma ECoG
# =============================================================================
print(f"Loading high-gamma ECoG for {args.subject}...")

sub_id = args.subject.replace('sub-', '')
fif_path = os.path.join(args.dataset_dir, 'derivatives', 'ecogprep',
                        args.subject, 'ieeg',
                        f'{args.subject}_task-podcast_desc-highgamma_ieeg.fif')

raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
print(f"  Channels:      {len(raw.ch_names)}")
print(f"  Sampling freq:  {raw.info['sfreq']} Hz")
print(f"  Duration:       {raw.times[-1]:.1f} s")


# =============================================================================
# Downsample the continuous raw data before epoching
# =============================================================================
# Resampling the raw data first to 32 Hz
print(f"Downsampling raw to {args.sfreq_resample} Hz...")
raw = raw.resample(sfreq=args.sfreq_resample, npad='auto', verbose=False)
print(f"  New sampling freq: {raw.info['sfreq']} Hz")


# =============================================================================
# Create epochs around word onsets
# =============================================================================
# Each event marks the onset of a word in the continuous ECoG recording.
print(f"Creating epochs (tmin={args.tmin}, tmax={args.tmax})...")

events = np.zeros((n_words_total, 3), dtype=int)
events[:, 0] = (df_words.start.values * raw.info['sfreq']).astype(int)
events[:, 2] = 1  # event id

# Sort events chronologically (some words may have unordered timestamps)
sort_order = np.argsort(events[:, 0])
events = events[sort_order]

epochs = mne.Epochs(
    raw,
    events,
    tmin=args.tmin,
    tmax=args.tmax,
    baseline=None,
    proj=False,
    event_id=None,
    preload=True,
    event_repeated='merge',
    verbose=False,
)

n_epochs = len(epochs)
n_electrodes = len(epochs.ch_names)
n_lags = len(epochs.times)
print(f"  Epochs before drop: {len(events)}")
print(f"  Epochs after drop:  {n_epochs}")
print(f"  Shape: ({n_epochs}, {n_electrodes}, {n_lags})")


# =============================================================================
# Align features to surviving epochs
# =============================================================================
# Some epochs may be dropped by MNE (e.g., too close to recording boundaries).
# epochs.selection gives indices into the sorted events array.
# We need to map back to original word indices (from tutorial)
epoch_selection_sorted = epochs.selection
epoch_selection = sort_order[epoch_selection_sorted]
aligned_features = word_embeddings[epoch_selection]

print(f"  Aligned features shape: {aligned_features.shape}")
assert aligned_features.shape[0] == n_epochs


# =============================================================================
# Extract electrode metadata
# =============================================================================
ch_names = np.array(raw.info['ch_names'])

# Extract 3D coordinates from channel info (typically in MNI or head space)
ch_coords = np.array([ch['loc'][:3] for ch in raw.info['chs']])

# Get time axis
times = epochs.times

# =============================================================================
# Save neural data
# =============================================================================
print("Saving outputs...")

neural_file = os.path.join(output_dir,
                           f'zada2025_{args.subject}_neural.h5')
epochs_data = epochs.get_data(copy=False).astype(np.float32)
with h5py.File(neural_file, 'w') as f:
    f.create_dataset('neural_data', data=epochs_data)
print(f"  Neural data:  {neural_file}")

# Save aligned features
features_file = os.path.join(output_dir,
                             f'zada2025_{args.subject}_features.npy')
np.save(features_file, aligned_features.astype(np.float32))
print(f"  Features:     {features_file}")

# Save metadata
metadata = {
    'ecog': {
        'subject_id': args.subject,
        'n_electrodes': n_electrodes,
        'n_lags': n_lags,
        'sfreq': args.sfreq_resample,
        'tmin': args.tmin,
        'tmax': args.tmax,
        'times': times,
        'ch_names': ch_names,
        'ch_coords': ch_coords,
    },
    'encoding_model': {
    },
}

metadata_file = os.path.join(output_dir,
                             f'zada2025_{args.subject}_metadata.npy')
np.save(metadata_file, metadata)
