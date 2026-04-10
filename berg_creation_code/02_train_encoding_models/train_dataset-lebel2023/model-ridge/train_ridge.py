"""Train OPT-1.3B ridge regression encoding models on the LeBel et al. (2023)
deep-fMRI-dataset, following Antonello, Vaidya & Huth (NeurIPS 2023).

Pipeline: OPT hidden states (layer 18) → Lanczos downsample to TR →
z-score + FIR delays → ridge regression → voxelwise correlation.

Pipeline steps:
1. Model setup: Load OPT-1.3B from HuggingFace, extract layer 18 hidden states
2. Feature extraction: For each story, extract word-level hidden states using
   a dynamic context window (grow to 512 words, reset to 256)
3. Temporal alignment: Lanczos-downsample word-level features to TR rate (0.5 Hz)
4. Preprocessing: Z-score per story, then apply FIR delays (2, 4, 6, 8 s)
5. Response loading: Load preprocessed BOLD responses from prepared data
6. Model training: Voxelwise ridge regression with bootstrap cross-validation
7. Evaluation: Predict held-out test story, compute voxelwise correlation
8. Output: Ridge weights, test predictions, and updated metadata


Parameters
----------
deep_fmri_repo : str
    Path to the cloned deep-fMRI-dataset repository.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
subjects : list of str
    Subject identifiers. Default: UTS01, UTS02, UTS03.
model_name : str
    HuggingFace model ID. Default: facebook/opt-1.3b.
layer : int
    Layer to extract (1-indexed). Default: 18.
trim_train : int
    TRs to trim from start of training features. Default: 10.
trim_test : int
    TRs to trim from start of test features. Default: 50.
trim_end : int
    TRs to trim from end of all features. Default: 5.
ndelays : int
    Number of FIR delays (at 2 s TR). Default: 4.
nboots : int
    Number of bootstrap samples for ridge CV. Default: 5.
device : str
    Torch device. Default: auto.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
import logging
from os.path import join
from tqdm import tqdm


# ========================================================================
# Force all caches / temp files into workspace
# ========================================================================
# import os

# BASE = "/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch"

# os.environ["HF_HOME"] = f"{BASE}/hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = f"{BASE}/hf_cache"
# os.environ["TORCH_HOME"] = f"{BASE}/torch_cache"
# os.environ["XDG_CACHE_HOME"] = f"{BASE}/cache"

# os.environ["TMPDIR"] = f"{BASE}/tmp"
# os.environ["TEMP"] = f"{BASE}/tmp"
# os.environ["TMP"] = f"{BASE}/tmp"

# # create folders if missing
# os.makedirs(f"{BASE}/hf_cache", exist_ok=True)
# os.makedirs(f"{BASE}/torch_cache", exist_ok=True)
# os.makedirs(f"{BASE}/cache", exist_ok=True)
# os.makedirs(f"{BASE}/tmp", exist_ok=True)
### 


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(
    description='Train OPT-1.3B ridge regression encoding models for fMRI.')

parser.add_argument('--deep_fmri_repo', type=str, required=True,
    help='Path to the cloned deep-fMRI-dataset repository.')
parser.add_argument('--berg_dir', type=str, required=True,
    help='Path to the BERG data directory.')
parser.add_argument('--subjects', nargs='+', type=str,
    default=['UTS01', 'UTS02', 'UTS03'],
    help='Subject identifiers.  Default: UTS01 UTS02 UTS03.')
parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b',
    help='HuggingFace model identifier.  Default: facebook/opt-1.3b.')
parser.add_argument('--layer', type=int, default=18,
    help='Layer to extract hidden states from (1-indexed).  Default: 18.')
parser.add_argument('--context_min_words', type=int, default=256,
    help='Context window size after reset (in words).  Default: 256.')
parser.add_argument('--context_max_words', type=int, default=512,
    help='Context window size before reset (in words).  Default: 512.')
parser.add_argument('--trim_train', type=int, default=10,
    help='TRs to trim from start of training features.  Default: 10.')
parser.add_argument('--trim_test', type=int, default=50,
    help='TRs to trim from start of test features.  Default: 50.')
parser.add_argument('--trim_end', type=int, default=5,
    help='TRs to trim from end of all features.  Default: 5.')
parser.add_argument('--ndelays', type=int, default=4,
    help='Number of FIR delays.  Default: 4.')
parser.add_argument('--nboots', type=int, default=5,
    help='Number of bootstrap CV folds.  Default: 5.')
parser.add_argument('--chunklen', type=int, default=20,
    help='Chunk length for bootstrap CV.  Default: 20.')
parser.add_argument('--device', type=str, default='auto',
    help='Torch device (cpu / cuda / auto).  Default: auto.')

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

print('>>> Train OPT-1.3B encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# ============================================================================
# Add deep-fMRI-dataset ridge_utils to sys.path
# ============================================================================
encoding_dir = join(args.deep_fmri_repo, 'encoding')
sys.path.insert(0, encoding_dir)

from ridge_utils.interpdata import lanczosinterp2D    # noqa: E402
from ridge_utils.ridge import bootstrap_ridge          # noqa: E402
from ridge_utils.npp import zscore                     # noqa: E402

try:                                                   # noqa: E402
    from ridge_utils.utils import make_delayed
except ImportError:
    from ridge_utils.util import make_delayed


# ============================================================================
# Resolve device
# ============================================================================
if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
print(f'\nDevice: {device}')


# ============================================================================
# Data paths
# ============================================================================
data_dir = join(args.berg_dir, 'model_training_datasets',
                'train_dataset-lebel2023')

stimuli_path = join(data_dir, 'lebel2023_stimuli.h5')


# ############################################################################
#  OPT FEATURE EXTRACTION
# ############################################################################

def tokenize_story(tokenizer, words):
    """Tokenize words individually with leading-space convention, returning
    a word-to-token mapping. Skips empty strings and bare possessives."""
    bos_id = tokenizer.bos_token_id
    all_tokens     = [bos_id]
    real_word_indices = []
    word_first_tok = []
    word_last_tok  = []

    is_first = True
    for i, w in enumerate(words):
        if w.strip() == '' or w == "'s":
            continue
        real_word_indices.append(i)
        prefix = '' if is_first else ' '
        toks = tokenizer.encode(prefix + w, add_special_tokens=False)
        word_first_tok.append(len(all_tokens))
        all_tokens.extend(toks)
        word_last_tok.append(len(all_tokens) - 1)
        is_first = False

    return all_tokens, real_word_indices, word_first_tok, word_last_tok


def extract_opt_features_for_story(
    model, tokenizer, words, layer, device,
    context_min_words=256, context_max_words=512,
):
    """Extract one hidden-state vector per word using dynamic context windows.
    Context grows to context_max_words, then resets to context_min_words.
    Non-real words get a copy of the previous real word's vector.
    """
    
    hidden_dim = model.config.hidden_size
    all_tokens, real_word_indices, word_first_tok, word_last_tok = \
        tokenize_story(tokenizer, words)
    n_real = len(real_word_indices)

    if n_real == 0:
        return np.zeros((len(words), hidden_dim), dtype=np.float32)

    real_features   = np.zeros((n_real, hidden_dim), dtype=np.float32)
    phase_start     = 0
    next_to_assign  = 0

    for rw in range(n_real):
        words_in_ctx = rw - phase_start + 1
        at_end       = (rw == n_real - 1)

        if words_in_ctx < context_max_words and not at_end:
            continue

        # Build token context
        if phase_start == 0:
            ctx_tokens = all_tokens[:word_last_tok[rw] + 1]
        else:
            tok_start = word_first_tok[phase_start]
            tok_end   = word_last_tok[rw] + 1
            ctx_tokens = [all_tokens[0]] + all_tokens[tok_start:tok_end]

        input_ids = torch.tensor([ctx_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            hidden = (
                model(input_ids, output_hidden_states=True)
                .hidden_states[layer][0]
                .cpu().float().numpy()
            )

        for w in range(next_to_assign, rw + 1):
            if phase_start == 0:
                rel = word_last_tok[w]
            else:
                rel = word_last_tok[w] - word_first_tok[phase_start] + 1
            real_features[w] = hidden[rel]

        next_to_assign = rw + 1
        if not at_end:
            phase_start = max(0, rw - context_min_words + 1)

    # Map back to full word list
    features = np.zeros((len(words), hidden_dim), dtype=np.float32)
    last_feat = np.zeros(hidden_dim, dtype=np.float32)
    rp = 0
    for i in range(len(words)):
        if rp < n_real and real_word_indices[rp] == i:
            features[i] = real_features[rp]
            last_feat = real_features[rp]
            rp += 1
        else:
            features[i] = last_feat

    return features


# ============================================================================
# Load OPT model
# ============================================================================
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

print(f'\nLoading model: {args.model_name} ...')
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
)
model.eval()
model.to(device)

hidden_dim = model.config.hidden_size
n_layers   = model.config.num_hidden_layers
print(f'  hidden_dim={hidden_dim}, n_layers={n_layers}, '
      f'extracting layer {args.layer}')


# ============================================================================
# Load stimuli and extract features for all stories
# ============================================================================
first_meta_path = join(data_dir,
    f'lebel2023_{args.subjects[0]}_metadata.npy')
first_meta = np.load(first_meta_path, allow_pickle=True).item()
all_train_stories = list(first_meta['encoding_model']['train_stories'])
all_test_stories  = list(first_meta['encoding_model']['test_stories'])
all_stories = sorted(set(all_train_stories) | set(all_test_stories))

print(f'\nStories: {len(all_train_stories)} train, '
      f'{len(all_test_stories)} test')

# Extract word-level features and Lanczos-downsample per story
print(f'\nExtracting OPT-1.3B features and downsampling to TR rate ...')
downsampled_features = {}   # {story: (n_TRs, hidden_dim)}

with h5py.File(stimuli_path, 'r') as stim_hf:
    for story in tqdm(all_stories, desc='Feature extraction'):
        grp = stim_hf[story]
        words       = [w.decode() if isinstance(w, bytes) else w
                       for w in grp['words'][:]]
        word_onsets = grp['word_onsets'][:]
        tr_times    = grp['tr_times'][:]

        # Word-level hidden states -> (n_words, hidden_dim)
        word_features = extract_opt_features_for_story(
            model, tokenizer, words, args.layer, device,
            context_min_words=args.context_min_words,
            context_max_words=args.context_max_words,
        )

        # Lanczos downsample -> (n_TRs, hidden_dim)
        ds_feat = lanczosinterp2D(word_features, word_onsets, tr_times,
                                  window=3)
        downsampled_features[story] = ds_feat.astype(np.float32)

        tqdm.write(f'  {story}: {len(words)} words -> {ds_feat.shape[0]} TRs')

# Free GPU memory
del model
torch.cuda.empty_cache() if device.type == 'cuda' else None
print('Feature extraction complete.')


# ############################################################################
#  RIDGE REGRESSION PER SUBJECT
# ############################################################################

delays = range(1, args.ndelays + 1)
alphas = np.logspace(1, 4, 15)

for subject in args.subjects:
    print(f'\n{"="*60}')
    print(f'Training encoding model for subject: {subject}')
    print(f'{"="*60}')

    # Load subject metadata
    meta_path = join(data_dir, f'lebel2023_{subject}_metadata.npy')
    metadata = np.load(meta_path, allow_pickle=True).item()
    train_stories = list(metadata['encoding_model']['train_stories'])
    test_stories  = list(metadata['encoding_model']['test_stories'])
    n_voxels = metadata['fmri']['n_voxels']

    # ----------------------------------------------------------------
    # Build training stimulus matrix (z-score + FIR delays)
    # ----------------------------------------------------------------
    print(f'  Building training stimulus matrix '
          f'(trim_start={args.trim_train}, trim_end={args.trim_end}) ...')

    Rstim_parts = []
    for story in train_stories:
        feat = downsampled_features[story]
        trimmed = feat[args.trim_train:-args.trim_end]
        Rstim_parts.append(np.nan_to_num(zscore(trimmed)))
    Rstim = np.vstack(Rstim_parts)
    delRstim = make_delayed(Rstim, delays)

    # ----------------------------------------------------------------
    # Build test stimulus matrix
    # ----------------------------------------------------------------
    print(f'  Building test stimulus matrix '
          f'(trim_start={args.trim_test}, trim_end={args.trim_end}) ...')

    Pstim_parts = []
    for story in test_stories:
        feat = downsampled_features[story]
        trimmed = feat[args.trim_test:-args.trim_end]
        Pstim_parts.append(np.nan_to_num(zscore(trimmed)))
    Pstim = np.vstack(Pstim_parts)
    delPstim = make_delayed(Pstim, delays)

    # ----------------------------------------------------------------
    # Load training responses
    # ----------------------------------------------------------------
    print(f'  Loading training responses ...')

    train_path = join(data_dir, f'lebel2023_{subject}_split-train.h5')
    Rresp_parts = []
    with h5py.File(train_path, 'r') as hf:
        for story in train_stories:
            Rresp_parts.append(hf[f'{story}/data'][:])
    Rresp = np.vstack(Rresp_parts)

    # ----------------------------------------------------------------
    # Load test responses
    # ----------------------------------------------------------------
    print(f'  Loading test responses ...')

    test_resp_trim = args.trim_test - args.trim_train
    test_path = join(data_dir, f'lebel2023_{subject}_split-test.h5')
    Presp_parts = []
    with h5py.File(test_path, 'r') as hf:
        for story in test_stories:
            resp = hf[f'{story}/data'][:]
            Presp_parts.append(resp[test_resp_trim:])
    Presp = np.vstack(Presp_parts)

    print(f'  delRstim: {delRstim.shape}  Rresp: {Rresp.shape}')
    print(f'  delPstim: {delPstim.shape}  Presp: {Presp.shape}')

    # ----------------------------------------------------------------
    # Fit ridge regression with bootstrap cross-validation
    # ----------------------------------------------------------------
    nchunks = int(Rresp.shape[0] * 0.25 / args.chunklen)

    print(f'  Ridge parameters: nboots={args.nboots}, '
          f'chunklen={args.chunklen}, nchunks={nchunks}')
    print(f'  Alphas: {alphas[0]:.0f} to {alphas[-1]:.0f} '
          f'({len(alphas)} values)')
    print(f'  Fitting ridge regression ...')

    wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim, Rresp, delPstim, Presp, alphas,
        args.nboots, args.chunklen, nchunks,
        singcutoff=1e-10, single_alpha=False, use_corr=False)

    print(f'  Mean test correlation: {np.mean(corrs):.4f}')
    print(f'  Max test correlation:  {np.max(corrs):.4f}')
    print(f'  Voxels with r > 0.1:   {np.sum(corrs > 0.1)}')

    # ----------------------------------------------------------------
    # Test predictions
    pred = delPstim @ wt

    # Save weights
    # ----------------------------------------------------------------
    save_dir = join(args.berg_dir, 'encoding_models', 'modality-fmri',
        'train_dataset-lebel2023', 'model-opt_1_3b_ridge',
        'encoding_models_weights')
    os.makedirs(save_dir, exist_ok=True)

    weights = {
        'ridge_weights': wt,
        'ridge_alphas': valphas,
        'model_name': args.model_name,
        'layer': args.layer,
        'hidden_dim': hidden_dim,
        'ndelays': args.ndelays,
        'trim_train': args.trim_train,
        'trim_test': args.trim_test,
        'trim_end': args.trim_end,
        'context_min_words': args.context_min_words,
        'context_max_words': args.context_max_words,
    }

    weights_path = join(save_dir, f'weights_{subject}.npy')
    np.save(weights_path, weights)
    print(f'  Saved weights to: {weights_path}')

    # ----------------------------------------------------------------
    # Save test predictions
    # ----------------------------------------------------------------
    results_dir = join(args.berg_dir, 'results', 'test_encoding_models',
        'modality-fmri', 'train_dataset-lebel2023', 'opt_1_3b_ridge')
    os.makedirs(results_dir, exist_ok=True)

    pred_path = join(results_dir, f'fmri_test_pred_{subject}.npy')
    np.save(pred_path, pred.astype(np.float32))
    print(f'  Saved test predictions to: {pred_path}')

    # ----------------------------------------------------------------
    # Update metadata with encoding model results
    # ----------------------------------------------------------------
    metadata['encoding_model']['correlation'] = corrs
    np.save(meta_path, metadata)
    print(f'  Updated metadata with correlations: {meta_path}')


# ============================================================================
# Summary
# ============================================================================
print(f'\n{"="*60}')
print('Done.  All encoding models trained and saved.')
print(f'{"="*60}')


"""
Example usage
=============

python berg_creation_code/02_train_encoding_models/train_dataset-lebel2023/model-ridge/train_ridge.py \
    --deep_fmri_repo /Volumes/ExtremeSSD/Repositories/deep-fMRI-dataset \
    --berg_dir /Volumes/ExtremeSSD/brain-encoding-response-generator \
    --device cpu \
    --model_name facebook/opt-6.7b \
    --layer 27

python berg_creation_code/02_train_encoding_models/train_dataset-lebel2023/model-ridge/train_ridge.py \
    --deep_fmri_repo /pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/deep-fMRI-dataset \
    --berg_dir /pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/BERG/brain-encoding-response-generator \
    --device cuda \
    --model_name facebook/opt-6.7b \
    --layer 27

# Quick test run with fewer bootstraps:
python berg_creation_code/02_train_encoding_models/train_dataset-lebel2023/model-ridge/train_ridge.py \
    --deep_fmri_repo /Volumes/ExtremeSSD/Repositories/deep-fMRI-dataset \
    --berg_dir /Volumes/ExtremeSSD/brain-encoding-response-generator
    --device cuda --nboots 3 --subjects UTS03
"""