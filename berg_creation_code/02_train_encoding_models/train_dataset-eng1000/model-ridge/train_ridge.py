"""Train English1000 voxelwise encoding models on the LeBel et al. (2023)
natural language fMRI dataset and save the results into the BERG directory
structure.

This script uses the deep-fMRI-dataset repository (HuthLab) to train ridge
regression encoding models that predict BOLD fMRI responses from English1000
semantic word embeddings (985 dimensions). One model is trained per subject.

Before running this script, you must:

1. Clone the deep-fMRI-dataset repository:
   $ git clone git@github.com:HuthLab/deep-fMRI-dataset.git

2. Install it:
   $ cd deep-fMRI-dataset
   $ pip install .

3. Download the preprocessed data (requires datalad):
   $ sudo apt-get install datalad
   $ cd encoding
   $ python load_dataset.py -download_preprocess

   This will download ~20 GB of preprocessed BOLD data, stimulus features,
   and metadata into the repository's data directory. See the deep-fMRI-
   dataset README for alternative download locations and options.

For more information on the dataset and encoding models, see:
    LeBel, A., Wagner, L., Jain, S. et al. A natural language fMRI dataset
    for voxelwise encoding models. Sci Data 10, 555 (2023).
    https://doi.org/10.1038/s41597-023-02437-z

    Dataset: https://openneuro.org/datasets/ds003020
    Code: https://github.com/HuthLab/deep-fMRI-dataset

Parameters
----------
deep_fmri_repo : str
    Path to the cloned deep-fMRI-dataset repository.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG
subjects : list of str
    Subject identifiers to train models for. Default: all 8 subjects.
sessions : list of int
    Training sessions to use (1-5). Default: all 5 sessions.
trim : int
    Number of TRs to trim from each story edge. Default: 5.
ndelays : int
    Number of FIR delays for HRF estimation (at 2s TR). Default: 4.
nboots : int
    Number of bootstrap samples for ridge CV. Default: 50.
chunklen : int
    Length of chunks for bootstrap CV. Default: 40.
nchunks : int
    Number of chunks held out per bootstrap. Default: 125.
"""


import os
import sys
import argparse
import json
import numpy as np
import h5py
import logging
from os.path import join


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--deep_fmri_repo', type=str, required=True,
    help='Path to the cloned deep-fMRI-dataset repository.')
parser.add_argument('--berg_dir', type=str, required=True,
    help='Path to the BERG data directory.')
parser.add_argument('--subjects', nargs='+', type=str,
    default=['UTS01', 'UTS02', 'UTS03', 'UTS05', 'UTS06',
             'UTS07', 'UTS08'],
    help='Subject identifiers to train. Default: all 8 subjects.')
parser.add_argument('--sessions', nargs='+', type=int,
    default=[1, 2, 3, 4, 5],
    help='Training sessions (1-5). Default: all 5.')
parser.add_argument('--trim', type=int, default=5)
parser.add_argument('--ndelays', type=int, default=4)
parser.add_argument('--nboots', type=int, default=50)
parser.add_argument('--chunklen', type=int, default=40)
parser.add_argument('--nchunks', type=int, default=125)
parser.add_argument('--singcutoff', type=float, default=1e-10)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

print('>>> Train LeBel et al. (2023) English1000 encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Add the deep-fMRI-dataset repository to the Python path
# =============================================================================
encoding_dir = join(args.deep_fmri_repo, 'encoding')
assert os.path.isdir(encoding_dir), \
    f'Could not find encoding directory at: {encoding_dir}. ' \
    f'Make sure --deep_fmri_repo points to the deep-fMRI-dataset repository.'
sys.path.insert(0, encoding_dir)

from encoding_utils import apply_zscore_and_hrf, get_response
from feature_spaces import get_feature_space
from ridge_utils.ridge import bootstrap_ridge
from config import EM_DATA_DIR


# =============================================================================
# Define BERG output paths
# =============================================================================
berg_model_dir = join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-eng1000', 'model-ridge')
weights_dir = join(berg_model_dir, 'encoding_models_weights')
metadata_dir = join(berg_model_dir, 'metadata')


# =============================================================================
# Resolve train/test story split from session configuration
# =============================================================================
sessions = list(map(str, args.sessions))
with open(join(EM_DATA_DIR, 'sess_to_story.json'), 'r') as f:
    sess_to_story = json.load(f)

train_stories, test_stories = [], []
for sess in sessions:
    stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
    train_stories.extend(stories)
    if tstory not in test_stories:
        test_stories.append(tstory)

assert len(set(train_stories) & set(test_stories)) == 0, \
    'Train and test stories overlap!'
allstories = list(set(train_stories) | set(test_stories))

print('\nStory split:')
print(f'  Train stories: {len(train_stories)}')
print(f'  Test stories:  {len(test_stories)} ({test_stories})')


# =============================================================================
# Extract and downsample English1000 features (shared across subjects)
# =============================================================================
print('\nExtracting English1000 features...')
downsampled_feat = get_feature_space('eng1000', allstories)

print(f'  trim: {args.trim}, ndelays: {args.ndelays}')
delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat,
    args.trim, args.ndelays)
delPstim = apply_zscore_and_hrf(test_stories, downsampled_feat,
    args.trim, args.ndelays)
print(f'  Train stimulus matrix: {delRstim.shape}')
print(f'  Test stimulus matrix:  {delPstim.shape}')


# =============================================================================
# Train encoding models for each subject
# =============================================================================
alphas = np.logspace(1, 3, 10)

for subject in args.subjects:
    print(f'\n{"="*60}')
    print(f'Training encoding model for subject: {subject}')
    print(f'{"="*60}')

    # -----------------------------------------------------------------
    # Filter stories that actually exist for this subject
    # -----------------------------------------------------------------
    subject_dir = join(
        args.deep_fmri_repo,
        'data',
        'ds003020',
        'derivative',
        'preprocessed_data',
        subject
    )

    available_stories = {
        f.replace('.hf5', '')
        for f in os.listdir(subject_dir)
        if f.endswith('.hf5')
    }

    train_stories_sub = [s for s in train_stories if s in available_stories]
    test_stories_sub = [s for s in test_stories if s in available_stories]

    missing = set(train_stories) - available_stories
    if missing:
        logging.warning(f'{subject} missing stories: {sorted(missing)}')

    # -----------------------------------------------------------------
    # Recompute stimulus matrices for this subject
    # -----------------------------------------------------------------
    delRstim_sub = apply_zscore_and_hrf(train_stories_sub, downsampled_feat,
        args.trim, args.ndelays)
    delPstim_sub = apply_zscore_and_hrf(test_stories_sub, downsampled_feat,
        args.trim, args.ndelays)

    # -----------------------------------------------------------------
    # Load BOLD responses
    # -----------------------------------------------------------------
    zRresp = get_response(train_stories_sub, subject)
    zPresp = get_response(test_stories_sub, subject)
    n_voxels = zRresp.shape[1]

    print(f'  Train response: {zRresp.shape}')
    print(f'  Test response:  {zPresp.shape}')
    print(f'  Number of voxels: {n_voxels}')

    # -----------------------------------------------------------------
    # Fit ridge regression with bootstrap cross-validation
    # -----------------------------------------------------------------
    print(f'  Ridge parameters: nboots={args.nboots}, chunklen='
          f'{args.chunklen}, nchunks={args.nchunks}')

    wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim_sub, zRresp, delPstim_sub, zPresp, alphas,
        args.nboots, args.chunklen, args.nchunks,
        singcutoff=args.singcutoff, single_alpha=False,
        use_corr=False)

    print(f'  Mean test correlation: {np.mean(corrs):.4f}')
    print(f'  Max test correlation:  {np.max(corrs):.4f}')
    print(f'  Voxels with r > 0.1:   {np.sum(corrs > 0.1)}')

    # -----------------------------------------------------------------
    # Save encoding model weights
    # -----------------------------------------------------------------
    sub_weights_dir = join(weights_dir, f'sub-{subject}')
    os.makedirs(sub_weights_dir, exist_ok=True)
    np.savez(join(sub_weights_dir, 'weights.npz'), wt)
    print(f'  Saved weights to: {sub_weights_dir}')

    # -----------------------------------------------------------------
    # Compute noise ceiling from individual repeats
    # -----------------------------------------------------------------
    # The test story (wheretheressmoke) was presented once per scanning
    # session. The preprocessed HDF5 files store individual repeat
    # responses under the 'individual_repeats' key with shape
    # (n_repeats, n_TRs, n_voxels).
    #
    # The noise ceiling is computed using the regularized CCnorm method
    # from Schoppe et al. (2016) as described in LeBel et al. (2023):
    #
    #   1. CC_half: mean pairwise Pearson correlation across repeats,
    #      computed per voxel across time.
    #   2. CC_max (Spearman-Brown correction):
    #        CC_max = sqrt(2 / (1 + 1 / CC_half^2))
    #      This estimates the expected correlation between the true
    #      signal and the average of all repeats.
    #   3. Regularized noise ceiling:
    #        noise_ceiling = max(CC_max, CC_floor)
    #      where CC_floor = 0.3 prevents unbounded corrections for
    #      poorly-modeled voxels.
    print(f'  Computing noise ceiling from individual repeats...')
    test_story_name = test_stories_sub[0]
    test_story_path = join(subject_dir, f'{test_story_name}.hf5')
    hf = h5py.File(test_story_path, 'r')

    if 'individual_repeats' in hf.keys():
        repeats = hf['individual_repeats'][:]  # (n_repeats, n_TRs, n_voxels)
        hf.close()
        n_repeats = repeats.shape[0]
        print(f'    Found {n_repeats} individual repeats, '
              f'shape: {repeats.shape}')

        # Compute CC_half: mean pairwise correlation across repeats.
        pair_corrs = []
        for i in range(n_repeats):
            for j in range(i + 1, n_repeats):
                r_i = repeats[i]  # (n_TRs, n_voxels)
                r_j = repeats[j]
                # Z-score each repeat across time, then compute correlation.
                r_i_z = (r_i - r_i.mean(0)) / (r_i.std(0) + 1e-10)
                r_j_z = (r_j - r_j.mean(0)) / (r_j.std(0) + 1e-10)
                pair_corrs.append((r_i_z * r_j_z).mean(0))
        cc_half = np.mean(pair_corrs, axis=0)  # (n_voxels,)

        # Spearman-Brown correction: CC_max = sqrt(2 / (1 + 1/CC_half^2))
        cc_half_safe = np.clip(np.abs(cc_half), 1e-10, None)
        cc_max = np.sqrt(2.0 / (1.0 + 1.0 / (cc_half_safe ** 2)))

        # Regularize: floor CC_max at 0.3 (LeBel et al., 2023).
        cc_floor = 0.3
        noise_ceiling = np.maximum(cc_max, cc_floor)

        print(f'    Mean CC_half: {np.mean(cc_half):.4f}')
        print(f'    Mean noise ceiling (CC_max, floored): '
              f'{np.mean(noise_ceiling):.4f}')
    else:
        hf.close()
        print(f'    WARNING: individual_repeats not found in '
              f'{test_story_path}. Setting noise ceiling to NaN.')
        noise_ceiling = np.full(n_voxels, np.nan)

    # -----------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------
    os.makedirs(metadata_dir, exist_ok=True)
    metadata = {
        'fmri': {
            'subject_id': subject,
            'n_voxels': n_voxels,
            'train_stories': np.array(train_stories_sub),
            'test_stories': np.array(test_stories_sub),
        },
        'encoding_models': {
            'correlation': corrs,
            'noise_ceiling': noise_ceiling,
        },
    }

    metadata_path = join(metadata_dir, f'sub-{subject}.npy')
    np.save(metadata_path, metadata)
    print(f'  Saved metadata to: {metadata_path}')


print(f'\n{"="*60}')
print('Done. All encoding models trained and saved.')
print(f'BERG output directory: {berg_model_dir}')
print(f'{"="*60}')


"""
python berg_creation_code/02_train_encoding_models/train_dataset-eng1000/model-ridge/train_ridge.py \
    --deep_fmri_repo /Volumes/ExtremeSSD/Repositories/deep-fMRI-dataset \
    --berg_dir /Volumes/ExtremeSSD/brain-encoding-response-generator


python berg_creation_code/02_train_encoding_models/train_dataset-eng1000/model-ridge/train_ridge.py \
    --deep_fmri_repo /pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/deep-fMRI-dataset \
    --berg_dir /pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/BERG/brain-encoding-response-generator

"""