"""Prepare the LeBel et al. (2023) deep-fMRI-dataset for BERG model training:
 - extract and consolidate BOLD fMRI responses per subject,
 - extract stimulus data (words, word onset times, TR times) from TextGrids,
 - split training and test stories,
 - extract ROI masks from the pycortex database,
 - compute noise ceiling from individual test-story repeats,
 - save comprehensive per-subject metadata.

After preparation, the data is saved as:
 - Training responses: per-story HDF5 groups with shape (n_TRs, n_voxels)
 - Test responses: per-story HDF5 groups with shape (n_TRs, n_voxels),
   plus individual repeats for noise ceiling
 - Stimuli: per-story HDF5 groups with words, word onsets, and TR times
The data is saved in HDF5 and NumPy formats for efficient loading during
model training.

Parameters
----------
deep_fmri_repo : str
    Path to the cloned deep-fMRI-dataset repository.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
subjects : list of str
    Subject identifiers. Default: UTS01, UTS02, UTS03.


Output Files Created (per subject):
────────────────────────────────────────────────────────────────
lebel2023_stimuli.h5                  : Shared across subjects
    Per story group:
        words              : (n_words,)      - Word strings
        word_onsets        : (n_words,)      - Word onset times in seconds
        tr_times           : (n_TRs,)        - fMRI acquisition times in seconds

lebel2023_{subject}_split-train.h5    : Training BOLD responses
    Per story group:
        data               : (n_TRs, n_voxels) - Z-scored BOLD signal

lebel2023_{subject}_split-test.h5     : Test BOLD responses
    Per story group:
        data               : (n_TRs, n_voxels) - Averaged BOLD signal
        individual_repeats : (n_reps, n_TRs, n_voxels) - Per-repeat responses

lebel2023_{subject}_metadata.npy      :

    'fmri':
        subject_id          : str      - Subject identifier
        n_voxels            : int      - Number of cortical voxels
        tr                  : float    - Repetition time in seconds (2.0)
        voxel_size_mm       : float    - Isotropic voxel size (2.6)

    'roi':
        {roi_name}          : (n_voxels,) bool - Voxel mask per ROI

    'encoding_model':
        train_stories       : list     - Training story names
        test_stories        : list     - Test story names
        noise_ceiling       : (n_voxels,) - CCmax (Schoppe et al. 2016)
"""


import os
import sys
import json
import argparse
import numpy as np
import h5py
from os.path import join
from tqdm import tqdm


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(
    description='Prepare LeBel et al. (2023) fMRI data for BERG training.')

parser.add_argument('--deep_fmri_repo', type=str, required=True,
    help='Path to the cloned deep-fMRI-dataset repository.')
parser.add_argument('--berg_dir', type=str, required=True,
    help='Path to the BERG data directory.')
parser.add_argument('--subjects', nargs='+', type=str,
    default=['UTS01', 'UTS02', 'UTS03'],
    help='Subject identifiers.  Default: UTS01 UTS02 UTS03.')

args = parser.parse_args()

print('>>> LeBel et al. (2023) data preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# ============================================================================
# Add the deep-fMRI-dataset repository to sys.path
# ============================================================================
encoding_dir = join(args.deep_fmri_repo, 'encoding')
sys.path.insert(0, encoding_dir)


# Imports
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles  # noqa: E402
from ridge_utils.dsutils import make_word_ds                                   # noqa: E402
from config import DATA_DIR                                                    # noqa: E402


def get_story_wordseqs(stories):
    """Load word DataSequences from TextGrids (reimplemented to avoid tables)."""
    grids = load_textgrids(stories, DATA_DIR)
    with open(join(DATA_DIR, 'ds003020', 'derivative', 'respdict.json')) as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    return make_word_ds(grids, trfiles)


# ============================================================================
# Output directory
# ============================================================================
output_dir = join(args.berg_dir, 'model_training_datasets',
                  'train_dataset-lebel2023')
os.makedirs(output_dir, exist_ok=True)


# ============================================================================
# Discover stories and resolve train/test split
# ============================================================================
# Discover stories from the filesystem
# The file ess_to_story.json only covers the base 5 sessions (27 stories).  
# The extended dataset for UTS01-03 contains ~84 stories. 
# We find all stories that have both a preprocessed response file AND a TextGrid annotation.
#
# The test story "wheretheressmoke" was repeated across scanning sessions
# for noise ceiling estimation.  All other stories are used for training.

TEST_STORIES = ['wheretheressmoke']

# Find stories with TextGrids
textgrid_dir = join(DATA_DIR, 'ds003020', 'derivative', 'TextGrids')
stories_with_tg = {
    f.replace('.TextGrid', '')
    for f in os.listdir(textgrid_dir) if f.endswith('.TextGrid')
}

# Find stories with respdict entries (needed for TR times)
with open(join(DATA_DIR, 'ds003020', 'derivative', 'respdict.json')) as f:
    respdict = json.load(f)

# Stories usable for stimulus extraction = have both TextGrid + respdict
all_stories = sorted(stories_with_tg & set(respdict.keys()))

test_stories_all = [s for s in TEST_STORIES if s in all_stories]
train_stories_all = sorted(s for s in all_stories if s not in TEST_STORIES)

print(f'\nStories discovered (from filesystem):')
print(f'  Total unique: {len(all_stories)}')
print(f'  Training:     {len(train_stories_all)}')
print(f'  Test:         {len(test_stories_all)} ({test_stories_all})')


# ============================================================================
# Part 1 — Extract stimulus data (words, word onsets, TR times)
# ============================================================================
# This is shared across subjects since all subjects heard the same stories.

print('\n' + '='*60)
print('Part 1: Extracting stimulus data from TextGrids')
print('='*60)

stimuli_path = join(output_dir, 'lebel2023_stimuli.h5')


print(f'  Loading word sequences for {len(all_stories)} stories ...')
wordseqs = get_story_wordseqs(all_stories)

# HDF5 variable-length string type
str_dt = h5py.special_dtype(vlen=str)

with h5py.File(stimuli_path, 'w') as hf:
    for story in tqdm(all_stories, desc='  Saving stimuli'):
        ds = wordseqs[story]
        grp = hf.create_group(story)

        # Words as variable-length strings
        words_arr = np.array(list(ds.data), dtype=object)
        grp.create_dataset('words', data=words_arr, dtype=str_dt)

        # Word onset times (seconds)
        grp.create_dataset('word_onsets', data=np.array(ds.data_times,
                                                        dtype=np.float64))

        # TR acquisition times (seconds)
        grp.create_dataset('tr_times', data=np.array(ds.tr_times,
                                                        dtype=np.float64))

print(f'  Saved stimuli to: {stimuli_path}')


# ============================================================================
# Part 2 — Extract BOLD responses and metadata per subject
# ============================================================================

for subject in args.subjects:
    print(f'\n{"="*60}')
    print(f'Part 2: Processing subject {subject}')
    print(f'{"="*60}')

    # Path to the preprocessed HDF5 files in the deep-fMRI-dataset repo
    subject_resp_dir = join(DATA_DIR, 'ds003020', 'derivative',
                            'preprocessed_data', subject)

    # Find which stories actually exist for this subject
    available = {
        f.replace('.hf5', '')
        for f in os.listdir(subject_resp_dir) if f.endswith('.hf5')
    }
    train_stories_sub = sorted(s for s in train_stories_all if s in available)
    test_stories_sub = sorted(s for s in test_stories_all if s in available)
    missing = (set(train_stories_all) | set(test_stories_all)) - available
    if missing:
        print(f'  WARNING: {subject} missing {len(missing)} stories: '
              f'{sorted(missing)[:5]}{"..." if len(missing)>5 else ""}')

    print(f'  Available training stories: {len(train_stories_sub)}')
    print(f'  Available test stories:     {len(test_stories_sub)}')

    # ----------------------------------------------------------------
    # 2a) Consolidate training responses
    # ----------------------------------------------------------------
    train_path = join(output_dir, f'lebel2023_{subject}_split-train.h5')
    n_voxels = None

    if os.path.exists(train_path):
        print(f'  Training file exists, skipping: {train_path}')
        with h5py.File(train_path, 'r') as hf:
            first_story = list(hf.keys())[0]
            n_voxels = hf[f'{first_story}/data'].shape[1]
    else:
        print(f'  Consolidating training responses ...')
        with h5py.File(train_path, 'w') as out_hf:
            for story in tqdm(train_stories_sub, desc='  Training'):
                src = join(subject_resp_dir, f'{story}.hf5')
                with h5py.File(src, 'r') as in_hf:
                    data = in_hf['data'][:]  # (n_TRs, n_voxels)
                    if n_voxels is None:
                        n_voxels = data.shape[1]
                    out_hf.create_dataset(f'{story}/data',
                                          data=data.astype(np.float32))
        print(f'  Saved training responses ({len(train_stories_sub)} stories, '
              f'{n_voxels} voxels) to: {train_path}')

    # ----------------------------------------------------------------
    # 2b) Consolidate test responses (averaged + individual repeats)
    # ----------------------------------------------------------------
    test_path = join(output_dir, f'lebel2023_{subject}_split-test.h5')

    if os.path.exists(test_path):
        print(f'  Test file exists, skipping: {test_path}')
    else:
        print(f'  Consolidating test responses ...')
        with h5py.File(test_path, 'w') as out_hf:
            for story in tqdm(test_stories_sub, desc='  Test'):
                src = join(subject_resp_dir, f'{story}.hf5')
                with h5py.File(src, 'r') as in_hf:
                    # Averaged response
                    data = in_hf['data'][:]  # (n_TRs, n_voxels)
                    grp = out_hf.create_group(story)
                    grp.create_dataset('data', data=data.astype(np.float32))

                    # Individual repeats (if available)
                    if 'individual_repeats' in in_hf:
                        reps = in_hf['individual_repeats'][:]
                        grp.create_dataset('individual_repeats',
                                           data=reps.astype(np.float32))
                        print(f'    {story}: {data.shape[0]} TRs, '
                              f'{reps.shape[0]} repeats')
                    else:
                        print(f'    {story}: {data.shape[0]} TRs, '
                              f'no individual repeats')
        print(f'  Saved test responses to: {test_path}')

    # ----------------------------------------------------------------
    # 2c) Compute noise ceiling from individual repeats
    # ----------------------------------------------------------------
    # Uses the Schoppe et al. (2016) method as implemented by
    # Antonello et al. (2023, NeurIPS)
    #
    # CCmax is floored at 0.25 to regularise estimates for noisy
    # voxels (Antonello et al., Section 2.5).
    #
    # Test repeats are trimmed by 40 TRs from the start to match
    # the evaluation window (Antonello et al., Section 3.5)

    cc_max = np.full(n_voxels, np.nan)
    test_repeat_trim = 40  # Additional TRs to remove from start

    with h5py.File(test_path, 'r') as hf:
        for story in test_stories_sub:
            if f'{story}/individual_repeats' not in hf:
                print(f'    WARNING: No individual repeats for {story}')
                continue

            repeats = hf[f'{story}/individual_repeats'][:]
            N = repeats.shape[0]
            print(f'    {story}: {N} repeats, shape {repeats.shape}')

            # Trim the first 40 TRs to match test evaluation window
            repeats = repeats[:, test_repeat_trim:, :]
            print(f'    After trimming {test_repeat_trim} TRs: '
                  f'{repeats.shape}')

            # Mean response across repeats → (n_TRs, n_voxels)
            mean_resp = np.mean(repeats, axis=0)

            # TP: noise power — mean within-repeat temporal variance
            #   var across time for each repeat → (N, n_voxels)
            #   then average across repeats    → (n_voxels,)
            within_var = np.var(repeats, axis=1, ddof=1)
            TP = np.mean(within_var, axis=0)

            # SP: signal power
            var_mean = np.var(mean_resp, axis=0, ddof=1)
            SP = (1.0 / (N - 1)) * (N * var_mean - TP)
            SP = np.maximum(SP, 1e-10)

            # CCmax per voxel
            cc_max_story = np.nan_to_num(
                1.0 / np.sqrt(1.0 + (1.0 / N) * (TP / SP - 1.0)))

            # Floor at 0.25 (Antonello et al., Section 2.5)
            cc_max_story = np.maximum(cc_max_story, 0.25)

            # Use first test story (consistent with single-story
            # evaluation in both LeBel and Antonello papers)
            if np.all(np.isnan(cc_max)):
                cc_max = cc_max_story

    noise_ceiling = cc_max

    print(f'    Noise ceiling (CCmax, floored at 0.25):')
    print(f'      Mean:   {np.nanmean(noise_ceiling):.4f}')
    print(f'      Median: {np.nanmedian(noise_ceiling):.4f}')
    print(f'      Max:    {np.nanmax(noise_ceiling):.4f}')
    print(f'      Voxels with CCmax > 0.35: '
          f'{np.sum(noise_ceiling > 0.35)}')
    print(f'      Voxels with CCmax > 0.50: '
          f'{np.sum(noise_ceiling > 0.50)}')

    # ----------------------------------------------------------------
    # 2d) Extract ROI masks from pycortex
    # ----------------------------------------------------------------
    print(f'  Extracting ROI masks from pycortex ...')

    db_path = join(args.deep_fmri_repo, 'data', 'ds003020', 'derivative',
                   'pycortex-db')

    roi_dict = {}
    if os.path.isdir(db_path):
        import nibabel as nib
        import cortex
        import cortex.utils as cu
        import cortex.database
        import cortex.dataset.braindata as braindata

        cortex.database.default_filestore = db_path
        new_db = cortex.database.Database(db_path)
        cortex.db = new_db
        cu.db = new_db
        braindata.db = new_db

        # Load brain mask (nibabel: x,y,z → pycortex: z,y,x)
        xfm_dir = join(db_path, subject, 'transforms')
        xfm_names = [x for x in os.listdir(xfm_dir) if not x.startswith('.')]
        assert xfm_names, f'No transforms found for {subject}'
        xfmname = xfm_names[0]

        mask_path = join(xfm_dir, xfmname, 'mask_thick.nii.gz')
        assert os.path.exists(mask_path), \
            f'mask_thick.nii.gz not found: {mask_path}'
        mask_vol = nib.load(mask_path).get_fdata()
        mask_bool = np.transpose(mask_vol, (2, 1, 0)).astype(bool)
        print(f'    Mask shape (transposed): {mask_bool.shape}, '
              f'{np.count_nonzero(mask_bool)} voxels')

        # Map ROIs to encoding-model voxel space
        rois = cu.get_roi_masks(subject, xfmname)
        for name in sorted(rois):
            flat = rois[name][mask_bool].astype(bool)
            n = np.count_nonzero(flat)
            if n > 0:
                roi_dict[name.strip()] = flat
                print(f'      {name}: {n} voxels')

        print(f'    Total ROIs with voxels: {len(roi_dict)}')
    else:
        print(f'    WARNING: pycortex-db not found at {db_path}. '
              f'Skipping ROI extraction.')

    # ----------------------------------------------------------------
    # 2e) Save metadata
    # ----------------------------------------------------------------
    metadata = {
        'fmri': {
            'subject_id': subject,
            'n_voxels': n_voxels,
            'tr': 2.0,
            'voxel_size_mm': 2.6,
        },
        'roi': roi_dict,
        'encoding_model': {
            'train_stories': np.array(train_stories_sub),
            'test_stories': np.array(test_stories_sub),
            'noise_ceiling': noise_ceiling,
        },
    }

    metadata_path = join(output_dir, f'lebel2023_{subject}_metadata.npy')
    np.save(metadata_path, metadata)
    print(f'  Saved metadata to: {metadata_path}')


# ============================================================================
# Summary
# ============================================================================
print(f'\n{"="*60}')
print('Done.  All data prepared.')
print(f'  Output directory: {output_dir}')
print(f'{"="*60}')


"""
Example usage
=============

python berg_creation_code/01_prepare_data/train_dataset-lebel2023/prepare_rige.py \
    --deep_fmri_repo /Volumes/ExtremeSSD/Repositories/deep-fMRI-dataset \
    --berg_dir /Volumes/ExtremeSSD/brain-encoding-response-generator
    
    
python berg_creation_code/01_prepare_data/train_dataset-lebel2023/prepare_rige.py \
    --deep_fmri_repo /pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/deep-fMRI-dataset \
    --berg_dir /pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/BERG/brain-encoding-response-generator \
"""