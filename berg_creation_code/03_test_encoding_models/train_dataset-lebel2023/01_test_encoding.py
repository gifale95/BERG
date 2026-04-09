"""Test trained OPT-1.3B encoding models on held-out test stimuli.

Computes voxelwise Pearson correlation (CCabs), noise-ceiling-normalised
correlation (CCnorm), and per-ROI summary statistics.
"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import pearsonr


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(
    description='Test OPT-1.3B fMRI encoding models.')

parser.add_argument('--berg_dir', required=True, type=str,
    help='Path to the BERG data directory.')
parser.add_argument('--subjects', nargs='+', type=str,
    default=['UTS01', 'UTS02', 'UTS03'],
    help='Subject identifiers.  Default: UTS01 UTS02 UTS03.')
parser.add_argument('--trim_test', type=int, default=50,
    help='TRs trimmed from start of test features.  Default: 50.')
parser.add_argument('--trim_train', type=int, default=10,
    help='TRs trimmed from start of train features.  Default: 10.')

args = parser.parse_args()

print('>>> Test OPT-1.3B fMRI encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# ============================================================================
# Paths
# ============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-lebel2023')
results_dir = os.path.join(args.berg_dir, 'results', 'test_encoding_models',
    'modality-fmri', 'train_dataset-lebel2023', 'opt_1_3b_ridge')
model_meta_dir = os.path.join(args.berg_dir, 'encoding_models',
    'modality-fmri', 'train_dataset-lebel2023', 'model-opt_1_3b_ridge',
    'metadata')
os.makedirs(model_meta_dir, exist_ok=True)


# ============================================================================
# Evaluate each subject
# ============================================================================
for subject in args.subjects:
    print(f'\n{"="*60}')
    print(f'Evaluating subject: {subject}')
    print(f'{"="*60}')

    # ----------------------------------------------------------------
    # Load metadata
    # ----------------------------------------------------------------
    data_meta_path = os.path.join(data_dir,
        f'lebel2023_{subject}_metadata.npy')
    metadata = np.load(data_meta_path, allow_pickle=True).item()
    test_stories = list(metadata['encoding_model']['test_stories'])
    n_voxels = metadata['fmri']['n_voxels']
    noise_ceiling = metadata['encoding_model']['noise_ceiling']
    roi_dict = metadata['roi']

    # ----------------------------------------------------------------
    # Load test predictions
    # ----------------------------------------------------------------
    pred_path = os.path.join(results_dir,
        f'fmri_test_pred_{subject}.npy')
    pred = np.load(pred_path)
    print(f'  Test predictions shape: {pred.shape}')

    # ----------------------------------------------------------------
    # Load actual test responses
    # ----------------------------------------------------------------
    test_resp_trim = args.trim_test - args.trim_train  # 50 - 10 = 40
    test_path = os.path.join(data_dir,
        f'lebel2023_{subject}_split-test.h5')

    resp_parts = []
    with h5py.File(test_path, 'r') as hf:
        for story in test_stories:
            resp = hf[f'{story}/data'][:]
            resp_parts.append(resp[test_resp_trim:])
    actual = np.vstack(resp_parts)
    print(f'  Actual responses shape:  {actual.shape}')

    # ----------------------------------------------------------------
    # Compute voxelwise Pearson correlation (CCabs)
    # ----------------------------------------------------------------
    print(f'  Computing voxelwise correlations ...')
    corrs = np.zeros(n_voxels, dtype=np.float32)
    for v in range(n_voxels):
        r, _ = pearsonr(actual[:, v], pred[:, v])
        corrs[v] = r
    corrs = np.nan_to_num(corrs)

    print(f'    Mean r:        {np.mean(corrs):.4f}')
    print(f'    Median r:      {np.median(corrs):.4f}')
    print(f'    Max r:         {np.max(corrs):.4f}')
    print(f'    Voxels r>0.1:  {np.sum(corrs > 0.1)}')
    print(f'    Voxels r>0.3:  {np.sum(corrs > 0.3)}')
    print(f'    Voxels r>0.5:  {np.sum(corrs > 0.5)}')

    # ----------------------------------------------------------------
    # CCnorm = CCabs / CCmax
    # ----------------------------------------------------------------

    cc_norm = corrs / noise_ceiling
    cc_norm = np.nan_to_num(cc_norm)

    print(f'\n    CCnorm statistics (voxels with CCmax > 0.35):')
    good_mask = noise_ceiling > 0.35
    n_good = np.sum(good_mask)
    print(f'    N voxels:      {n_good}')
    if n_good > 0:
        print(f'    Mean CCnorm:   {np.mean(cc_norm[good_mask]):.4f}')
        print(f'    Median CCnorm: {np.median(cc_norm[good_mask]):.4f}')

    # ----------------------------------------------------------------
    # Per-ROI summary
    # ----------------------------------------------------------------
    if roi_dict:
        print(f'\n    Per-ROI encoding accuracy (mean r, CCmax>0.35 voxels):')
        print(f'    {"ROI":<20s} {"N voxels":>10s} {"Mean r":>10s} '
              f'{"Mean CCnorm":>12s} {"Mean CCmax":>12s}')
        print(f'    {"-"*64}')
        for roi_name in sorted(roi_dict.keys()):
            roi_mask = roi_dict[roi_name] & good_mask
            n_roi = np.sum(roi_mask)
            if n_roi > 0:
                mean_r = np.mean(corrs[roi_mask])
                mean_norm = np.mean(cc_norm[roi_mask])
                mean_ceil = np.mean(noise_ceiling[roi_mask])
                print(f'    {roi_name:<20s} {n_roi:>10d} {mean_r:>10.4f} '
                      f'{mean_norm:>12.4f} {mean_ceil:>12.4f}')

    # ----------------------------------------------------------------
    # Save results to metadata
    # ----------------------------------------------------------------
    model_metadata = metadata.copy()
    model_metadata['encoding_model']['correlation'] = corrs
    model_metadata['encoding_model']['cc_norm'] = cc_norm

    model_meta_path = os.path.join(model_meta_dir,
        f'metadata_{subject}.npy')
    np.save(model_meta_path, model_metadata)
    print(f'\n  Saved model metadata to: {model_meta_path}')


# ============================================================================
# Summary
# ============================================================================
print(f'\n{"="*60}')
print('Done.  All subjects evaluated.')
print(f'  Model metadata dir: {model_meta_dir}')
print(f'{"="*60}')


"""
Example usage
=============

python berg_creation_code/03_test_encoding_models/train_dataset-lebel2023/01_test_encoding.py \
    --berg_dir /Volumes/ExtremeSSD/brain-encoding-response-generator
"""