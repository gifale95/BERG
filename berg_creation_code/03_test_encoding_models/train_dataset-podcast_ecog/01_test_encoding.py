"""Evaluate encoding model accuracy using 2-fold cross-validation, matching the
paper's methodology (Zada et al., Scientific Data 2025).

The podcast is split into two halves. For each fold, the model is trained on
one half and evaluated on the other by correlating predicted and actual neural
activity per electrode and lag. The two fold correlations are averaged to
produce one correlation value per electrode × lag.

Results are saved into the subject's metadata file.

Parameters
----------
subject : str
    Subject identifier (e.g., 'sub-01').
berg_dir : str
    Directory of the BERG framework.
"""

import argparse
import os
import numpy as np
import h5py
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from himalaya.backend import set_backend, get_backend
from himalaya.ridge import RidgeCV


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True, type=str,
                    help="Subject identifier (e.g., 'sub-01').")
parser.add_argument("--berg_dir", required=True, type=str,
                    help="Directory of the BERG framework.")
args = parser.parse_args()

print('>>> Test Podcast ECoG Encoding Model (2-fold CV) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:20} {}'.format(key, val))

# Set himalaya backend
if torch.cuda.is_available():
    set_backend("torch_cuda")
    print("\nUsing CUDA backend")
else:
    set_backend("numpy")
    print("\nUsing NumPy backend")


# =============================================================================
# Helpers
# =============================================================================
def to_numpy(x):
    """Convert torch tensor or array to numpy."""
    if hasattr(x, 'numpy'):
        return x.numpy(force=True)
    return np.asarray(x)


def correlation_score_vectorized(y_true, y_pred):
    """Pearson correlation per column (vectorized).

    Parameters
    ----------
    y_true : (n_samples, n_targets)
    y_pred : (n_samples, n_targets)

    Returns
    -------
    correlations : (n_targets,)
    """
    y_true = y_true - y_true.mean(axis=0)
    y_pred = y_pred - y_pred.mean(axis=0)
    num = (y_true * y_pred).sum(axis=0)
    den = np.sqrt((y_true ** 2).sum(axis=0) * (y_pred ** 2).sum(axis=0))
    # Avoid division by zero for constant columns
    den = np.where(den == 0, 1.0, den)
    return num / den


# =============================================================================
# Load data
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
                        'train_dataset-podcast_ecog')

# Load metadata
metadata_path = os.path.join(data_dir,
                             f'podcast_ecog_{args.subject}_metadata.npy')
metadata = np.load(metadata_path, allow_pickle=True).item()

n_electrodes = metadata['ecog']['n_electrodes']
n_lags = metadata['ecog']['n_lags']
times = metadata['ecog']['times']

# Load features (X)
features_path = os.path.join(data_dir,
                             f'podcast_ecog_{args.subject}_features.npy')
X = np.load(features_path).astype(np.float32)

# Load neural data (Y)
neural_path = os.path.join(data_dir,
                           f'podcast_ecog_{args.subject}_neural.h5')
with h5py.File(neural_path, 'r') as f:
    Y = f['neural_data'][:].reshape(f['neural_data'].shape[0], -1).astype(np.float32)

print(f"\nSubject {args.subject}: {n_electrodes} electrodes, {n_lags} lags")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")


# =============================================================================
# 2-fold cross-validation (matching paper)
# =============================================================================
print("\nRunning 2-fold cross-validation...")

alphas = np.logspace(1, 10, 10)
inner_cv = KFold(n_splits=5, shuffle=False)
outer_cv = KFold(n_splits=2, shuffle=False)

fold_correlations = []

for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X)):
    print(f"\n  Fold {fold_idx + 1}/2: "
          f"train={len(train_index)}, test={len(test_index)}")

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Standardize X
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Standardize Y
    scaler_Y = StandardScaler()
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_test = scaler_Y.transform(Y_test)

    # Fit RidgeCV
    ridge = RidgeCV(alphas=alphas, cv=inner_cv, fit_intercept=True)
    ridge.fit(X_train, Y_train)

    best_alpha = to_numpy(ridge.best_alphas_)
    print(f"    Best alpha: {best_alpha}")

    # Predict and compute correlation
    Y_pred = to_numpy(ridge.predict(X_test))
    Y_test_np = to_numpy(Y_test)

    corr = correlation_score_vectorized(Y_test_np, Y_pred)
    corr = corr.reshape(n_electrodes, n_lags)
    fold_correlations.append(corr)

    print(f"    Mean correlation: {corr.mean():.4f}")
    print(f"    Max correlation:  {corr.max():.4f}")

# Average correlations across folds (matching paper)
correlation_results = np.stack(fold_correlations).mean(axis=0)

print(f"\n--- Results ---")
print(f"Correlation results shape: {correlation_results.shape}")
print(f"Mean correlation:          {correlation_results.mean():.4f}")
print(f"Max correlation:           {correlation_results.max():.4f}")

# Per-electrode summary (max across lags)
max_per_electrode = correlation_results.max(axis=1)
print(f"Max-over-lags per electrode: "
      f"min={max_per_electrode.min():.4f}, "
      f"median={np.median(max_per_electrode):.4f}, "
      f"max={max_per_electrode.max():.4f}")

# Best lag per electrode
best_lag_idx = correlation_results.argmax(axis=1)
best_lag_times = times[best_lag_idx]
print(f"Best lag times (s):          "
      f"min={best_lag_times.min():.3f}, "
      f"median={np.median(best_lag_times):.3f}, "
      f"max={best_lag_times.max():.3f}")

# Number of electrodes with positive encoding
n_positive = (max_per_electrode > 0).sum()
print(f"Electrodes with r > 0:       {n_positive}/{n_electrodes}")


# =============================================================================
# Save encoding accuracy into metadata
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-ecog',
                        'train_dataset-podcast_ecog', 'model-gpt2_xl',
                        'metadata')
os.makedirs(save_dir, exist_ok=True)

file_name = f'metadata_{args.subject}.npy'
metadata_save_path = os.path.join(save_dir, file_name)

# Start from the preparation metadata and add encoding results
metadata['encoding_model']['correlation_results'] = correlation_results

np.save(metadata_save_path, metadata)
print(f"\nMetadata saved to: {metadata_save_path}")

print("\nDone!")
