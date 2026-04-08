"""Train a ridge regression encoding model to predict Podcast ECoG high-gamma
activity from GPT-2 XL word embeddings.

The model is trained on ALL available epochs (all words) for a given subject.
This differs from the paper's 2-fold CV evaluation setup — here we maximize
model quality for BERG by using all data. Encoding accuracy is evaluated
separately in 01_test_encoding.py using 2-fold CV matching the paper.

Pipeline:
1. Load pre-extracted GPT-2 XL word embeddings (layer 24, 1600-dim)
2. Load epoched high-gamma ECoG, reshape to (n_words, n_electrodes * n_lags)
3. StandardScaler on X (features) and Y (neural data)
4. Fit RidgeCV (himalaya) with 5-fold inner CV for alpha selection
5. Save scaler params + ridge weights

Note: Both X and Y are standardized before fitting, following the paper's
tutorial. Y scaling ensures all electrode×lag targets contribute equally to
alpha selection. The Y scaler params are saved so predictions can be
inverse-transformed back to original high-gamma units at BERG inference time.

Parameters
----------
subject : str
    Subject identifier (e.g., 'sub-01').
berg_dir : str
    Directory of the BERG framework.

Output Files Created (per subject):
────────────────────────────────────────────────────────────────
weights_{subject}.npy :
    'scaler_X_param':
        scale_, mean_, var_, n_features_in_, n_samples_seen_
    'scaler_Y_param':
        scale_, mean_, var_, n_features_in_, n_samples_seen_
    'ridge_param':
        coef_            : (n_electrodes*n_lags, 1600)
        intercept_       : (n_electrodes*n_lags,)
        best_alphas_     : selected regularization strength
        n_features_in_   : int
    'shape':
        n_electrodes     : int
        n_lags           : int
"""

import argparse
import os
import numpy as np
import h5py
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from himalaya.backend import set_backend
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

print('>>> Train Podcast ECoG Encoding Model <<<')
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
# Helper
# =============================================================================
def to_numpy(x):
    """Convert torch tensor or array to numpy."""
    if hasattr(x, 'numpy'):
        return x.numpy(force=True)
    return np.asarray(x)


# =============================================================================
# Load data
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
                        'train_dataset-zada2025')

# Load metadata
metadata_path = os.path.join(data_dir,
                             f'zada2025_{args.subject}_metadata.npy')
metadata = np.load(metadata_path, allow_pickle=True).item()

n_electrodes = metadata['ecog']['n_electrodes']
n_lags = metadata['ecog']['n_lags']
print(f"\nSubject {args.subject}: {n_electrodes} electrodes, {n_lags} lags")

# Load features (X)
features_path = os.path.join(data_dir,
                             f'zada2025_{args.subject}_features.npy')
X = np.load(features_path)
print(f"Features shape: {X.shape}")

# Load neural data (Y) and reshape to (n_epochs, n_electrodes * n_lags)
neural_path = os.path.join(data_dir,
                           f'zada2025_{args.subject}_neural.h5')
with h5py.File(neural_path, 'r') as f:
    Y = f['neural_data'][:].reshape(f['neural_data'].shape[0], -1)
print(f"Neural data shape: {Y.shape}")

assert X.shape[0] == Y.shape[0], "Feature and neural data row count mismatch"
assert Y.shape[1] == n_electrodes * n_lags, "Neural data column count mismatch"

# Cast to float32
X = X.astype(np.float32)
Y = Y.astype(np.float32)


# =============================================================================
# Standardize features and neural data
# =============================================================================
print("\nStandardizing data...")

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)


# =============================================================================
# Fit RidgeCV on all data
# =============================================================================
# Ridge regression with 5-fold inner CV for alpha selection, matching the
# paper's tutorial setup. Training on ALL data (no outer CV) to maximize
# model quality for BERG.
print("\nFitting RidgeCV on all data...")

alphas = np.logspace(1, 10, 10)
inner_cv = KFold(n_splits=5, shuffle=False)

ridge = RidgeCV(alphas=alphas, cv=inner_cv, fit_intercept=True)
ridge.fit(X_scaled, Y_scaled)

best_alphas = to_numpy(ridge.best_alphas_)
print(f"  Best alpha: {best_alphas}")

# Training R² as sanity check
Y_pred_train = to_numpy(ridge.predict(X_scaled))
ss_res = ((Y_scaled - Y_pred_train) ** 2).sum(axis=0)
ss_tot = ((Y_scaled - Y_scaled.mean(axis=0)) ** 2).sum(axis=0)
r2_train = 1 - ss_res / ss_tot
r2_train_reshaped = r2_train.reshape(n_electrodes, n_lags)
print(f"  Training R² per electrode (mean over lags): "
      f"min={r2_train_reshaped.mean(1).min():.4f}, "
      f"median={np.median(r2_train_reshaped.mean(1)):.4f}, "
      f"max={r2_train_reshaped.mean(1).max():.4f}")


# =============================================================================
# Save the trained encoding model weights
# =============================================================================
weights = {
    'scaler_X_param': {
        'scale_': scaler_X.scale_,
        'mean_': scaler_X.mean_,
        'var_': scaler_X.var_,
        'n_features_in_': scaler_X.n_features_in_,
        'n_samples_seen_': scaler_X.n_samples_seen_,
    },
    'scaler_Y_param': {
        'scale_': scaler_Y.scale_,
        'mean_': scaler_Y.mean_,
        'var_': scaler_Y.var_,
        'n_features_in_': scaler_Y.n_features_in_,
        'n_samples_seen_': scaler_Y.n_samples_seen_,
    },
    'ridge_param': {
        'coef_': to_numpy(ridge.coef_),
        'intercept_': to_numpy(ridge.intercept_),
        'best_alphas_': best_alphas,
        'n_features_in_': ridge.n_features_in_,
    },
    'shape': {
        'n_electrodes': n_electrodes,
        'n_lags': n_lags,
    },
}

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-ecog',
                        'train_dataset-zada2025', 'model-gpt2_xl',
                        'encoding_models_weights')
os.makedirs(save_dir, exist_ok=True)

file_name = f'weights_{args.subject}.npy'
np.save(os.path.join(save_dir, file_name), weights)
print(f"\nWeights saved to: {os.path.join(save_dir, file_name)}")

print("\nDone!")
