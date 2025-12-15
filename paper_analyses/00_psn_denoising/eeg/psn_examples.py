"""PSN examples.

PSN GitHub: https://github.com/jacob-prince/PSN

"""

import numpy as np
from gsn.perform_gsn import perform_gsn
import psn
from psn import simulate
from psn import PSN


# =============================================================================
# Simulate data
# =============================================================================
data, _, ground_truth = simulate.generate_data(
    nvox=25,
    ncond=50, 
    ntrial=3,
    noise_multiplier=3,
    align_alpha=0.5,
    align_k=10,
    signal_decay=2,
    noise_decay=1.25,
    want_fig=True,
    random_seed=42
)


# =============================================================================
# PSN
# =============================================================================
# denoisingtype : int, default=0
#     Type of denoising to perform:
#     - 0: Trial-averaged denoising (returns nunits x nconds)
#     - 1: Single-trial denoising (returns nunits x nconds x ntrials)

# Estimate GSN
denoiser = PSN(
    basis='signal',
    cv='unit',
    scoring='mse',
    mag_threshold=0.95,
    unit_groups=None,
    truncate=0,
    ranking=None,
    cv_thresholds=None,
    cv_mode=None,
    denoisingtype=0,
    verbose=True,
    wantfig=False,
    gsn_kwargs=None
)

denoiser.fit(data)

denoised_data = denoiser.transform(data)


# =============================================================================
# PSN (from custom GSN basis)
# =============================================================================
def _compute_symmetric_eigen(matrix):
    """Compute eigendecomposition of a matrix, enforcing symmetry and sorting by magnitude."""
    # Force symmetry for numerical stability
    matrix_sym = (matrix + matrix.T) / 2

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(matrix_sym)

    # Sort by absolute value of eigenvalues (descending)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Standardize eigenvector signs
    evecs = _standardize_eigenvector_signs(evecs)

    magnitudes = np.abs(evals)

    return evecs, magnitudes, matrix_sym

def _standardize_eigenvector_signs(evecs):
    """Standardize eigenvector signs by making the mean of each eigenvector positive."""
    standardized_evecs = evecs.copy()

    for i in range(evecs.shape[1]):
        if np.mean(evecs[:, i]) < 0:
            standardized_evecs[:, i] = -evecs[:, i]

    return standardized_evecs

# Get the signal covariance through GSN
results = perform_gsn(data, {'wantshrinkage': True})
signal = results['cSb']

# Create the custom basis
custom_basis = _compute_symmetric_eigen(signal)[0]

# PSN with custom basis
denoiser_custom = PSN(
    basis=custom_basis,
    cv='unit',
    scoring='mse',
    mag_threshold=0.95,
    unit_groups=None,
    truncate=0,
    ranking=None,
    cv_thresholds=None,
    cv_mode=None,
    denoisingtype=0,
    verbose=True,
    wantfig=False,
    gsn_kwargs=None
)
denoiser_custom.fit(data)
denoised_data_custom = denoiser_custom.transform(data)

# Difference between default and custom PSN
diff = (denoised_data - denoised_data_custom).flatten()
print(min(diff))
print(max(diff))
print(np.mean(diff**2))
print(np.mean(abs(diff)))