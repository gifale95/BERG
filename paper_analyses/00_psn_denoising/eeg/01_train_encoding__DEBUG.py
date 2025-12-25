import numpy as np
import psn
from psn import PSN

# Generate random data of shape (Units, Conditions, Repeats)
n_units = 100
n_conditions_train = 1000
n_conditions_test = 200
n_repeats_train = 4
n_repeats_test = 80
eeg_train = np.random.randn(
    n_units, n_conditions_train, n_repeats_train).astype(np.float32)
eeg_test = np.random.randn(
    n_units, n_conditions_test, n_repeats_test).astype(np.float32)

# Initialize PSN
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
    denoisingtype=1,
    verbose=True,
    wantfig=False,
    gsn_kwargs=None
)

# Fit PSN on the EEG train responses
denoiser.fit(eeg_train)

# Apply the fited PSN to denoise the EEG train and test responses
eeg_train_denoised = denoiser.transform(eeg_train)
eeg_test_denoised = denoiser.transform(eeg_test)

# !!! The numerical values of each unit are constant across all image
# conditions and repeats of both the "eeg_train_denoised" and
# "eeg_test_denoised" variables. Why is this the case? Am I doing something
# wrong?