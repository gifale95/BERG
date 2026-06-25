"""Run generate_response end to end with stubbed weights.

The real scaler/PCA/regression weights live in S3, but we can check the
pipeline (preprocess -> features -> scale -> PCA -> regress -> reshape) by
plugging in small fake stand-ins of the right shape. No download needed."""

import numpy as np
import pytest
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

eeg = pytest.importorskip("berg.models.eeg.things_eeg2_vit_b_32")
EEGEncodingModel = eeg.EEGEncodingModel

N_CHANNELS = 17
N_TIMES = EEGEncodingModel.TIMEPOINTS_LENGTH  # 140
FEATURE_DIM = 64
N_PCA = 10


class _StubFeatureExtractor:
    """Stand-in for the torchvision extractor: returns one layer of fake
    features per image, which is enough to drive the rest of the pipeline."""

    def __call__(self, batch):
        return {"layer": torch.randn(batch.shape[0], FEATURE_DIM)}


def _fit(n_out):
    rng = np.random.RandomState(0)
    x = rng.randn(50, FEATURE_DIM).astype(np.float32)
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=N_PCA).fit(scaler.transform(x))
    reg = LinearRegression().fit(pca.transform(scaler.transform(x)),
                                 rng.randn(50, n_out))
    return scaler, pca, reg


@pytest.fixture
def stub_model():
    model = EEGEncodingModel(subject=EEGEncodingModel.VALID_SUBJECTS[0],
                             device="cpu", berg_dir="/nonexistent")
    # Attributes that load_model() would normally populate from S3:
    model.ch_names = [f"E{i}" for i in range(N_CHANNELS)]
    model.times = np.linspace(0, 1, N_TIMES)
    model.channel_indices = range(N_CHANNELS)
    model.selected_timepoints = range(N_TIMES)
    model.transform = lambda t: t.float()           # skip ImageNet normalization
    model.feature_extractor = _StubFeatureExtractor()
    scaler, pca, reg = _fit(N_CHANNELS * N_TIMES)
    model.scaler = scaler
    model.pca = [pca]                                # one "repetition"
    model.regression_weights = [reg]
    return model


def test_generate_response_shape_and_dtype(stub_model, dummy_images):
    out = stub_model.generate_response(dummy_images, show_progress=False)
    # (images, repetitions, channels, timepoints)
    assert out.shape == (len(dummy_images), 1, N_CHANNELS, N_TIMES)
    assert out.dtype == np.float32


def test_generate_response_rejects_bad_stimulus(stub_model):
    with pytest.raises(Exception):
        stub_model.generate_response(np.zeros((3, 224, 224)), show_progress=False)
