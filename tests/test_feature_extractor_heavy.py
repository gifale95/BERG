"""Run the real torchvision backbone (not from S3) and check it builds and
produces the expected feature shapes. Marked `heavy` because it downloads the
backbone on first run."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.heavy

eeg = pytest.importorskip("berg.models.eeg.things_eeg2_vit_b_32")
EEGEncodingModel = eeg.EEGEncodingModel


@pytest.fixture(scope="module")
def loaded_extractor():
    model = EEGEncodingModel(subject=EEGEncodingModel.VALID_SUBJECTS[0],
                             device="cpu", berg_dir="/nonexistent")
    extractor = model._load_feature_extractor("cpu")
    import torchvision
    transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
    return extractor, transform


def test_backbone_extracts_12_layers(loaded_extractor, dummy_images):
    extractor, transform = loaded_extractor
    x = transform(torch.from_numpy(dummy_images))
    with torch.no_grad():
        feats = extractor(x)
    # The model selects 12 transformer layers.
    assert len(feats) == 12
    for tensor in feats.values():
        assert tensor.shape[0] == len(dummy_images)


def test_flattened_features_are_finite(loaded_extractor, dummy_images):
    extractor, transform = loaded_extractor
    x = transform(torch.from_numpy(dummy_images))
    with torch.no_grad():
        feats = extractor(x)
    flat = torch.hstack([torch.flatten(l, start_dim=1) for l in feats.values()])
    assert flat.shape[0] == len(dummy_images)
    assert torch.isfinite(flat).all()
