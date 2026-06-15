"""Model construction and parameter errors, using EEG as the example.
Validation happens in __init__, before load_model reads any file, so the error
paths work without weights."""

import pytest

from berg.core.exceptions import InvalidParameterError

eeg = pytest.importorskip("berg.models.eeg.things_eeg2_vit_b_32")
EEGEncodingModel = eeg.EEGEncodingModel

VALID_SUBJECT = EEGEncodingModel.VALID_SUBJECTS[0]
INVALID_SUBJECT = max(s for s in EEGEncodingModel.VALID_SUBJECTS if isinstance(s, int)) + 1000


def test_valid_construction_does_not_touch_disk():
    # berg_dir points nowhere, but load_model is not called, so this must work.
    model = EEGEncodingModel(subject=VALID_SUBJECT, device="cpu", berg_dir="/nonexistent")
    assert model.subject == VALID_SUBJECT
    assert model.device == "cpu"


def test_invalid_subject_raises():
    with pytest.raises(InvalidParameterError):
        EEGEncodingModel(subject=INVALID_SUBJECT, device="cpu", berg_dir="/nonexistent")


def test_invalid_selection_key_raises():
    with pytest.raises(InvalidParameterError):
        EEGEncodingModel(
            subject=VALID_SUBJECT,
            device="cpu",
            selection={"not_a_real_key": []},
            berg_dir="/nonexistent",
        )


def test_invalid_channel_raises():
    with pytest.raises(Exception):
        EEGEncodingModel(
            subject=VALID_SUBJECT,
            device="cpu",
            selection={"channels": ["NOT_A_CHANNEL"]},
            berg_dir="/nonexistent",
        )


def test_get_model_id_classmethod():
    assert EEGEncodingModel.get_model_id() == "eeg-things_eeg_2-vit_b_32"
