# tests/test_basic_models.py

import pytest
from nest import NEST

def test_describe_models():
    nest = NEST("./")  # This does not lead anywhere - but we do not need data anyways
    model_ids = nest.list_models()
    assert len(model_ids) > 0, "No models found."

    for model_id in model_ids:
        info = nest.describe(model_id)
        assert info is not None, f"Describe failed for {model_id}"
