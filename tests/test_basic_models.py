# tests/test_basic_models.py

import pytest
from nest import NEST

def test_describe_models():
    nest = NEST()
    model_ids = nest.available_models()
    assert len(model_ids) > 0, "No models found."

    for model_id in model_ids:
        info = nest.describe(model_id)
        assert info is not None, f"Describe failed for {model_id}"
