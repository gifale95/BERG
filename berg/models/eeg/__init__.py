# Import models to register them. Guarded via safe_import (see _loader.py).
from berg.models._loader import safe_import

for _model in (
    "berg.models.eeg.things_eeg2_alexnet",
    "berg.models.eeg.things_eeg2_alexnet_untrained",
    "berg.models.eeg.things_eeg2_vit_b_32",
):
    safe_import(_model)
