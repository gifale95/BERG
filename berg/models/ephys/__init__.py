# Import models to register them. Guarded via safe_import (see _loader.py).
from berg.models._loader import safe_import

safe_import("berg.models.ephys.brainscore_vision_models")
