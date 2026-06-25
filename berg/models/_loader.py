"""Defensive import of model modules.

Models register themselves when imported, so the modality __init__ files import
all of them. Some models need optional packages (MOSAIC, fnn, BrainScore, ...);
if one is missing, the import would otherwise crash all of `import berg`.
safe_import skips those models with a warning, but still re-raises errors coming
from berg's own code so real bugs aren't hidden.
"""

import importlib
import warnings

__all__ = ["safe_import"]


def safe_import(module_path: str) -> bool:
    """Import a model module. Returns True if it loaded, False if it was skipped
    because an optional dependency is missing. Re-raises on internal berg.*
    import errors."""
    try:
        importlib.import_module(module_path)
        return True
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        # Missing berg.* import = a real bug, not an optional dependency.
        if missing == "berg" or missing.startswith("berg."):
            raise
        warnings.warn(
            f"BERG: skipped optional model '{module_path}' "
            f"(missing dependency: {missing or exc}). "
            f"Install the model's optional dependencies to enable it.",
            stacklevel=2,
        )
        return False
