# Import models to register them. Guarded so a missing optional dependency
# (here: `fnn`) skips the model instead of aborting `import berg`.
from berg.models._loader import safe_import

safe_import("berg.models.calcium_2p.calcium_2p_wang_2025_3dcnn")
