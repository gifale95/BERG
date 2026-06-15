# Import models to register them. Each import is guarded with safe_import so a
# model whose optional dependency is missing is skipped (with a warning)
# instead of aborting `import berg`. See berg/models/_loader.py.
from berg.models._loader import safe_import

for _model in (
    "berg.models.fmri.bmd_s3d",
    "berg.models.fmri.nsd_fsaverage_alexnet",
    "berg.models.fmri.nsd_fsaverage_alexnet_untrained",
    "berg.models.fmri.nsd_fsaverage_huze",
    "berg.models.fmri.nsd_fsaverage_vit_b_32",
    "berg.models.fmri.nsd_fwrf",
    "berg.models.fmri.things_fmri_1_vit_b_32",
    "berg.models.fmri.tuckute_2024_gpt2_xl",
    "berg.models.fmri.cneuromod_algo2025_text2fmri",
    "berg.models.fmri.lebel2023_opt_1_3b_ridge",
    "berg.models.fmri.cneuromod_algo2025_vibe",
    "berg.models.fmri.dascoli_2026_tribe_v2",
    "berg.models.fmri.mosaic_CNN8_multihead_subAll_verticesVisual",
    "berg.models.fmri.mosaic_CNN8_multihead_subNSD_verticesAll",
    "berg.models.fmri.brainscore_language_models",
):
    safe_import(_model)
