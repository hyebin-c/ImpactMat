"""ImpactMat: modal impact-sound rendering from continuous material parameters."""
from .config import Config, load_config
from .materials import DEFAULT_MATERIALS, Material, resolve_material
from .render import modal_features, render_from_params

__all__ = [
    "Config", "load_config", "DEFAULT_MATERIALS", "Material", "resolve_material",
    "modal_features", "render_from_params",
]
