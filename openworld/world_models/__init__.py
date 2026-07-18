from openworld.world_models.base_world_model import WorldModel
from openworld.world_models.registry import build_world_model, WORLD_MODEL_REGISTRY

__all__ = [
    "WorldModel",
    "build_world_model",
    "WORLD_MODEL_REGISTRY",
    "VidWMWorldModel",
    "VidWMConfig",
]


def __getattr__(name: str):
    # vidwm pulls in the SVD/diffusers-0.34 stack; import lazily so that
    # `import openworld.world_models` works in venvs without it (e.g. the weaver
    # eval venv on diffusers 0.35 / torch 2.7).
    if name in ("VidWMWorldModel", "VidWMConfig"):
        from openworld.world_models import vidwm_world_model

        return getattr(vidwm_world_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
