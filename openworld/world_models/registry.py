"""World-model registry used by the policy-eval / RL stack (``WorldModelEnv``).

Exposes the **SVD bidirectional** model (``vidwm``), a ``dummy`` stub, the
**autoregressive Wan student** (``ar_wan``), and the **WEAVER** model
(``weaver``). Every non-dummy entry is registered *lazily* (see the
``_register_*`` helpers) so that requesting one model never imports another
model's backbone deps — important because these models live in mutually
incompatible dependency sets (e.g. weaver needs diffusers>=0.35 / torch 2.7 and
runs in its own venv, where importing the SVD ``vidwm`` stack would fail). See
``docs/MODELS.md`` for the full model × function matrix.
"""

from typing import Any, Dict

from openworld.world_models.base_world_model import WorldModel
from openworld.world_models.dummy_world_model import DummyWorldModel

WORLD_MODEL_REGISTRY: Dict[str, type] = {
    "dummy": DummyWorldModel,
}


def _register_vidwm() -> None:
    """Lazily register the SVD bidirectional / Ctrl-World adapter."""
    if "vidwm" in WORLD_MODEL_REGISTRY:
        return
    from openworld.world_models.vidwm_world_model import VidWMWorldModel
    WORLD_MODEL_REGISTRY["vidwm"] = VidWMWorldModel


def _register_ar() -> None:
    """Lazily register the autoregressive Wan student adapter."""
    if "ar_wan" in WORLD_MODEL_REGISTRY:
        return
    from openworld.world_models.ar_world_model import ARWanWorldModel
    WORLD_MODEL_REGISTRY["ar_wan"] = ARWanWorldModel


def _register_weaver() -> None:
    """Lazily register the WEAVER adapter (runs in the weaver venv)."""
    if "weaver" in WORLD_MODEL_REGISTRY:
        return
    from openworld.world_models.weaver_world_model import WeaverWorldModel
    WORLD_MODEL_REGISTRY["weaver"] = WeaverWorldModel


_LAZY = {"vidwm": _register_vidwm, "ar_wan": _register_ar, "weaver": _register_weaver}


def build_world_model(name: str, **kwargs: Any) -> WorldModel:
    """Instantiate a world model by registry name."""
    if name in _LAZY:
        _LAZY[name]()
    if name not in WORLD_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown world model '{name}'. Available: {list(WORLD_MODEL_REGISTRY)} "
            f"(+ lazily: {sorted(_LAZY)})"
        )
    return WORLD_MODEL_REGISTRY[name](**kwargs)
