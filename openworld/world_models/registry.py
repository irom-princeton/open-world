"""World-model registry used by the policy-eval / RL stack (``WorldModelEnv``).

Note: this registry only exposes the **SVD bidirectional** world model (``vidwm``,
which wraps the top-level ``vidwm/`` package) plus a ``dummy`` stub. The
autoregressive (Wan/Cosmos) world model lives in ``openworld.autoregressive`` and
is **not registered here yet** — it has its own training/replay entrypoints and is
not currently driveable from ``run_evaluation.py`` / ``run_rl_finetune.py``. See
``docs/MODELS.md`` for the full model × function matrix.
"""

from typing import Any, Dict

from openworld.world_models.base_world_model import WorldModel
from openworld.world_models.dummy_world_model import DummyWorldModel
from openworld.world_models.vidwm_world_model import VidWMWorldModel

WORLD_MODEL_REGISTRY: Dict[str, type] = {
    "dummy": DummyWorldModel,
    "vidwm": VidWMWorldModel,  # SVD bidirectional world model
}


def build_world_model(name: str, **kwargs: Any) -> WorldModel:
    """Instantiate a world model by registry name."""
    if name not in WORLD_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown world model '{name}'. Available: {list(WORLD_MODEL_REGISTRY)}"
        )
    return WORLD_MODEL_REGISTRY[name](**kwargs)
