from typing import Any, Dict

from openworld.world_models.base_world_model import WorldModel
from openworld.world_models.ctrl_world_world_model import CtrlWorldWorldModel
from openworld.world_models.dummy_world_model import DummyWorldModel
from openworld.world_models.vidwm_world_model import VidWMWorldModel

WORLD_MODEL_REGISTRY: Dict[str, type] = {
    "dummy": DummyWorldModel,
    "vidwm": VidWMWorldModel,
    "ctrlworld": CtrlWorldWorldModel,
}


def build_world_model(name: str, **kwargs: Any) -> WorldModel:
    """Instantiate a world model by registry name."""
    if name not in WORLD_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown world model '{name}'. Available: {list(WORLD_MODEL_REGISTRY)}"
        )
    return WORLD_MODEL_REGISTRY[name](**kwargs)
