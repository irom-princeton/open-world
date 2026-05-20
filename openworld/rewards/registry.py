from typing import Any

from openworld.rewards.base_reward_model import RewardModel
from openworld.utils.optional_dependencies import (
    BackendSpec,
    load_backend_class,
    require_modules,
)

REWARD_MODEL_REGISTRY: dict[str, BackendSpec] = {
    "dummy": BackendSpec(
        module_path="openworld.rewards.dummy_reward_model",
        class_name="DummyRewardModel",
    ),
    "robometer": BackendSpec(
        module_path="openworld.rewards.robometer_reward_model",
        class_name="RobometerRewardModel",
    ),
    "vlm": BackendSpec(
        module_path="openworld.rewards.vlm_reward_model",
        class_name="VLMRewardModel",
        # backend (openai / gemini) is chosen at runtime; each SDK is imported
        # lazily by VLMRewardModel with a clear error if missing.
    ),
    "topreward": BackendSpec(
        module_path="openworld.rewards.topreward_reward_model",
        class_name="TOPRewardModel",
        extra_name="reward-topreward",
        required_modules=("requests", "openai"),
    ),
}


def build_reward_model(name: str, **kwargs: Any) -> RewardModel:
    """Instantiate a reward model by registry name."""
    if name not in REWARD_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown reward model '{name}'. Available: {list(REWARD_MODEL_REGISTRY)}"
        )

    spec = REWARD_MODEL_REGISTRY[name]
    require_modules(
        backend_name=name,
        backend_kind="reward",
        required_modules=spec.required_modules,
        extra_name=spec.extra_name,
    )
    reward_cls = load_backend_class(spec)
    return reward_cls(**kwargs)
