from typing import Any

from openworld.policies.base_policy import Policy
from openworld.utils.optional_dependencies import (
    BackendSpec,
    load_backend_class,
    require_modules,
)

POLICY_REGISTRY: dict[str, BackendSpec] = {
    "openpi": BackendSpec(
        module_path="openworld.policies.openpi_policy",
        class_name="OpenPIPolicy",
        extra_name="policy-openpi",
        required_modules=("jax", "flax", "websockets"),
    ),
    "openpi_libero": BackendSpec(
        module_path="openworld.policies.openpi_libero_policy",
        class_name="OpenPILiberoPolicy",
        extra_name="policy-openpi",
        required_modules=("jax", "flax", "websockets"),
    ),
    "dp": BackendSpec(
        module_path="openworld.policies.dp_policy",
        class_name="DPPolicy",
        extra_name="policy-dp",
        required_modules=("av", "gym", "websockets"),
    ),
}


def build_policy(name: str, **kwargs: Any) -> Policy:
    """Instantiate a policy by registry name."""
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY)}"
        )

    spec = POLICY_REGISTRY[name]
    require_modules(
        backend_name=name,
        backend_kind="policy",
        required_modules=spec.required_modules,
        extra_name=spec.extra_name,
    )
    policy_cls = load_backend_class(spec)
    return policy_cls(**kwargs)
