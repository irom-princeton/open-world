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
        required_modules=("gym", "websockets"),
    ),
    "molmoact2": BackendSpec(
        module_path="openworld.policies.molmoact2_policy",
        class_name="MolmoAct2Policy",
        extra_name="policy-molmoact2",
        required_modules=("requests", "json_numpy"),
    ),
}


def build_policy(name: str, **kwargs: Any) -> Policy:
    """Instantiate a policy by registry name."""
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY)}"
        )

    spec = POLICY_REGISTRY[name]
    # In websocket mode (``server_url`` set) the OpenPI policy runs out-of-process
    # on a server that holds jax/flax; the local client only needs ``websockets``.
    # This is what lets the policy drive a world model whose venv can't host
    # jax/openpi (e.g. weaver's torch-2.7 venv).
    required_modules = spec.required_modules
    if name in ("openpi", "openpi_libero") and kwargs.get("server_url"):
        required_modules = tuple(m for m in required_modules if m not in ("jax", "flax"))
    require_modules(
        backend_name=name,
        backend_kind="policy",
        required_modules=required_modules,
        extra_name=spec.extra_name,
    )
    policy_cls = load_backend_class(spec)
    return policy_cls(**kwargs)
