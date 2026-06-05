"""Backbone registry. ``build_backbone(cfg)`` returns a :class:`DiTBackbone`.

The heavy diffusers/HF imports live inside each adapter's constructor, so
importing this module (e.g. to grab ``DummyDiT`` for tests) never pulls Wan or
Cosmos weights.
"""

from __future__ import annotations

from .base import DiTBackbone


def build_backbone(cfg) -> DiTBackbone:
    """Construct the backbone named by ``cfg.backbone``.

    * ``random_init_backbone=True`` -> build untrained weights (CI / shape tests).
    * otherwise -> ``from_pretrained(cfg.resolved_backbone_ckpt)``.
    """
    name = cfg.backbone
    if name == "dummy":
        from .dummy import DummyDiT
        return DummyDiT(in_channels=cfg.in_channels, cross_attn_dim=cfg.cross_attn_dim)

    if name == "wan_1_3b":
        from .wan import WanBackbone
        if cfg.random_init_backbone:
            return WanBackbone.random_init(cross_attn_dim=cfg.cross_attn_dim, small=True)
        return WanBackbone.from_pretrained(
            cfg.resolved_backbone_ckpt, cross_attn_dim=cfg.cross_attn_dim, torch_dtype=cfg.dtype
        )

    if name == "cosmos_predict2_2b":
        from .cosmos_predict2 import CosmosBackbone
        if cfg.random_init_backbone:
            return CosmosBackbone.random_init(cross_attn_dim=cfg.cross_attn_dim, small=True)
        return CosmosBackbone.from_pretrained(
            cfg.resolved_backbone_ckpt, cross_attn_dim=cfg.cross_attn_dim, torch_dtype=cfg.dtype
        )

    if name == "svd":
        from .svd import SVDBackbone
        return SVDBackbone()

    raise ValueError(f"unknown backbone {name!r}")


__all__ = ["DiTBackbone", "build_backbone"]
