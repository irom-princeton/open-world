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
    bb = _construct(cfg, name)
    # fp32 master weights + bf16 autocast: the backbone forward runs under this
    # compute dtype while params/grads/optimizer stay in cfg.dtype (None -> no
    # autocast). Dummy/SVD ignore it (CPU / not block-causal).
    bb.autocast_dtype = cfg.autocast_dtype
    return bb


def _construct(cfg, name: str) -> DiTBackbone:
    mode = getattr(cfg, "action_cond_mode", "cross_attn")
    # packed latent frames per real (action) frame: sequence_pack expands each
    # real frame into one packed frame per camera view; height_stack keeps 1.
    frame_repeat = cfg.num_cams if cfg.multiview_layout == "sequence_pack" else 1

    if name == "wan_1_3b":
        from .wan import WanBackbone
        if cfg.random_init_backbone:
            return WanBackbone.random_init(
                cross_attn_dim=cfg.cross_attn_dim, small=True,
                action_mode=mode, action_frame_repeat=frame_repeat)
        return WanBackbone.from_pretrained(
            cfg.resolved_backbone_ckpt, cross_attn_dim=cfg.cross_attn_dim, torch_dtype=cfg.dtype,
            action_mode=mode, action_frame_repeat=frame_repeat,
        )

    # The remaining backbones only implement the baseline cross-attn injection.
    if mode != "cross_attn":
        raise NotImplementedError(
            f"action_cond_mode={mode!r} is only implemented for the Wan backbone; "
            f"backbone {name!r} supports 'cross_attn' only."
        )

    if name == "dummy":
        from .dummy import DummyDiT
        return DummyDiT(in_channels=cfg.in_channels, cross_attn_dim=cfg.cross_attn_dim)

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
