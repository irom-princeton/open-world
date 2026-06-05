"""SVD backbone placeholder.

The existing ``CrtlWorld`` model is a Stable-Video-Diffusion *UNet* with temporal
*convolutions* and a fixed frame budget. Block-causal attention + KV-cache (and
the Self-Forcing recipe built around them) assume a *DiT* with clean temporal
attention: making SVD causal means converting temporal convs to causal convs and
bolting a cache onto a UNet, fighting the architecture and reusing none of the
existing self-forcing code. This is exactly why the recommendation is to base the
autoregressive model on Wan-1.3B / Cosmos-Predict2 rather than SVD.

This module is kept as a registry entry (and a home for a future causal-UNet
experiment) but is not implemented. Use ``backbone="wan_1_3b"`` or
``"cosmos_predict2_2b"``. The legacy SVD model still lives, untouched, in
``openworld.world_models.ctrl_world`` for the bidirectional baseline.
"""

from __future__ import annotations

from .base import DiTBackbone


class SVDBackbone(DiTBackbone):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SVD is a UNet+temporal-conv backbone and is not a block-causal DiT "
            "substrate; use backbone='wan_1_3b' or 'cosmos_predict2_2b'. The "
            "bidirectional SVD model remains available as "
            "openworld.world_models.ctrl_world.CrtlWorld. See docs/AUTOREGRESSIVE.md."
        )
