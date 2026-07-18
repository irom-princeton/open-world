"""Backbone abstraction shared by Wan / Cosmos / SVD / Dummy.

A backbone is a velocity (flow-matching) predictor over latent video. It exposes
two forward modes so the same weights serve both training and rollout:

* :meth:`forward_train` — full clip with a block-causal mask (exact; used by the
  self-forcing generator and the DMD critic/teacher scores).
* :meth:`forward_cached` — one block at a time, reading/updating a
  :class:`KVCache` (the autoregressive rollout used at inference and to generate
  the student trajectory during self-forcing).

Both take latents shaped ``[B, F, C, H, W]`` (F = latent frames, already
multi-view-packed by ``conditioning/multiview.py``) and a condition tensor
``cond`` shaped ``[B, L, cross_attn_dim]`` (action+text embeddings) plus a
per-frame or scalar ``timestep``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..causal.kv_cache import KVCache


class DiTBackbone(nn.Module, ABC):
    # Subclasses set these in __init__.
    in_channels: int
    cross_attn_dim: int
    num_self_layers: int
    patch_spatial: int
    patch_temporal: int

    # Compute dtype for the transformer forward when parameters are kept in a
    # higher precision (fp32 master weights + bf16 autocast). ``None`` -> run in
    # the parameter dtype (pure fp32 or pure bf16). Set by ``build_backbone``.
    autocast_dtype: "torch.dtype | None" = None

    @abstractmethod
    def forward_train(
        self,
        latents: torch.Tensor,        # [B, F, C, H, W]
        timestep: torch.Tensor,       # [B] or [B, F]
        cond: torch.Tensor,           # [B, L, cross_attn_dim]
        *,
        frames_per_block: int,
        window: int | None = None,
        causal: bool = True,          # False -> full bidirectional attention (teacher)
    ) -> torch.Tensor:                # velocity, same shape as latents
        ...

    @abstractmethod
    def forward_cached(
        self,
        latent_block: torch.Tensor,   # [B, Fb, C, H, W]
        timestep: torch.Tensor,       # [B] or [B, Fb]
        cond: torch.Tensor,           # [B, L, cross_attn_dim]
        *,
        kv_cache: KVCache,
        start_frame: int,             # absolute latent-frame index of this block
        commit: bool = True,          # persist this block's K/V (False mid-denoise)
    ) -> torch.Tensor:                # velocity, same shape as latent_block
        ...

    def make_kv_cache(self, *, max_blocks: int | None = None, static: bool = False) -> KVCache:
        return KVCache(self.num_self_layers, max_blocks=max_blocks, static=static)

    def tokens_per_frame(self, latent_h: int, latent_w: int) -> int:
        return (latent_h // self.patch_spatial) * (latent_w // self.patch_spatial)

    def slice_cond_to_frames(self, cond: torch.Tensor, start_frame: int, num_frames: int) -> torch.Tensor:
        """Slice an action-cond tensor to the action frames covering latent frames
        ``[start_frame, start_frame + num_frames)``.

        Default: no-op -- global-cond backbones (``cross_attn`` / ``cross_attn_pe``)
        attend to the whole cond. Frame-aligned backbones override this so scoring a
        *sub-window* of the rollout keeps the per-frame cross-attention aligned. The
        DMD score path scores the *generated* clip, which starts at the history
        offset (not frame 0); passing the full, unsliced cond would condition
        generated frame ``j`` on action ``j`` instead of action ``start_frame + j``."""
        return cond
