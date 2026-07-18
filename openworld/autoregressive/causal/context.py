"""Per-forward causal state, shared by the patched attention layers.

A backbone sets the mode (+ mask or cache) on one :class:`CausalContext` before
calling its diffusers transformer; every patched self-attention layer reads the
same context and applies the dispatch in :func:`causal_sdpa`. Unlike the earlier
design there is **no call-order counter** — each processor is assigned a fixed
``layer_idx`` at attach time, so the cache slot a layer uses does not depend on
the order diffusers happens to invoke blocks (robust to gradient checkpointing /
block reordering).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_cache import KVCache


@dataclass
class CausalContext:
    mode: str = "off"                       # "off" | "train" | "cache"
    dense_mask: torch.Tensor | None = None  # [S, S] bool — train mode
    kv_cache: KVCache | None = None         # cache mode
    commit: bool = True                     # cache mode: persist this block's K/V?
    num_self_layers: int = 0                # set by attach_block_causal*
    # Per-frame action alignment ("cross_attn_aligned"): a [S_q, L_kv] bool mask
    # applied in *cross*-attention (attn2) so latent frame f attends only to its
    # own action token. ``None`` -> unmasked (global) cross-attention. Read by the
    # AlignedCrossAttnProcessor; set per-forward by the backbone.
    cross_mask: torch.Tensor | None = None
