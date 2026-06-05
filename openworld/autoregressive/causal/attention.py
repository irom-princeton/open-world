"""Scaled-dot-product attention helpers for block-causal training and cached
autoregressive inference.

Key invariant that makes the KV-cache exact: during a rollout, the cache only
ever holds keys/values from the current block and earlier ones. So the new
block's queries can attend to the *entire* cache with **no mask** and get
exactly the same result as a full masked forward restricted to those blocks.
Training therefore uses a dense block-causal mask; cached inference uses plain
full attention against the cache. ``tests/test_causal_rollout.py`` asserts the
two agree to numerical tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .mask import dense_block_causal_mask


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0
) -> torch.Tensor:
    """Unmasked SDPA. Used on the cached path (cache is already causal)."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


def block_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_ids_q: torch.Tensor | None = None,
    block_ids_kv: torch.Tensor | None = None,
    window: int | None = None,
    dense_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Block-causal SDPA over ``[B, heads, S, head_dim]`` tensors.

    Pass either a precomputed boolean ``dense_mask`` ``[Sq, Skv]`` (reused
    across layers — cheaper) or the per-token ``block_ids`` to build one.
    """
    if dense_mask is None:
        if block_ids_q is None or block_ids_kv is None:
            raise ValueError("provide dense_mask or both block_ids_q/block_ids_kv")
        dense_mask = dense_block_causal_mask(block_ids_q, block_ids_kv, window=window)
    # SDPA expects an additive or boolean mask broadcastable to [B, H, Sq, Skv].
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=dense_mask.to(q.device), dropout_p=dropout_p
    )
