"""The single block-causal attention dispatch.

Every self-attention site — the DummyDiT and the patched Wan/Cosmos processors —
routes through :func:`causal_sdpa`, so there is exactly one place that decides
how a query block attends given the :class:`CausalContext` mode:

* ``"train"`` — full clip with the precomputed dense block-causal mask (exact).
* ``"cache"`` — append this layer's K/V to the cache (or, mid-denoise, attend the
  committed cache + the transient current block) and attend with no mask; the
  cache is already causal.
* ``"off"`` — plain full attention (the bidirectional teacher path).

Backbone-specific q/k/v/RoPE prep stays in each backbone's processor; only this
shared tail is common.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .context import CausalContext


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0
) -> torch.Tensor:
    """Unmasked SDPA (cross-attention, and the cached-path tail)."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


def causal_sdpa(
    ctx: CausalContext,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Apply the block-causal attention for one self-attention layer.

    Tensors are ``[B, heads, S, head_dim]``. ``layer_idx`` selects this layer's
    KV-cache slot in ``"cache"`` mode.
    """
    if ctx.mode == "cache":
        kv = ctx.kv_cache
        if getattr(kv, "static", False):
            # Fixed-shape buffer: write current -> scratch (+ ring on commit) and
            # attend over the whole buffer with the shared validity mask, both set by
            # KVCache.begin_forward. Constant shapes + tensor write-pos -> CUDA-graph
            # capturable. (The real Wan rollout takes the equivalent path inside the
            # block-causal processor, reading ``attn.kv`` directly; this branch serves
            # the DummyDiT / op-level equivalence checks that call causal_sdpa.)
            k_all, v_all = kv.self_attn[layer_idx].extend(key, value, ctx.commit, kv.write_pos)
            return F.scaled_dot_product_attention(query, k_all, v_all, attn_mask=kv.attn_mask)
        key, value = kv.extend_self(layer_idx, key, value, commit=ctx.commit)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=None)
    if ctx.mode == "train":
        mask = ctx.dense_mask.to(query.device) if ctx.dense_mask is not None else None
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    # "off" -> bidirectional full attention
    return F.scaled_dot_product_attention(query, key, value, attn_mask=None)
