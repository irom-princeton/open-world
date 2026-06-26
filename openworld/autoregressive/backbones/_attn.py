"""Block-causal self-attention processors for the diffusers Wan / Cosmos DiTs.

These live with the backbones (not in ``causal/``) because they mirror the exact
q/k/v/qk-norm/RoPE math of the stock ``WanAttnProcessor2_0`` /
``CosmosAttnProcessor2_0`` — i.e. they are backbone-specific. The *generic* part
(mode dispatch + KV-cache) is shared via :func:`causal_sdpa`.

Each block gets its own processor instance carrying a fixed ``layer_idx`` (its
cache slot), assigned in registration order at attach time — so correctness does
not depend on the order diffusers later invokes the blocks. Cross-attention
(``attn2``) keeps the stock processor: its K/V come from the constant
action+text condition and there is nothing causal to enforce.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..causal.attention import causal_sdpa
from ..causal.context import CausalContext


class BlockCausalWanAttnProcessor:
    """Drop-in replacement for ``WanAttnProcessor2_0`` on self-attention."""

    def __init__(self, context: CausalContext, layer_idx: int):
        self.ctx = context
        self.layer_idx = layer_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None,
                 kv_mask=None, kv_wpos=None, kv_commit=False):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs):
                dtype = torch.float32 if x.device.type == "mps" else torch.float64
                x_rot = torch.view_as_complex(x.to(dtype).unflatten(3, (-1, 2)))
                return torch.view_as_real(x_rot * freqs).flatten(3, 4).type_as(x)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        if kv_mask is not None:
            # Static (CUDA-graph friendly) cache: the K/V live as REGISTERED BUFFERS
            # on the ``attn.kv`` submodule (module state -> Dynamo tracks the in-place
            # mutation, cudagraph treats them as persistent). RoPE is already baked
            # into ``key`` above. Write current -> scratch (+ ring on commit), then
            # attend the whole fixed-shape buffer with the shared validity mask. Only
            # the mask / ring write-pos / commit flag are threaded as inputs; the K/V
            # are never copied per replay (that was the failed explicit-input path).
            k_all, v_all = attn.kv.extend(key, value, kv_commit, kv_wpos)
            out = F.scaled_dot_product_attention(query, k_all, v_all, attn_mask=kv_mask)
        else:
            out = causal_sdpa(self.ctx, query, key, value, self.layer_idx)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class AlignedCrossAttnProcessor:
    """Wan *cross*-attention (``attn2``) with optional per-frame alignment.

    Mirrors the stock ``WanAttnProcessor2_0`` cross-attn math (q from the latent,
    k/v from the action condition, qk-norm, no RoPE) but applies
    ``ctx.cross_mask`` — the ``[S_q, L_kv]`` per-frame mask from
    :func:`frame_aligned_cross_mask` — so latent frame *f* attends only to its own
    action token. ``cross_mask is None`` -> ordinary global cross-attention,
    numerically identical to the stock processor (used by the non-aligned modes
    and the cached-rollout slices that pre-restrict the K/V instead)."""

    def __init__(self, context: CausalContext):
        self.ctx = context

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        mask = self.ctx.cross_mask
        if mask is not None:
            mask = mask.to(query.device)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class BlockCausalCosmosAttnProcessor:
    """Block-causal / KV-cached replacement for ``CosmosAttnProcessor2_0``."""

    def __init__(self, context: CausalContext, layer_idx: int):
        self.ctx = context
        self.layer_idx = layer_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
        # GQA expand (mirrors the stock processor)
        qd, kd, vd = query.size(3), key.size(3), value.size(3)
        key = key.repeat_interleave(qd // kd, dim=3)
        value = value.repeat_interleave(qd // vd, dim=3)

        out = causal_sdpa(self.ctx, query, key, value, self.layer_idx)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def _attach(transformer, context: CausalContext, block_cls_name: str, proc_cls) -> CausalContext:
    """Assign a fixed-``layer_idx`` block-causal processor to each self-attention
    block (in registration order). Returns the same context with
    ``num_self_layers`` set."""
    n = 0
    for module in transformer.modules():
        if module.__class__.__name__ == block_cls_name:
            module.attn1.processor = proc_cls(context, n)
            n += 1
    if n == 0:
        raise RuntimeError(f"no {block_cls_name} found in transformer")
    context.num_self_layers = n
    return context


def attach_block_causal(transformer, context: CausalContext) -> CausalContext:
    """Patch every Wan self-attention layer (``attn1``) in place."""
    return _attach(transformer, context, "WanTransformerBlock", BlockCausalWanAttnProcessor)


def attach_cross_align(transformer, context: CausalContext) -> None:
    """Patch every Wan *cross*-attention layer (``attn2``) with the per-frame
    :class:`AlignedCrossAttnProcessor` (reads ``context.cross_mask``). Used only
    by ``action_cond_mode="cross_attn_aligned"``."""
    n = 0
    for module in transformer.modules():
        if module.__class__.__name__ == "WanTransformerBlock":
            module.attn2.processor = AlignedCrossAttnProcessor(context)
            n += 1
    if n == 0:
        raise RuntimeError("no WanTransformerBlock found in transformer")


def attach_block_causal_cosmos(transformer, context: CausalContext) -> CausalContext:
    """Patch every Cosmos self-attention layer (``attn1``) in place."""
    return _attach(transformer, context, "CosmosTransformerBlock", BlockCausalCosmosAttnProcessor)
