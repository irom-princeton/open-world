"""Convert a diffusers Wan transformer to block-causal + KV-cached attention.

We swap each ``WanTransformerBlock.attn1`` (self-attention) processor for
:class:`BlockCausalWanAttnProcessor`, which mirrors the math of diffusers 0.34's
``WanAttnProcessor2_0`` exactly (q/k/v projections, qk-norm, complex RoPE) and
adds two modes selected by a shared, mutable :class:`CausalContext`:

* ``"train"`` — apply a dense block-causal mask over the full clip. RoPE is
  computed over the full sequence by the transformer, so this path is exact.
* ``"cache"`` — append this layer's keys/values to a :class:`KVCache` and attend
  the (current-block) queries against the whole cache with no mask. The cache is
  already causal, so this equals the masked full forward restricted to the
  retained blocks.

Cross-attention (``attn2``) is left as the stock processor: its keys/values come
from the constant action+text condition, so recomputing them per block is cheap
and there is nothing causal to enforce.

RoPE + KV-cache note: on the ``"cache"`` path the current block's queries/keys
must use RoPE at their *absolute* frame positions, not 0-based. The backbone is
responsible for handing this processor ``rotary_emb`` already sliced to the
block's absolute positions (``backbones/wan.py`` precomputes full-horizon freqs
and slices them). The DummyDiT in the tests exercises the same cache logic with
RoPE disabled, isolating the mask/cache equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kv_cache import KVCache


@dataclass
class CausalContext:
    """Shared state read by every patched attention layer during one forward.

    The backbone sets ``mode`` (+ ``dense_mask`` or ``kv_cache``) before calling
    the diffusers transformer, then the processors consult it. ``_layer`` is an
    internal counter the processors use to claim their cache slot in call order.
    """

    mode: str = "off"                       # "off" | "train" | "cache"
    dense_mask: torch.Tensor | None = None  # [S, S] bool, train mode
    kv_cache: KVCache | None = None         # cache mode
    window: int | None = None
    commit: bool = True                     # cache mode: persist this block's K/V?
    _layer: int = 0

    def begin(self) -> None:
        self._layer = 0

    def next_layer(self) -> int:
        i = self._layer
        self._layer += 1
        return i


class BlockCausalWanAttnProcessor:
    """Drop-in replacement for ``WanAttnProcessor2_0`` on self-attention."""

    def __init__(self, context: CausalContext):
        self.ctx = context

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # self-attention only; cross-attn keeps the stock processor.
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

        ctx = self.ctx
        if ctx.mode == "cache":
            layer = ctx.next_layer()
            key, value = ctx.kv_cache.extend_self(layer, key, value, commit=ctx.commit)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None)
        elif ctx.mode == "train":
            mask = ctx.dense_mask
            if mask is not None:
                mask = mask.to(query.device)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        else:  # "off" -> stock full attention (bidirectional teacher behaviour)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class BlockCausalCosmosAttnProcessor:
    """Block-causal / KV-cached replacement for ``CosmosAttnProcessor2_0``."""

    def __init__(self, context: CausalContext):
        self.ctx = context

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
        # GQA expand (mirrors stock processor)
        qd, kd, vd = query.size(3), key.size(3), value.size(3)
        key = key.repeat_interleave(qd // kd, dim=3)
        value = value.repeat_interleave(qd // vd, dim=3)

        ctx = self.ctx
        if ctx.mode == "cache":
            layer = ctx.next_layer()
            key, value = ctx.kv_cache.extend_self(layer, key, value, commit=ctx.commit)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None)
        elif ctx.mode == "train":
            mask = ctx.dense_mask.to(query.device) if ctx.dense_mask is not None else None
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def _attach(transformer, context, block_cls_name, proc):
    n = 0
    for module in transformer.modules():
        if module.__class__.__name__ == block_cls_name:
            module.attn1.processor = proc
            n += 1
    if n == 0:
        raise RuntimeError(f"no {block_cls_name} found in transformer")
    context._num_self_layers = n
    return context


def attach_block_causal(transformer, context: CausalContext) -> CausalContext:
    """Patch every Wan self-attention layer (``attn1``) in place. Idempotent."""
    return _attach(transformer, context, "WanTransformerBlock", BlockCausalWanAttnProcessor(context))


def attach_block_causal_cosmos(transformer, context: CausalContext) -> CausalContext:
    """Patch every Cosmos self-attention layer (``attn1``) in place."""
    return _attach(transformer, context, "CosmosTransformerBlock", BlockCausalCosmosAttnProcessor(context))
