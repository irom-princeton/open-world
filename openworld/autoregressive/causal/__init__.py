"""Block-causal attention + KV-cache: the autoregressive memory mechanism.

A DiT processes a video latent as a flat sequence of ``num_frames *
tokens_per_frame`` tokens. To make generation autoregressive we group the
latent frames into *blocks* of ``frames_per_block`` frames and constrain
attention so a token may attend to every token in its own block and all earlier
blocks, but nothing in the future. Within a block (and, for multi-view, across
all camera views at the same time) attention stays bidirectional. This is the
"block-causal" mask used by OmniDreams (``num_frame_per_block = 2``) and the
Self-Forcing line of work.

At inference we never recompute the past: each new block's keys/values are
appended to a per-layer :class:`KVCache` and the new queries attend to the whole
cache. The cache *is* the model's persistent latent memory across the rollout —
the thing the SVD chunked sampler lacks.

The pieces here are backbone-agnostic and operate on plain ``[B, heads, S, Dh]``
tensors, so they are exercised by a tiny ``DummyDiT`` in the unit tests without
downloading any weights. ``convert.py`` wires them into a real diffusers Wan
transformer.
"""

from .mask import (
    block_ids_for_video,
    dense_block_causal_mask,
    make_flex_block_causal_mask,
)
from .kv_cache import KVCache, LayerKVCache
from .attention import block_causal_attention

__all__ = [
    "block_ids_for_video",
    "dense_block_causal_mask",
    "make_flex_block_causal_mask",
    "KVCache",
    "LayerKVCache",
    "block_causal_attention",
]
