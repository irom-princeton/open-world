"""Per-layer KV-cache for autoregressive rollout.

During a rollout we generate one block of latent frames at a time. Each layer's
self-attention appends the new block's keys/values here and attends its new
queries against the whole (past + current) cache. The cache is the model's
persistent memory; an optional ``max_blocks`` sliding window bounds it for
constant-time/constant-memory infinite rollouts (local attention).

Cross-attention keys/values come from the (action+text) condition, which is
constant across the rollout, so they are computed once and cached separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerKVCache:
    """Self-attention K/V cache for a single transformer layer.

    Tensors are ``[B, heads, S_cached, head_dim]``. ``block_lens`` records how
    many tokens each appended block contributed so the sliding window can evict
    whole blocks (never split a block, or cross-view attention within the oldest
    retained block would break).
    """

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    block_lens: list[int] = field(default_factory=list)
    max_blocks: int | None = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        self.block_lens.append(k_new.shape[2])
        self._evict()

    def extend(self, k_new: torch.Tensor, v_new: torch.Tensor, commit: bool):
        """Return the keys/values this block should attend to.

        ``commit=True``: persist the (finalized, clean) block, evict beyond the
        window, and return the resulting windowed cache — so the attention set
        equals the masked forward's. ``commit=False``: transient — return the
        committed (windowed) cache ++ the current noisy block, without persisting
        it (used for the intermediate denoising steps within a block)."""
        if commit:
            self.append(k_new, v_new)            # appends + evicts past the window
            return self.k, self.v                # post-evict windowed cache
        if self.k is None:
            return k_new, v_new
        return torch.cat([self.k, k_new], dim=2), torch.cat([self.v, v_new], dim=2)

    def _evict(self) -> None:
        if self.max_blocks is None:
            return
        while len(self.block_lens) > self.max_blocks:
            drop = self.block_lens.pop(0)
            self.k = self.k[:, :, drop:, :]
            self.v = self.v[:, :, drop:, :]

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def reset(self) -> None:
        self.k = self.v = None
        self.block_lens = []


class KVCache:
    """Holds the self-attn caches for every layer (and, optionally, the
    constant cross-attn K/V) for one rollout.

    Usage::

        cache = KVCache(num_layers, max_blocks=cfg.max_kv_blocks)
        for block in range(num_blocks):
            out = backbone(latent_block, t, cond, kv_cache=cache, start_frame=...)
    """

    def __init__(self, num_layers: int, *, max_blocks: int | None = None):
        self.self_attn = [LayerKVCache(max_blocks=max_blocks) for _ in range(num_layers)]
        self.cross_attn: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self.num_layers = num_layers

    # -- self attention -------------------------------------------------
    def append_self(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append + return the full cache (commit-always; used at plain rollout)."""
        c = self.self_attn[layer]
        c.append(k, v)
        return c.k, c.v

    def extend_self(self, layer: int, k: torch.Tensor, v: torch.Tensor, commit: bool = True):
        """Attend to past ++ current; persist current only if ``commit``."""
        return self.self_attn[layer].extend(k, v, commit)

    def kept_block_count(self, layer: int = 0) -> int:
        return len(self.self_attn[layer].block_lens)

    # -- cross attention (constant condition) ---------------------------
    def set_cross(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.cross_attn[layer] = (k, v)

    def get_cross(self, layer: int):
        return self.cross_attn[layer]

    def has_cross(self, layer: int) -> bool:
        return self.cross_attn[layer] is not None

    def reset(self) -> None:
        for c in self.self_attn:
            c.reset()
        self.cross_attn = [None] * self.num_layers
