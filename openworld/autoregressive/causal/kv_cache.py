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
import torch.nn as nn


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
    # consistency_aug: anchored + strided + windowed eviction (opt-in). When
    # ``recent_blocks`` is None the legacy FIFO-over-``max_blocks`` policy is used
    # (fully backward-compatible). When set, a committed block with commit-index
    # ``g`` (0-based) is retained iff it is one of the first ``anchor_blocks``
    # (the GT-primed anchor), OR within the last ``recent_blocks``, OR a strided
    # "memory" block (``(g-anchor) % memory_stride == 0``, capped to the most-recent
    # ``memory_blocks``). This keeps a bounded window that stays tethered to early
    # context -- WEAVER's memory frames, expressed purely as an eviction policy.
    # RoPE is baked into keys at commit and softmax is permutation-invariant over the
    # key axis, so retaining a non-contiguous subset is exact.
    anchor_blocks: int = 0
    memory_stride: int = 0
    memory_blocks: int = 0
    recent_blocks: int | None = None
    block_ids: list[int] = field(default_factory=list)   # commit index of each retained block
    committed: int = 0                                    # total blocks ever committed

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        self.block_lens.append(k_new.shape[2])
        self.block_ids.append(self.committed)
        self.committed += 1
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
        if self.recent_blocks is not None:
            return self._evict_anchored()
        if self.max_blocks is None:
            return
        while len(self.block_lens) > self.max_blocks:
            drop = self.block_lens.pop(0)
            self.block_ids.pop(0)
            self.k = self.k[:, :, drop:, :]
            self.v = self.v[:, :, drop:, :]

    def _evict_anchored(self) -> None:
        """consistency_aug eviction: keep anchor + strided memory + recent window.

        Blocks can be dropped from the *middle* (a retained memory block with newer
        blocks evicted around it), so this gathers the surviving token spans rather
        than slicing a prefix."""
        T = self.committed
        mem_ids: set[int] = set()
        if self.memory_stride and self.memory_stride > 0:
            cand = [g for g in self.block_ids
                    if g >= self.anchor_blocks and (g - self.anchor_blocks) % self.memory_stride == 0]
            if self.memory_blocks and self.memory_blocks > 0:
                cand = cand[-self.memory_blocks:]
            mem_ids = set(cand)
        keep = [(g < self.anchor_blocks) or (g >= T - self.recent_blocks) or (g in mem_ids)
                for g in self.block_ids]
        if all(keep):
            return
        spans, new_lens, new_ids = [], [], []
        pos = 0
        for kp, L, g in zip(keep, self.block_lens, self.block_ids):
            if kp:
                spans.append(torch.arange(pos, pos + L, device=self.k.device))
                new_lens.append(L)
                new_ids.append(g)
            pos += L
        index = torch.cat(spans)
        self.k = self.k.index_select(2, index)
        self.v = self.v.index_select(2, index)
        self.block_lens, self.block_ids = new_lens, new_ids

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def reset(self) -> None:
        self.k = self.v = None
        self.block_lens = []
        self.block_ids = []
        self.committed = 0


class StaticLayerKVCache(nn.Module):
    """Fixed-shape self-attn K/V cache for one layer, held as **registered buffers
    on an ``nn.Module``** (the gpt-fast pattern).

    :class:`LayerKVCache` grows by ``torch.cat`` and slices on eviction, so its
    length -- and every attention shape -- changes each step, defeating
    ``torch.compile(mode="reduce-overhead")`` (recompiles + no CUDA graph). This
    instead preallocates ONE buffer of ``W + block_tok`` tokens once
    (``W = max_blocks * block_tok``): ``[0:W]`` is the committed ring window and
    ``[W:]`` a scratch slot for the current block. Every forward writes the current
    block into the scratch slot (in place), attends over the SAME whole buffer with
    a validity mask, and -- on ``commit`` -- ALSO writes the block into its ring
    slot for future blocks. No ``torch.cat`` and no python-int-dependent indexing
    inside the forward (the ring slot is a *tensor* ``write_pos`` the caller updates
    eagerly), so the compiled region is shape-invariant across steps.

    The K/V are buffers of an ``nn.Module`` *attached as a submodule onto each
    ``attn1``* (``attn1.kv``); because the submodule is inside the compiled module
    hierarchy, Dynamo/Inductor tracks the in-place mutation and CUDA-graph replays
    treat the buffers as **persistent** memory (mutated in place, NOT copied per
    replay). That is the one thing that makes compile correct across a mutating
    rollout where the failed "K/V as explicit graph inputs" approach was not (the
    cudagraph copies inputs into static slots each replay, breaking cross-step
    persistence). Only the mask + ``write_pos`` (small, changing every step) are
    threaded as inputs.

    Correctness vs the growing cache: RoPE is baked into keys *before* caching and
    softmax is permutation-invariant over the key axis, so the ring's (out-of-order
    once it wraps) slot layout is exact -- only *which* slots are valid matters, and
    that is the mask. Before the ring wraps, filled slots are a contiguous prefix.
    The current block is counted once, via the scratch slot; the just-committed ring
    slot is excluded from the current forward's validity mask.
    """

    def __init__(self, max_blocks: int):
        super().__init__()
        self.max_blocks = max_blocks
        self.block_tok: int | None = None
        # buffers allocated lazily in :meth:`allocate` (shapes need a forward).
        # persistent=False -> never written to a checkpoint (inference-only state).
        self.register_buffer("k", None, persistent=False)   # [B, H, W+block_tok, D]
        self.register_buffer("v", None, persistent=False)

    def allocate(self, B: int, H: int, block_tok: int, D: int, device, dtype) -> None:
        """(Re)allocate the persistent buffer when missing or shape/dtype/device
        changed; otherwise a no-op (so addresses stay stable across rollout resets
        and a captured CUDA graph remains valid). Called EAGERLY from
        :meth:`KVCache.begin_forward` -- never inside a compiled region -- so the
        ``mark_static_address`` (which Dynamo forbids tracing) and the allocation
        stay out of the graph; :meth:`extend` then only reads/writes the buffers."""
        W = block_tok * self.max_blocks
        if (self.k is not None and self.block_tok == block_tok
                and tuple(self.k.shape) == (B, H, W + block_tok, D)
                and self.k.dtype == dtype and self.k.device == device):
            return
        self.block_tok = block_tok
        self.k = torch.zeros(B, H, W + block_tok, D, device=device, dtype=dtype)
        self.v = torch.zeros(B, H, W + block_tok, D, device=device, dtype=dtype)
        # Fixed addresses: lets CUDA-graph capture treat the in-place writes as
        # persistent state rather than per-replay graph inputs.
        for buf in (self.k, self.v):
            try:
                torch._dynamo.mark_static_address(buf)
            except Exception:
                pass

    def extend(self, k_new: torch.Tensor, v_new: torch.Tensor, commit: bool,
               write_pos: torch.Tensor):
        """Write current -> scratch (always) and -> ring slot (on commit), then
        return the whole fixed-shape buffer to attend over. ``write_pos`` is a
        ``[block_tok]`` LongTensor of ring positions (set eagerly by the caller)."""
        bt = self.block_tok
        # Cast to the buffer dtype INDEPENDENTLY for k and v: under bf16 autocast on
        # fp32 master weights the cached key comes out fp32 (qk-norm + RoPE type_as)
        # while the value stays bf16, so a single shared check would leave one
        # mismatched. Upcasting bf16->fp32 is exact, so the buffered values equal what
        # the growing cache stores (and the autocast SDPA rounds both back identically).
        if k_new.dtype != self.k.dtype:
            k_new = k_new.to(self.k.dtype)
        if v_new.dtype != self.v.dtype:
            v_new = v_new.to(self.v.dtype)
        self.k[:, :, -bt:, :] = k_new                # current block -> scratch slot
        self.v[:, :, -bt:, :] = v_new
        if commit:                                   # persist into the ring for future blocks
            self.k[:, :, :-bt, :].index_copy_(2, write_pos, k_new)
            self.v[:, :, :-bt, :].index_copy_(2, write_pos, v_new)
        return self.k, self.v

    def reset(self) -> None:
        # Keep the allocated buffers (stable addresses for CUDA-graph capture);
        # logical validity is governed by the shared mask + ``n_committed``, which
        # ``KVCache.reset`` clears. Stale slots are never attended (the validity
        # prefix only covers slots written this episode).
        pass


class KVCache:
    """Holds the self-attn caches for every layer (and, optionally, the
    constant cross-attn K/V) for one rollout.

    Usage::

        cache = KVCache(num_layers, max_blocks=cfg.max_kv_blocks)
        for block in range(num_blocks):
            out = backbone(latent_block, t, cond, kv_cache=cache, start_frame=...)

    ``static=True`` uses fixed-shape :class:`StaticLayerKVCache` buffers (requires a
    finite ``max_blocks``) so the per-block forward has constant shapes and can be
    captured as a CUDA graph. The per-layer caches are passed in via ``layer_caches``
    -- the registered-buffer submodules the backbone attaches onto each ``attn1`` --
    so their K/V are tracked module state under ``torch.compile``. The backbone calls
    :meth:`begin_forward` once per forward (eagerly) to refresh the shared validity
    mask + ring ``write_pos`` before the (possibly compiled) transformer.
    """

    def __init__(self, num_layers: int, *, max_blocks: int | None = None,
                 static: bool = False, layer_caches: "list | None" = None,
                 anchor_blocks: int = 0, memory_stride: int = 0,
                 memory_blocks: int = 0, recent_blocks: int | None = None):
        self.static = static
        self.num_layers = num_layers
        self.max_blocks = max_blocks
        if static:
            if max_blocks is None:
                raise ValueError("static KVCache requires a finite max_blocks (the attention window)")
            if recent_blocks is not None:
                # The static ring buffer is a fixed FIFO window for CUDA-graph capture;
                # anchored/strided eviction needs the growing cache. Training uses the
                # growing cache, so this only guards a misconfigured static deploy.
                raise ValueError("consistency_aug anchored eviction requires the growing (non-static) KV cache")
            self.self_attn = (layer_caches if layer_caches is not None
                              else [StaticLayerKVCache(max_blocks) for _ in range(num_layers)])
        else:
            self.self_attn = [LayerKVCache(
                max_blocks=max_blocks, anchor_blocks=anchor_blocks, memory_stride=memory_stride,
                memory_blocks=memory_blocks, recent_blocks=recent_blocks) for _ in range(num_layers)]
        self.cross_attn: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        # static-mode shared per-forward state (refreshed eagerly in begin_forward):
        self.attn_mask: torch.Tensor | None = None   # [1,1,1,W+block_tok] bool validity
        self.write_pos: torch.Tensor | None = None   # [block_tok] long, current ring slot
        self.n_committed: int = 0                    # committed blocks (slot = n % max_blocks)

    # -- static-cache per-forward setup ---------------------------------
    def begin_forward(self, *, commit: bool, block_tok: int, num_heads: int, head_dim: int,
                      batch: int, device, dtype) -> None:
        """Allocate buffers (once) + refresh the shared validity mask + ring
        ``write_pos`` for one (static) forward. No-op for a non-static cache.

        Called by the backbone *before* the (compiled) transformer so the python
        bookkeeping (allocation, ``mark_static_address``, how many blocks are
        committed) stays OUT of the compiled region -- the compiled attention only
        reads the preallocated buffers + the mask/``write_pos`` tensors.

        The mask marks, over the constant ``[W + block_tok]`` key axis, the filled
        window prefix (``min(n_committed, max_blocks) * block_tok`` columns) plus the
        always-valid current block (the trailing ``block_tok`` columns). It reflects
        the count *before* this block commits -- the current block is counted once,
        via the trailing scratch slot, not the ring.
        """
        if not self.static:
            return
        for c in self.self_attn:
            c.allocate(batch, num_heads, block_tok, head_dim, device, dtype)
        W = self.max_blocks * block_tok
        n_valid = min(self.n_committed, self.max_blocks) * block_tok   # before this block commits
        if (self.attn_mask is None or self.attn_mask.shape[-1] != W + block_tok
                or self.attn_mask.device != device):
            self.attn_mask = torch.zeros(1, 1, 1, W + block_tok, dtype=torch.bool, device=device)
            self.write_pos = torch.zeros(block_tok, dtype=torch.long, device=device)
        # validity mask: filled ring prefix + the always-valid current scratch slot.
        self.attn_mask.fill_(False)
        self.attn_mask[..., :n_valid] = True
        self.attn_mask[..., W:] = True
        # ring write position for THIS block (a tensor, so the compiled write is
        # invariant to the slot value); advance the committed count on commit.
        slot = self.n_committed % self.max_blocks
        self.write_pos.copy_(torch.arange(slot * block_tok, (slot + 1) * block_tok, device=device))
        if commit:
            # ``extend`` writes the current block into ring slot ``slot`` AND the
            # scratch slot. Exclude that ring slot from this commit forward's mask so
            # the current block is attended exactly once (via scratch). Without this,
            # once the ring has wrapped (``n_valid == W`` marks every slot valid) the
            # commit forward double-counts the current block -- and although its
            # *output* is discarded, it also caches this block's K/V for every layer,
            # so the layers >= 1 keys (which depend on the layer-0 attention output)
            # would be corrupted and every later block reads them. No-op before the
            # ring wraps (the slot is already outside the valid prefix).
            self.attn_mask[..., slot * block_tok:(slot + 1) * block_tok] = False
            self.n_committed += 1

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
        self.n_committed = 0
