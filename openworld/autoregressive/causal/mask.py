"""Block-causal attention masks over a flattened video-token sequence.

Token ordering convention (matches diffusers Wan/Cosmos patch-embedding, which
flattens ``(F, H, W) -> F*H*W`` with frame-major order): token ``t`` belongs to
latent frame ``t // tokens_per_frame``, and frame ``f`` belongs to block
``f // frames_per_block``. A query in block ``i`` may attend to a key in block
``j`` iff ``j <= i`` (causal), and additionally ``i - j < window`` when a finite
sliding window of blocks is requested (local attention / bounded KV-cache).
"""

from __future__ import annotations

import torch


def block_ids_for_video(
    num_frames: int,
    tokens_per_frame: int,
    frames_per_block: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a ``[num_frames * tokens_per_frame]`` LongTensor of block ids.

    All tokens of the same time-block (including every camera view packed into
    that block) share an id, so they attend to one another bidirectionally.
    """
    if num_frames % 1 != 0 or tokens_per_frame <= 0 or frames_per_block <= 0:
        raise ValueError("num_frames, tokens_per_frame, frames_per_block must be positive")
    frame_idx = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    return frame_idx // frames_per_block


def dense_block_causal_mask(
    block_ids_q: torch.Tensor,
    block_ids_kv: torch.Tensor,
    *,
    window: int | None = None,
) -> torch.Tensor:
    """Dense boolean attention mask ``[Sq, Skv]`` (``True`` = attend).

    Used for training (full sequence in one shot) and as the portable fallback
    when ``flex_attention`` is unavailable. ``window`` bounds how many blocks
    into the past a query may look (``None`` -> unbounded).
    """
    q = block_ids_q.view(-1, 1)
    kv = block_ids_kv.view(1, -1)
    allowed = kv <= q
    if window is not None:
        allowed = allowed & (q - kv < window)
    return allowed


def frame_aligned_cross_mask(
    num_q_frames: int,
    tokens_per_frame: int,
    num_kv_frames: int,
    *,
    frame_repeat: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-frame *cross*-attention mask ``[num_q_frames*tpf, num_kv_frames]``.

    Used by ``action_cond_mode="cross_attn_aligned"``: a latent query token in
    frame ``qf`` may attend to action token ``k`` iff ``k`` is that frame's own
    action, i.e. ``qf // frame_repeat == k``. ``frame_repeat`` is the number of
    packed latent frames per real (action) frame: 1 for ``height_stack`` (views
    folded into H, so ``num_q_frames == num_kv_frames``) and ``num_cams`` for the
    time-major ``sequence_pack`` layout (each real frame expands to V packed
    frames ``t0v0, t0v1, ...``). ``True`` = attend.
    """
    if num_q_frames % frame_repeat != 0:
        raise ValueError(f"num_q_frames {num_q_frames} not divisible by frame_repeat {frame_repeat}")
    real_of_q = (torch.arange(num_q_frames, device=device) // frame_repeat)  # [num_q_frames]
    real_of_q = real_of_q.repeat_interleave(tokens_per_frame).view(-1, 1)    # [Sq, 1]
    kv = torch.arange(num_kv_frames, device=device).view(1, -1)              # [1, Lkv]
    return real_of_q == kv


def make_flex_block_causal_mask(
    block_ids: torch.Tensor,
    *,
    window: int | None = None,
    batch: int | None = None,
    heads: int | None = None,
):
    """Build a ``torch.nn.attention.flex_attention.BlockMask`` for square
    self-attention over ``block_ids`` (same ids for Q and KV).

    flex_attention compiles the mask and runs fused-sparse attention, skipping
    fully-masked key blocks — this is the path you want at scale. Falls back to
    ``None`` (caller should use the dense mask) if flex_attention is missing.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except Exception:
        return None

    bids = block_ids
    s = bids.numel()

    def mask_mod(b, h, q_idx, kv_idx):
        causal = bids[kv_idx] <= bids[q_idx]
        if window is not None:
            return causal & (bids[q_idx] - bids[kv_idx] < window)
        return causal

    return create_block_mask(
        mask_mod, B=batch, H=heads, Q_LEN=s, KV_LEN=s, device=bids.device, _compile=True
    )
