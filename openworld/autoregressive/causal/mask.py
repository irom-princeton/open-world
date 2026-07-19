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


def dense_teacher_forcing_mask(
    block_ids: torch.Tensor,
    *,
    window: int | None = None,
) -> torch.Tensor:
    """Dense boolean mask ``[2S, 2S]`` for **teacher-forcing / clean-context** CD.

    The token sequence is the *doubled* clip ``[clean (S) || noisy (S)]`` (clean
    first), where both halves carry the same ``block_ids`` (frame ``f``'s clean
    token and its noisy token share block id). Mirrors Causal-Forcing's
    ``_prepare_teacher_forcing_mask``:

    * a **clean** query attends the clean keys **block-causally** (``kv`` in the
      clean half with ``block(kv) <= block(q)``) -- the clean context is a normal
      block-causal video and never sees the noisy half;
    * a **noisy** query in block ``i`` attends (a) the clean keys of the *previous*
      blocks ``[0, i)`` (its clean context) and (b) its **own** noisy block's tokens
      -- and NOTHING else (no other noisy blocks, no clean block ``i``);
    * every token attends itself (the diagonal), so a padded / degenerate row is
      never fully masked.

    ``window`` bounds how far back the clean context may reach (``block(q) - block(kv)
    < window``), matching the local-attention option of the plain block-causal mask.
    ``True`` = attend.
    """
    S = block_ids.numel()
    bq = block_ids.view(-1, 1)              # [S, 1]
    bkv = block_ids.view(1, -1)             # [1, S]

    causal = bkv <= bq                      # clean block-causal (clean query x clean key)
    same_block = bq == bkv                  # own block (noisy query x noisy key)
    # clean-context for a noisy query in block i is the STRICTLY-previous blocks
    # [0, i) (its own clean block i is excluded -- frame i is the frame being
    # denoised, so its clean latent is not given as context).
    prev_ctx = bkv < bq
    if window is not None:
        causal = causal & (bq - bkv < window)
        prev_ctx = prev_ctx & (bq - bkv < window)

    m = block_ids.new_zeros((2 * S, 2 * S), dtype=torch.bool)
    m[:S, :S] = causal                       # clean query x clean key: block-causal
    m[S:, :S] = prev_ctx                      # noisy query x clean key: previous clean context
    m[S:, S:] = same_block                    # noisy query x noisy key: own block only
    # clean query x noisy key stays False (clean context ignores the noisy half).
    m.fill_diagonal_(True)                    # self-attention (covers padded/edge rows)
    return m


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
