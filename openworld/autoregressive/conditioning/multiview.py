"""Multi-view packing for jointly predicting 3-4 robot camera views.

Two layouts (``ARWMArgs.multiview_layout``):

* ``"height_stack"`` — glue views along H into one tall latent
  ``[B, F, C, V*h, w]`` (the existing CrtlWorld layout). Works with an
  *unmodified* backbone: cross-view interaction is ordinary spatial attention
  within a frame, temporal causality is across real frames. Simplest and the
  default; no view-id embedding needed (spatial position already separates views).

* ``"sequence_pack"`` — concat views along the frame/sequence axis, time-major
  ``[B, F*V, C, h, w]`` ordered ``(t0v0, t0v1, ..., t1v0, ...)`` with a learned
  per-view embedding added to the latent (the OmniDreams layout). A time block of
  ``frames_per_block`` real steps then spans ``frames_per_block * V`` packed
  frames, so set the backbone's effective block size to
  ``effective_frames_per_block``. Views at the same time share a block id, so
  attention is bidirectional across views and causal across time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViewPacker:
    """Stateless (un)packer. ``layout`` is ``"height_stack"`` or ``"sequence_pack"``."""

    def __init__(self, layout: str, num_views: int):
        if layout not in ("height_stack", "sequence_pack"):
            raise ValueError(layout)
        self.layout = layout
        self.v = num_views

    def pack(self, views: torch.Tensor) -> torch.Tensor:
        """``[B, V, F, C, h, w] -> packed``."""
        B, V, F, C, h, w = views.shape
        assert V == self.v, f"expected {self.v} views, got {V}"
        if self.layout == "height_stack":
            # stack views down H: [B, F, C, V*h, w]
            return views.permute(0, 2, 3, 1, 4, 5).reshape(B, F, C, V * h, w)
        # sequence_pack: time-major interleave -> [B, F*V, C, h, w]
        return views.permute(0, 2, 1, 3, 4, 5).reshape(B, F * V, C, h, w)

    def unpack(self, packed: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Inverse of :meth:`pack` -> ``[B, V, F, C, h, w]``."""
        B = packed.shape[0]
        V = self.v
        if self.layout == "height_stack":
            _, F, C, Vh, w = packed.shape
            h = Vh // V
            x = packed.reshape(B, F, C, V, h, w)
            return x.permute(0, 3, 1, 2, 4, 5)
        _, FV, C, h, w = packed.shape
        F = FV // V
        x = packed.reshape(B, F, V, C, h, w)
        return x.permute(0, 1, 2, 3, 4, 5).permute(0, 2, 1, 3, 4, 5)

    def effective_frames_per_block(self, frames_per_block: int) -> int:
        return frames_per_block * (self.v if self.layout == "sequence_pack" else 1)

    def view_ids(self, num_frames: int, device) -> torch.Tensor:
        """Per-packed-frame view id (sequence_pack only), ``[F*V]``."""
        if self.layout != "sequence_pack":
            return torch.zeros(num_frames, dtype=torch.long, device=device)
        return torch.arange(self.v, device=device).repeat(num_frames)


class ViewEmbedding(nn.Module):
    """Learned per-view additive bias on the latent channels (sequence_pack).

    Applied before patch-embedding so each view is identifiable when packed into
    the shared sequence. No-op for height_stack (spatial position already
    separates views)."""

    def __init__(self, num_views: int, channels: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(num_views, channels))

    def forward(self, packed: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        # packed [B, Fp, C, h, w]; view_ids [Fp]
        bias = self.emb[view_ids]                      # [Fp, C]
        return packed + bias[None, :, :, None, None]
