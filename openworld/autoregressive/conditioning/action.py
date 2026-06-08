"""Action (+ optional text) conditioning.

Reuses the existing ``vidwm`` MLP action encoder (7-d EEF+gripper -> 1024) so the
robot-controllability signal is identical to the SVD model, then projects to the
backbone's cross-attention width (Wan: 4096 / UMT5, Cosmos: 1024). The projected
per-frame action tokens are what the DiT cross-attention attends to — the same
injection site Wan/Cosmos use for text, which is why a text-pretrained backbone
is convenient (a ready-made cross-attn pathway) even though we feed actions.

Classifier-free-guidance dropout matches ``CrtlWorld.forward`` (zero the whole
condition with 5% probability).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vidwm.action_encoders.unaligned_action_encoder import ActionEncoderUnaligned


# Cap for the learned temporal positional-embedding table (cross_attn_pe mode).
# Matches Wan's rope_max_seq_len; clips are far shorter than this.
_MAX_ACTION_FRAMES = 1024


class ActionConditioner(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cross_attn_dim: int,
        *,
        hidden_dim: int = 1024,
        text_cond: bool = True,
        cfg_dropout: float = 0.05,
        mode: str = "cross_attn",
    ):
        super().__init__()
        self.encoder = ActionEncoderUnaligned(
            action_dim=action_dim, hidden_dim=hidden_dim, text_cond=text_cond
        )
        self.proj = nn.Linear(hidden_dim, cross_attn_dim)
        self.cfg_dropout = cfg_dropout
        self.cross_attn_dim = cross_attn_dim
        self.mode = mode
        # Learned temporal PE on the per-frame action tokens (Fix 1). Only
        # "cross_attn_pe" reads it; zero-init so it starts as a no-op and the
        # other modes are unaffected. The strictly-aligned/adaln modes need no PE
        # (each frame already maps to exactly one action token).
        self.temporal_pe = (
            nn.Parameter(torch.zeros(_MAX_ACTION_FRAMES, hidden_dim))
            if mode == "cross_attn_pe"
            else None
        )

    def forward(
        self,
        actions: torch.Tensor,          # [B, T, action_dim]
        texts=None,
        tokenizer=None,
        text_encoder=None,
        *,
        frame_level_cond: bool = True,
        cfg_drop: bool | None = None,   # override dropout (False at eval)
    ) -> torch.Tensor:                  # [B, L, cross_attn_dim]
        # Text needs a tokenizer + text encoder; until that pathway is wired the
        # condition is action-only (the cross-attention already attends to the
        # per-frame action tokens). Drop texts rather than crash on a missing
        # tokenizer so the real training loop (which forwards `texts`) is safe.
        if text_encoder is None or tokenizer is None:
            texts = None
        out = self.encoder(
            actions, texts=texts, text_tokenizer=tokenizer, text_encoder=text_encoder,
            frame_level_cond=frame_level_cond, device=actions.device,
        )
        emb = out["action_with_text_embeds"]                # [B, L, hidden]
        if self.temporal_pe is not None:
            L = emb.shape[1]
            if L > self.temporal_pe.shape[0]:
                raise ValueError(f"action length {L} exceeds temporal_pe cap {self.temporal_pe.shape[0]}")
            emb = emb + self.temporal_pe[:L].to(emb.dtype)
        cond = self.proj(emb)
        do_drop = self.training if cfg_drop is None else cfg_drop
        if do_drop and self.cfg_dropout > 0:
            keep = (torch.rand(cond.shape[0], device=cond.device) > self.cfg_dropout)
            cond = cond * keep[:, None, None]
        return cond

    @torch.no_grad()
    def null_cond(self, batch: int, length: int, device, dtype=None) -> torch.Tensor:
        """Zeroed condition for the unconditional branch of CFG."""
        return torch.zeros(batch, length, self.cross_attn_dim, device=device, dtype=dtype)
