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


class ActionConditioner(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cross_attn_dim: int,
        *,
        hidden_dim: int = 1024,
        text_cond: bool = True,
        cfg_dropout: float = 0.05,
    ):
        super().__init__()
        self.encoder = ActionEncoderUnaligned(
            action_dim=action_dim, hidden_dim=hidden_dim, text_cond=text_cond
        )
        self.proj = nn.Linear(hidden_dim, cross_attn_dim)
        self.cfg_dropout = cfg_dropout
        self.cross_attn_dim = cross_attn_dim

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
        out = self.encoder(
            actions, texts=texts, text_tokenizer=tokenizer, text_encoder=text_encoder,
            frame_level_cond=frame_level_cond, device=actions.device,
        )
        cond = self.proj(out["action_with_text_embeds"])
        do_drop = self.training if cfg_drop is None else cfg_drop
        if do_drop and self.cfg_dropout > 0:
            keep = (torch.rand(cond.shape[0], device=cond.device) > self.cfg_dropout)
            cond = cond * keep[:, None, None]
        return cond

    @torch.no_grad()
    def null_cond(self, batch: int, length: int, device, dtype=None) -> torch.Tensor:
        """Zeroed condition for the unconditional branch of CFG."""
        return torch.zeros(batch, length, self.cross_attn_dim, device=device, dtype=dtype)
