"""Stage-1/2 *mid-training*: plain flow-matching fine-tuning of one model.

This is the objective omni-dreams uses for both its L2a "student-init" and L1b
"teacher" stages (post-training/.../joint_causal_cosmos_model.py): sample a clean
clip, add flow-matching noise at a uniform timestep, predict the velocity, and
MSE it against the target ``eps - x0``. The *only* difference between the two
stages is the attention pattern:

* **student-init (L2a)** -- ``causal=True``  -> block-causal attention (chunk2),
  the same masking the few-step student uses at rollout. Output initializes the
  self-forcing generator.
* **teacher (L1b)**      -- ``causal=False`` -> full bidirectional attention.
  Output initializes the self-forcing real-score teacher (and the critic).

Both stages are independent (each starts from the base backbone), so they run as
two parallel jobs; the self-forcing (L0) stage then loads both.

Unlike the DMD trainer there is no rollout, no second model, and no KV-cache --
one forward, one MSE, one optimizer step. AdamW hyperparameters mirror the
omni-dreams causal/teacher configs (lr 3e-5, wd 1e-3, grad-clip 0.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DiffusionTrainer:
    """Plain flow-matching trainer for a single ARWorldModel (backbone + action
    conditioner). ``causal`` selects student-init (True) vs teacher (False)."""

    def __init__(
        self,
        model,                       # ARWorldModel
        scheduler,                   # FlowMatchScheduler
        *,
        frames_per_block: int,
        causal: bool,
        lr: float = 3e-5,
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        max_grad_norm: float = 0.1,
        sigma_lo: float = 0.0,
        sigma_hi: float = 1.0,
    ):
        self.model = model
        self.sched = scheduler
        self.fpb = frames_per_block
        self.causal = causal
        self.max_grad_norm = max_grad_norm
        self.sigma_lo, self.sigma_hi = sigma_lo, sigma_hi
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    def train_step(self, latents: torch.Tensor, cond) -> dict:
        """One flow-matching step on a clean clip ``latents`` [B, F, C, H, W]."""
        x0 = latents
        B = x0.shape[0]
        sigma = self.sched.random_sigma(B, x0.device, lo=self.sigma_lo, hi=self.sigma_hi)
        eps = torch.randn_like(x0)
        x_sigma = self.sched.add_noise(x0, eps, sigma)
        t = self.sched.to_timestep(sigma)
        v_pred = self.model.forward_train(
            x_sigma, t, cond, frames_per_block=self.fpb, causal=self.causal)
        v_target = self.sched.velocity_target(x0, eps)
        loss = F.mse_loss(v_pred.float(), v_target.float())
        self.opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()
        return {"loss": loss.item(), "grad_norm": float(gn)}
