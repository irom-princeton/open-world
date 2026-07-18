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

    def _per_block_sigma(self, B: int, Fr: int, device) -> torch.Tensor:
        """Independent noise level per *block* (diffusion forcing), expanded to a
        per-frame [B, Fr] grid. Frames are grouped exactly as the block-causal mask
        groups them (``frame_idx // fpb``), so block b's frames all share its sigma.
        ``Fr`` is always a multiple of ``fpb`` (the dataset crops to whole blocks)."""
        nblk = Fr // self.fpb
        sig_blk = self.sched.random_sigma(
            B * nblk, device, lo=self.sigma_lo, hi=self.sigma_hi).view(B, nblk)
        return sig_blk.repeat_interleave(self.fpb, dim=1)            # [B, Fr]

    def forward_backward(self, latents: torch.Tensor, cond, *, loss_scale: float = 1.0) -> float:
        """Forward + scaled backward for ONE micro-batch; grads accumulate into
        ``.grad`` (no optimizer step). One flow-matching objective on a clean clip
        ``latents`` [B, F, C, H, W]:

        * causal (student-init): **diffusion forcing** -- an independent noise
          level per block, so the model learns to denoise a noisy block given
          cleaner context (the autoregressive rollout condition).
        * bidirectional (teacher): one noise level for the whole clip (standard
          video diffusion; ``consistent_noise``).

        ``loss_scale`` should be ``1 / accum_steps``: micro-batches are equal-size
        and the loss is a per-element mean, so averaging ``accum_steps`` scaled
        backwards makes the accumulated gradient identical to a true batch that
        many times larger (FSDP then means across ranks). Returns the *unscaled*
        loss for logging."""
        x0 = latents
        B, Fr = x0.shape[0], x0.shape[1]
        eps = torch.randn_like(x0)
        if self.causal:
            sigma_f = self._per_block_sigma(B, Fr, x0.device)       # [B, Fr]
            sig_b = sigma_f.view(B, Fr, *([1] * (x0.ndim - 2))).to(x0.dtype)
            x_sigma = (1 - sig_b) * x0 + sig_b * eps
            t = self.sched.to_timestep(sigma_f)                     # [B, Fr] -> per-frame timestep
        else:
            sigma = self.sched.random_sigma(B, x0.device, lo=self.sigma_lo, hi=self.sigma_hi)
            x_sigma = self.sched.add_noise(x0, eps, sigma)
            t = self.sched.to_timestep(sigma)                       # [B]
        v_pred = self.model.forward_train(
            x_sigma, t, cond, frames_per_block=self.fpb, causal=self.causal)
        v_target = self.sched.velocity_target(x0, eps)
        loss = F.mse_loss(v_pred.float(), v_target.float())
        (loss * loss_scale).backward()
        return loss.item()

    def optimizer_step(self) -> float:
        """Clip, step, and zero grads. Call once per ``accum_steps`` calls to
        ``forward_backward``. Returns the grad-norm."""
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(gn)

    def train_step(self, latents: torch.Tensor, cond) -> dict:
        """Single-step (no accumulation): one micro-batch == one optimizer step."""
        loss = self.forward_backward(latents, cond, loss_scale=1.0)
        gn = self.optimizer_step()
        return {"loss": loss, "grad_norm": gn}
