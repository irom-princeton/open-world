"""Autoregressive self-forcing rollout + DMD trainer.

The student generates a clip block-by-block with the KV-cache; each finished
block is committed to the cache (as clean context) and fed forward — so the
model is trained on the *same* imperfect history it will see at inference. This
is what closes the train/test gap that makes the SVD chunked sampler drift.

``generate_rollout`` produces the student's clean blocks. ``SelfForcingTrainer``
alternates DMD2-style updates: several critic (fake-score) steps that learn to
denoise the student's samples, then one generator step whose DMD loss pulls the
student toward the teacher distribution.

Gradient strategy (Self-Forcing): intermediate denoising steps and the clean
"commit" pass run under ``no_grad`` and the cache is treated as constant context;
gradient is retained only through the *final* denoising step of each block, which
is where the DMD loss attaches. This keeps backprop through long rollouts
tractable.
"""

from __future__ import annotations

from typing import Callable

import torch

from .scheduler import FlowMatchScheduler
from .dmd import dmd_generator_loss, critic_denoising_loss, make_cfg_score_fn
from ..causal.kv_cache import KVCache


def _cond_for(cond, block_idx):
    return cond(block_idx) if callable(cond) else cond


@torch.no_grad()
def _prime_history(generator, history_blocks, cond, scheduler, kv_cache, frames_per_block):
    """Commit clean history blocks into the cache (sigma≈0). Returns next frame."""
    start = 0
    zero_t = scheduler.to_timestep(torch.zeros(history_blocks[0].shape[0], device=history_blocks[0].device))
    for b, blk in enumerate(history_blocks):
        generator.forward_cached(
            blk, zero_t, _cond_for(cond, b), kv_cache=kv_cache, start_frame=start, commit=True
        )
        start += blk.shape[1]
    return start


def generate_rollout(
    generator,
    cond,
    scheduler: FlowMatchScheduler,
    *,
    frames_per_block: int,
    num_blocks: int,
    latent_block_shape: tuple,          # (B, fpb, C, H, W)
    history_blocks: list[torch.Tensor] | None = None,
    kv_cache: KVCache | None = None,
    last_step_grad: bool = False,
) -> tuple[list[torch.Tensor], KVCache]:
    """Few-step, KV-cached autoregressive rollout. Returns the student's clean
    blocks (the last one carrying grad iff ``last_step_grad``) and the cache."""
    if kv_cache is None:
        kv_cache = generator.make_kv_cache()
    B = latent_block_shape[0]
    gp = next(generator.parameters())
    dev, pdt = gp.device, gp.dtype
    start = 0
    if history_blocks:
        start = _prime_history(generator, history_blocks, cond, scheduler, kv_cache, frames_per_block)

    sigmas = scheduler.sigmas
    n_steps = scheduler.num_steps
    blocks: list[torch.Tensor] = []
    for b in range(num_blocks):
        x = torch.randn(latent_block_shape, device=dev, dtype=pdt)
        cb = _cond_for(cond, b)
        for i in range(n_steps):
            sigma_i = sigmas[i].to(dev).expand(B)
            t_i = scheduler.to_timestep(sigma_i)
            last = i == n_steps - 1
            grad_on = last and last_step_grad
            ctx = torch.enable_grad() if grad_on else torch.no_grad()
            with ctx:
                v = generator.forward_cached(
                    x if grad_on else x.detach(), t_i, cb,
                    kv_cache=kv_cache, start_frame=start, commit=False,
                )
                x0_hat = scheduler.x0_from_velocity(x if grad_on else x.detach(), v, sigma_i)
            if not last:
                # renoise to the next (lower) sigma for the next denoising step
                with torch.no_grad():
                    sigma_next = sigmas[i + 1].to(dev).expand(B)
                    x = scheduler.add_noise(x0_hat.detach(), torch.randn_like(x0_hat), sigma_next)
            else:
                x0_clean = x0_hat
        # commit the clean block as context for subsequent blocks
        with torch.no_grad():
            zero_t = scheduler.to_timestep(torch.zeros(B, device=dev))
            generator.forward_cached(
                x0_clean.detach(), zero_t, cb, kv_cache=kv_cache, start_frame=start, commit=True
            )
        blocks.append(x0_clean)
        start += frames_per_block
    return blocks, kv_cache


class SelfForcingTrainer:
    """Holds generator/critic/teacher + optimizers and runs DMD2 updates."""

    def __init__(
        self,
        generator,
        critic,                      # "fake score" backbone (trained online)
        teacher,                     # "real score" backbone (frozen, bidirectional)
        scheduler: FlowMatchScheduler,
        *,
        frames_per_block: int,
        gen_lr: float = 6e-6,
        critic_lr: float = 6e-6,
        critic_steps: int = 5,
        real_cfg: float = 3.5,
        dmd_lo: float = 0.02,
        dmd_hi: float = 0.98,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,        # 0 -> no clipping
    ):
        self.g, self.critic, self.teacher = generator, critic, teacher
        self.sched = scheduler
        self.fpb = frames_per_block
        self.critic_steps = critic_steps
        self.real_cfg = real_cfg
        self.lo, self.hi = dmd_lo, dmd_hi
        self.max_grad_norm = max_grad_norm
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.opt_g = torch.optim.AdamW(self.g.parameters(), lr=gen_lr,
                                       betas=betas, weight_decay=weight_decay)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr,
                                       betas=betas, weight_decay=weight_decay)

    def _clip(self, params):
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

    def _score_fns(self, null_cond):
        real = make_cfg_score_fn(self.teacher, frames_per_block=self.fpb, null_cond=null_cond, scale=self.real_cfg)
        fake = make_cfg_score_fn(self.critic, frames_per_block=self.fpb)
        return real, fake

    def _rollout(self, cond, num_blocks, latent_block_shape, history_blocks, last_step_grad):
        return generate_rollout(
            self.g, cond, self.sched, frames_per_block=self.fpb, num_blocks=num_blocks,
            latent_block_shape=latent_block_shape, history_blocks=history_blocks,
            last_step_grad=last_step_grad,
        )

    def critic_step(self, cond, *, num_blocks, latent_block_shape, history_blocks=None):
        # detach cond: the critic step must not push gradients into the generator
        # / action conditioner (it shares the cond tensor across blocks).
        cond_c = cond.detach() if torch.is_tensor(cond) else cond
        with torch.no_grad():
            blocks, _ = self._rollout(cond_c, num_blocks, latent_block_shape, history_blocks, last_step_grad=False)
        _, fake = self._score_fns(None)
        self.opt_c.zero_grad()
        # Blocks share the (detached) cond graph, so accumulate then backward once.
        losses = [critic_denoising_loss(x0, _cond_for(cond_c, 0), fake, self.sched, lo=self.lo, hi=self.hi)[0]
                  for x0 in blocks]
        total = torch.stack(losses).sum()
        total.backward()
        self._clip(self.critic.parameters())
        self.opt_c.step()
        return (total / max(1, len(losses))).item()

    def generator_step(self, cond, null_cond, *, num_blocks, latent_block_shape, history_blocks=None):
        blocks, _ = self._rollout(cond, num_blocks, latent_block_shape, history_blocks, last_step_grad=True)
        real, fake = self._score_fns(null_cond)
        self.opt_g.zero_grad()
        losses = [dmd_generator_loss(x0, _cond_for(cond, 0), real, fake, self.sched, lo=self.lo, hi=self.hi)[0]
                  for x0 in blocks if x0.requires_grad]
        if not losses:
            return 0.0
        # All blocks share the cond + backbone graph -> one backward over the sum.
        total = torch.stack(losses).sum()
        total.backward()
        self._clip(self.g.parameters())
        self.opt_g.step()
        return (total / len(losses)).item()

    def train_step(self, cond, null_cond, *, num_blocks, latent_block_shape, history_blocks=None):
        c_losses = [
            self.critic_step(cond, num_blocks=num_blocks, latent_block_shape=latent_block_shape, history_blocks=history_blocks)
            for _ in range(self.critic_steps)
        ]
        g_loss = self.generator_step(
            cond, null_cond, num_blocks=num_blocks, latent_block_shape=latent_block_shape, history_blocks=history_blocks
        )
        return {"gen_loss": g_loss, "critic_loss": sum(c_losses) / len(c_losses)}
