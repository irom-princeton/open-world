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


def _pix_slice(pixel_cond, start, n):
    """Slice the per-frame pixel/camera cond ``[B, F_total, K, H, W]`` to the ``n``
    frames of the block starting at frame ``start`` (None -> no extra channels)."""
    return None if pixel_cond is None else pixel_cond[:, start:start + n]


@torch.no_grad()
def _prime_history(generator, history_blocks, cond, scheduler, kv_cache, frames_per_block,
                   pixel_cond=None):
    """Commit clean history blocks into the cache (sigma≈0). Returns next frame."""
    start = 0
    zero_t = scheduler.to_timestep(torch.zeros(history_blocks[0].shape[0], device=history_blocks[0].device))
    for b, blk in enumerate(history_blocks):
        generator.forward_cached(
            blk, zero_t, _cond_for(cond, b), kv_cache=kv_cache, start_frame=start, commit=True,
            pixel_cond=_pix_slice(pixel_cond, start, blk.shape[1]),
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
    random_exit: bool = False,
    pixel_cond: torch.Tensor | None = None,   # [B, F_total, K, H, W] over history+generated frames
) -> tuple[list[torch.Tensor], KVCache]:
    """Few-step, KV-cached autoregressive rollout. Returns the student's clean
    blocks (the last one carrying grad iff ``last_step_grad``) and the cache.

    ``random_exit`` (Self-Forcing / omni-dreams): instead of always running the
    full denoising schedule, stop at a random step ``exit_idx`` -- the same step
    for every block in the rollout -- and take that step's x0 as the block output
    (grad attaches there iff ``last_step_grad``). This averages ~(n_steps+1)/2
    forwards/block instead of n_steps, and trains DMD across a *range* of
    intermediate noise levels rather than only the fully-denoised one.
    ``random_exit=False`` reproduces the old all-steps / grad-on-last behaviour."""
    if kv_cache is None:
        kv_cache = generator.make_kv_cache()
    B = latent_block_shape[0]
    gp = next(generator.parameters())
    dev, pdt = gp.device, gp.dtype
    start = 0
    if history_blocks:
        start = _prime_history(generator, history_blocks, cond, scheduler, kv_cache, frames_per_block,
                               pixel_cond=pixel_cond)

    sigmas = scheduler.sigmas
    n_steps = scheduler.num_steps
    # One exit step for the whole rollout (same across blocks, à la Self-Forcing's
    # same_step_across_blocks). Broadcast under DDP/FSDP so every rank agrees.
    if random_exit and n_steps > 1:
        exit_idx_t = torch.randint(0, n_steps, (1,), device=dev)
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.broadcast(exit_idx_t, src=0)
        exit_idx = int(exit_idx_t.item())
    else:
        exit_idx = n_steps - 1
    blocks: list[torch.Tensor] = []
    for b in range(num_blocks):
        x = torch.randn(latent_block_shape, device=dev, dtype=pdt)
        cb = _cond_for(cond, b)
        pcb = _pix_slice(pixel_cond, start, frames_per_block)
        for i in range(exit_idx + 1):
            sigma_i = sigmas[i].to(dev).expand(B)
            t_i = scheduler.to_timestep(sigma_i)
            last = i == exit_idx
            grad_on = last and last_step_grad
            ctx = torch.enable_grad() if grad_on else torch.no_grad()
            with ctx:
                v = generator.forward_cached(
                    x if grad_on else x.detach(), t_i, cb,
                    kv_cache=kv_cache, start_frame=start, commit=False, pixel_cond=pcb,
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
                x0_clean.detach(), zero_t, cb, kv_cache=kv_cache, start_frame=start, commit=True,
                pixel_cond=pcb,
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
        score_whole_clip: bool = True,
        random_exit: bool = False,
    ):
        self.g, self.critic, self.teacher = generator, critic, teacher
        self.sched = scheduler
        self.fpb = frames_per_block
        self.critic_steps = critic_steps
        self.real_cfg = real_cfg
        self.lo, self.hi = dmd_lo, dmd_hi
        self.max_grad_norm = max_grad_norm
        # Score the critic/teacher on the WHOLE generated clip in one bidirectional
        # forward (matches omni-dreams / Self-Forcing, where the score nets run with
        # num_frame_per_block = state_t). The alternative scores each block in
        # isolation: many more forward passes AND it feeds the bidirectional teacher
        # 2-frame fragments it was never trained on. Default on; flip to False to
        # reproduce the old per-block behaviour for an A/B.
        self.score_whole_clip = score_whole_clip
        # Stop each rollout at a random denoising step (see ``generate_rollout``).
        # Large rollout speedup; changes the training signal, so it is opt-in.
        self.random_exit = random_exit
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.opt_g = torch.optim.AdamW(self.g.parameters(), lr=gen_lr,
                                       betas=betas, weight_decay=weight_decay)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr,
                                       betas=betas, weight_decay=weight_decay)

    def _clip(self, params) -> float:
        """Clip (if enabled) and return the total grad-norm *before* clipping --
        a key collapse signal that was previously not surfaced. Under FSDP2 the
        params carry DTensor grads; ``clip_grad_norm_`` reduces across shards, so
        its return is the true global norm. When clipping is off we fall back to a
        best-effort local norm (correct for single-process; per-shard under FSDP)."""
        params = [p for p in params]
        if self.max_grad_norm and self.max_grad_norm > 0:
            return float(torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm))
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return float(torch.norm(torch.stack([g.norm() for g in grads])))

    def _score_fns(self, null_cond):
        # Both scores are bidirectional (teacher + critic-from-teacher), evaluated
        # on the full clip at a single uniform noise level -> causal=False.
        real = make_cfg_score_fn(self.teacher, frames_per_block=self.fpb,
                                 null_cond=null_cond, scale=self.real_cfg, causal=False)
        fake = make_cfg_score_fn(self.critic, frames_per_block=self.fpb, causal=False)
        return real, fake

    def _rollout(self, cond, num_blocks, latent_block_shape, history_blocks, last_step_grad):
        return generate_rollout(
            self.g, cond, self.sched, frames_per_block=self.fpb, num_blocks=num_blocks,
            latent_block_shape=latent_block_shape, history_blocks=history_blocks,
            last_step_grad=last_step_grad, random_exit=self.random_exit,
        )

    @staticmethod
    def _hist_frames(history_blocks):
        """Latent frames of GT history that prime the rollout -- the absolute frame
        offset of the *generated* clip, used to align the score cond to it."""
        return sum(b.shape[1] for b in history_blocks) if history_blocks else 0

    def _score_cond(self, cond, start_frame, num_frames):
        """Cond for scoring a generated sub-window: slice the full-window action
        cond to the frames being scored so ``cross_attn_aligned`` / ``adaln`` stay
        aligned (no-op for global cond modes). Fixes the off-by-history misalignment
        where the score saw generated frame ``j`` conditioned on action ``j`` rather
        than action ``start_frame + j``."""
        return self.g.slice_cond_to_frames(_cond_for(cond, 0), start_frame, num_frames)

    def critic_step(self, cond, *, num_blocks, latent_block_shape, history_blocks=None):
        # detach cond: the critic step must not push gradients into the generator
        # / action conditioner (it shares the cond tensor across blocks).
        cond_c = cond.detach() if torch.is_tensor(cond) else cond
        with torch.no_grad():
            blocks, _ = self._rollout(cond_c, num_blocks, latent_block_shape, history_blocks, last_step_grad=False)
        _, fake = self._score_fns(None)
        self.opt_c.zero_grad()
        hist = self._hist_frames(history_blocks)
        if self.score_whole_clip:
            # One denoising loss over the full clip, mean-reduced -- matching
            # omni-dreams / Self-Forcing (``F.mse_loss(..., reduction='mean')``).
            # The lr/lr_critic in the configs are copied from those recipes, which
            # assume this mean scale; the old sum-over-blocks scale over-drove the
            # optimizers by num_blocks (~8x), which diverged the student to noise.
            x0 = torch.cat(blocks, dim=1)
            cond_s = self._score_cond(cond_c, hist, x0.shape[1])
            loss = critic_denoising_loss(x0, cond_s, fake, self.sched, lo=self.lo, hi=self.hi)[0]
            loss.backward()
            reported = loss
        else:
            # Per-block scoring (A/B). Mean over blocks == mean over the clip, so the
            # gradient scale matches the whole-clip path -- the flag isolates *where*
            # the score is evaluated, not the effective LR.
            losses = [critic_denoising_loss(
                          x0, self._score_cond(cond_c, hist + b * self.fpb, self.fpb),
                          fake, self.sched, lo=self.lo, hi=self.hi)[0]
                      for b, x0 in enumerate(blocks)]
            total = torch.stack(losses).mean()
            total.backward()
            reported = total
        critic_grad_norm = self._clip(self.critic.parameters())
        self.opt_c.step()
        return {"critic_loss": reported.item(), "critic_grad_norm": critic_grad_norm}

    def generator_step(self, cond, null_cond, *, num_blocks, latent_block_shape, history_blocks=None):
        blocks, _ = self._rollout(cond, num_blocks, latent_block_shape, history_blocks, last_step_grad=True)
        self.opt_g.zero_grad()
        grad_blocks = [x0 for x0 in blocks if x0.requires_grad]
        if not grad_blocks:
            return {"gen_loss": 0.0}
        # Generated-clip statistics: std collapsing toward 0 (a constant/near-black
        # output) is the clearest collapse signal -- the logged DMD loss does NOT
        # reflect it. Cheap to compute and logged every step.
        x0_cat = torch.cat(grad_blocks, dim=1)
        with torch.no_grad():
            x0d = x0_cat.detach().float()
            x0_std = x0d.std().item()
            x0_absmean = x0d.abs().mean().item()
        hist = self._hist_frames(history_blocks)
        if self.score_whole_clip:
            # Single DMD loss over the full clip (1 fake + 1 teacher-CFG forward
            # instead of 3 per block), mean-reduced to match omni-dreams /
            # Self-Forcing. The configs' lr/lr_critic assume this mean scale; the old
            # sum-over-blocks scale over-drove the generator by num_blocks (~8x) and
            # diverged the student off-manifold (rainbow-noise samples).
            # Cond/null are sliced to the generated window so the score's per-frame
            # action alignment matches the clip (off-by-history fix).
            cond_s = self._score_cond(cond, hist, x0_cat.shape[1])
            null_s = self.g.slice_cond_to_frames(null_cond, hist, x0_cat.shape[1])
            real, fake = self._score_fns(null_s)
            loss, aux = dmd_generator_loss(x0_cat, cond_s, real, fake, self.sched, lo=self.lo, hi=self.hi)
            loss.backward()
            reported = loss
            dmd_grad_norm = aux["dmd_grad_norm"]
        else:
            # Per-block scoring (A/B). Mean over blocks == mean over the clip, so the
            # gradient scale matches the whole-clip path; the flag isolates *where*
            # the score is evaluated, not the effective LR.
            outs = []
            for b, x0 in enumerate(grad_blocks):
                sf = hist + b * self.fpb
                cond_b = self._score_cond(cond, sf, self.fpb)
                null_b = self.g.slice_cond_to_frames(null_cond, sf, self.fpb)
                real, fake = self._score_fns(null_b)
                outs.append(dmd_generator_loss(x0, cond_b, real, fake, self.sched, lo=self.lo, hi=self.hi))
            total = torch.stack([o[0] for o in outs]).mean()
            total.backward()
            reported = total
            dmd_grad_norm = sum(o[1]["dmd_grad_norm"] for o in outs) / len(outs)
        gen_grad_norm = self._clip(self.g.parameters())
        self.opt_g.step()
        return {
            "gen_loss": reported.item(),
            "gen_grad_norm": gen_grad_norm,
            "dmd_grad_norm": dmd_grad_norm,
            "gen_x0_std": x0_std,
            "gen_x0_absmean": x0_absmean,
        }

    def train_step(self, cond, null_cond, *, num_blocks, latent_block_shape, history_blocks=None):
        c_outs = [
            self.critic_step(cond, num_blocks=num_blocks, latent_block_shape=latent_block_shape, history_blocks=history_blocks)
            for _ in range(self.critic_steps)
        ]
        g_out = self.generator_step(
            cond, null_cond, num_blocks=num_blocks, latent_block_shape=latent_block_shape, history_blocks=history_blocks
        )
        n = len(c_outs)
        return {
            **g_out,                                                      # gen_loss + gen/dmd grad norms + x0 stats
            "critic_loss": sum(o["critic_loss"] for o in c_outs) / n,
            "critic_grad_norm": sum(o["critic_grad_norm"] for o in c_outs) / n,
        }
