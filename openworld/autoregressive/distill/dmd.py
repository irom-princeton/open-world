"""Distribution-Matching Distillation (DMD / DMD2) losses.

Two frozen-or-trained "scores" estimate the gradient that pulls the student's
output distribution onto the data distribution:

* **real score** — a frozen *bidirectional teacher* (the strong video prior),
  evaluated with classifier-free guidance.
* **fake score** — a *critic* trained online to denoise the student's own
  samples (tracks the student distribution).

The generator (student) is pushed along ``(x0_fake - x0_real)`` in clean-latent
space; the critic is trained with a plain flow-matching denoising loss on
student samples. Scores are passed in as ``score_fn(x_sigma, timestep, cond) ->
velocity`` callables so this module is backbone-agnostic and unit-testable.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .scheduler import FlowMatchScheduler

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_cfg_score_fn(
    backbone, *, frames_per_block: int, null_cond: torch.Tensor | None = None,
    scale: float = 1.0, causal: bool = False,
) -> ScoreFn:
    """Wrap a backbone's ``forward_train`` as a (CFG'd) velocity score function.

    ``causal=False`` (the default) is correct for the DMD real/fake scores: both
    are *bidirectional* models (the L1b teacher and the critic initialised from
    it), scored on the full clip at a single uniform noise level. This mirrors
    omni-dreams' self-forcing DMD, where the score nets run with
    ``num_frame_per_block = state_t`` (one block = whole clip -> bidirectional)
    and ``uniform_timestep=True``; only the *generator* rollout is block-causal.
    Passing ``causal=True`` here (the old default via ``forward_train``) scored
    the bidirectional teacher under a block-causal mask it was never trained with.
    """

    def score_fn(x_sigma, timestep, cond):
        # Cast scores to fp32: under bf16 autocast the backbone returns bf16, but
        # the DMD score difference (x0_fake - x0_real) and the CFG combination
        # difference two large, similar tensors — keep that arithmetic in fp32.
        v_cond = backbone.forward_train(
            x_sigma, timestep, cond, frames_per_block=frames_per_block, causal=causal).float()
        if scale == 1.0 or null_cond is None:
            return v_cond
        v_uncond = backbone.forward_train(
            x_sigma, timestep, null_cond, frames_per_block=frames_per_block, causal=causal).float()
        return v_uncond + scale * (v_cond - v_uncond)

    return score_fn


def _per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.abs().flatten(1).mean(dim=1).view(-1, *([1] * (x.ndim - 1)))


def dmd_generator_loss(
    x0: torch.Tensor,                 # student-generated clean block [B,F,C,H,W]
    cond: torch.Tensor,
    real_score_fn: ScoreFn,
    fake_score_fn: ScoreFn,
    scheduler: FlowMatchScheduler,
    *,
    lo: float,
    hi: float,
) -> tuple[torch.Tensor, dict]:
    """DMD score-distillation loss on the generator (gradient flows into x0)."""
    B = x0.shape[0]
    sigma = scheduler.random_sigma(B, x0.device, lo=lo, hi=hi)
    eps = torch.randn_like(x0)
    x_sigma = scheduler.add_noise(x0, eps, sigma)
    t = scheduler.to_timestep(sigma)
    # The score fns return fp32; under bf16 autocast x0 is bf16. Do the DMD target
    # arithmetic and the MSE in fp32 so the two near-equal tensors subtract cleanly
    # and the loss/backward dtypes match (bf16 x0 vs fp32 target would otherwise
    # raise "Found dtype BFloat16 but expected Float" in backward). x0.float() keeps
    # the grad path to the generator.
    x0f = x0.float()
    with torch.no_grad():
        v_real = real_score_fn(x_sigma, t, cond)
        v_fake = fake_score_fn(x_sigma, t, cond)
        x0_real = scheduler.x0_from_velocity(x_sigma, v_real, sigma).float()
        x0_fake = scheduler.x0_from_velocity(x_sigma, v_fake, sigma).float()
        grad = x0_fake - x0_real                          # DMD gradient direction
        norm = _per_sample_mean(x0f - x0_real) + 1e-3      # scale-normalise (DMD2)
        target = (x0f - grad / norm).detach()
    loss = 0.5 * F.mse_loss(x0f, target)
    return loss, {"dmd_grad_norm": grad.detach().abs().mean().item()}


def critic_denoising_loss(
    x0: torch.Tensor,                 # student samples (detached inside)
    cond: torch.Tensor,
    fake_score_fn: ScoreFn,
    scheduler: FlowMatchScheduler,
    *,
    lo: float = 0.02,
    hi: float = 0.98,
) -> tuple[torch.Tensor, dict]:
    """Flow-matching denoising loss that trains the critic (fake score) to model
    the student's distribution."""
    x0 = x0.detach()
    B = x0.shape[0]
    sigma = scheduler.random_sigma(B, x0.device, lo=lo, hi=hi)
    eps = torch.randn_like(x0)
    x_sigma = scheduler.add_noise(x0, eps, sigma)
    t = scheduler.to_timestep(sigma)
    v_pred = fake_score_fn(x_sigma, t, cond)              # fp32 (score fn casts)
    v_target = scheduler.velocity_target(x0, eps)
    loss = F.mse_loss(v_pred.float(), v_target.float())   # fp32 MSE (x0/eps may be bf16)
    return loss, {"critic_loss": loss.detach().item()}
