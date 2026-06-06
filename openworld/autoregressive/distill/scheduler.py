"""Rectified-flow scheduler with a few-step denoising list.

Convention (data at sigma=0, noise at sigma=1)::

    x_sigma = (1 - sigma) * x0 + sigma * eps
    v*      = eps - x0                          (the flow velocity, d x / d sigma)
    x0_hat  = x_sigma - sigma * v_pred          (recover clean from a velocity pred)

``denoising_step_list`` holds integer timesteps in ``[0, num_train_timestep]``
(OmniDreams default ``[1000, 750, 500, 250]``); ``sigma = step / num_train_timestep``.
``warp=True`` applies Wan/Cosmos's flow ``shift`` so the few steps are spaced the
way the teacher was trained (more resolution near the data end).
"""

from __future__ import annotations

import torch


class FlowMatchScheduler:
    def __init__(
        self,
        denoising_step_list: tuple[int, ...] = (1000, 750, 500, 250),
        *,
        num_train_timestep: int = 1000,
        warp: bool = True,
        shift: float = 5.0,
    ):
        self.num_train_timestep = num_train_timestep
        self.shift = shift
        steps = torch.tensor(sorted(denoising_step_list, reverse=True), dtype=torch.float32)
        sigmas = steps / num_train_timestep
        if warp:
            sigmas = self._shift_sigma(sigmas)
        # ascending-in-noise list used for sampling: start at the noisiest.
        self.sigmas = sigmas                     # [N], descending (1.0-ish -> small)
        self.warp = warp

    def _shift_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        s = self.shift
        return s * sigma / (1 + (s - 1) * sigma)

    # -- noising ---------------------------------------------------------
    # sigma is built in fp32; cast to the operand dtype so a bf16 latent stays
    # bf16 (otherwise the multiply promotes to fp32 and leaks fp32 into the loss
    # graph, breaking the bf16 backward).
    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
        return (1 - sigma) * x0 + sigma * eps

    def velocity_target(self, x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return eps - x0

    def x0_from_velocity(self, x_sigma: torch.Tensor, v_pred: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(-1, *([1] * (x_sigma.ndim - 1))).to(x_sigma.dtype)
        return x_sigma - sigma * v_pred

    def step_sigma(self, i: int, device, batch: int) -> torch.Tensor:
        """The sigma (as a [B] tensor) for denoising-list index ``i``."""
        return self.sigmas[i].to(device).expand(batch)

    def to_timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map a sigma in [0,1] back to the integer-ish timestep the backbones
        expect for their timestep embedding."""
        return (sigma * self.num_train_timestep)

    def random_sigma(self, batch: int, device, *, lo: float, hi: float) -> torch.Tensor:
        """Uniform sigma in [lo, hi] for the DMD score-distillation noise level."""
        u = torch.rand(batch, device=device)
        return lo + (hi - lo) * u

    @property
    def num_steps(self) -> int:
        return len(self.sigmas)
