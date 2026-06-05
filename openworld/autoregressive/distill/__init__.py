"""Self-forcing / DMD distillation: train the causal student on its own rollouts.

* :class:`FlowMatchScheduler` — rectified-flow noising/denoising + the few-step
  ``denoising_step_list`` (with the OmniDreams "warp" of integer steps onto the
  sigma grid).
* :mod:`dmd` — Distribution-Matching Distillation losses: the generator
  (student) score-distillation loss against a frozen bidirectional teacher
  ("real score") and a trained critic ("fake score"), plus the critic's own
  denoising loss.
* :mod:`self_forcing` — the autoregressive rollout (few-step, KV-cached, blocks
  fed back as clean context) and the :class:`SelfForcingTrainer` that ties the
  generator/critic updates together.

This is a faithful reference implementation (adapted from the Self-Forcing recipe
that OmniDreams vendors); shapes/gradients are unit-tested on the DummyDiT, but
convergence needs the real Wan/Cosmos teacher + tuning.
"""

from .scheduler import FlowMatchScheduler
from .dmd import dmd_generator_loss, critic_denoising_loss
from .self_forcing import SelfForcingTrainer, generate_rollout

__all__ = [
    "FlowMatchScheduler",
    "dmd_generator_loss",
    "critic_denoising_loss",
    "SelfForcingTrainer",
    "generate_rollout",
]
