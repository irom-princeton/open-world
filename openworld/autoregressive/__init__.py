"""Autoregressive, self-forcing world model for robot manipulation.

This subpackage adds an *autoregressive* video world model on top of the
existing (bidirectional, chunked) ``CrtlWorld`` SVD model in
``openworld.world_models.ctrl_world``.

Motivation
----------
The SVD-based model denoises each future chunk from fresh Gaussian noise with no
persistent latent memory, so objects drift / disappear over long rollouts. This
package ports the recipe that NVIDIA OmniDreams uses for long-horizon driving
video:

1. A DiT backbone (Wan2.1-1.3B or Cosmos-Predict2-2B) initialised from a strong
   bidirectional video prior (``backbones/``).
2. Block-causal attention + a KV-cache so the model carries a real latent memory
   across time instead of re-denoising from scratch (``causal/``).
3. Self-forcing / DMD distillation: the model is trained on its *own* rollouts,
   closing the train/inference gap that causes error accumulation
   (``distill/``).

Conditioning stays minimal (robot actions + first frame + history via the
KV-cache); unlike driving there is no HD-map "state" to render, so the burden of
representing object state is carried by the causal memory. See
``docs/AUTOREGRESSIVE.md`` for the full design rationale.

Nothing here imports heavy backbone weights at module import time, so the
package is safe to import for unit tests with the ``DummyDiT`` backbone.
"""

from .config import ARWMArgs

__all__ = ["ARWMArgs"]
