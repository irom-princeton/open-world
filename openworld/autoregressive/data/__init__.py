"""Data layer for the autoregressive world model.

Two stages, so the framework flexibly ingests different raw-data formats:

1. **Formats** (``data/formats/``) — a small adapter per raw dataset type
   (``droid_ctrl_world`` is the first). Each implements ``list_episodes`` +
   ``load_episode`` (RGB video paths + video-rate 7-d actions + text). This is
   the only thing a new dataset type needs to provide.
2. **Precompute** (``scripts/preprocess_ar_latents.py`` + ``encode.py``) turns
   any format's RGB into a *standard* on-disk latent layout (16-ch Wan-VAE
   latents per camera + per-latent-frame actions + text). A single shared
   :class:`ARLatentDataset` then trains on that layout regardless of source.

So adding a data type = writing one format adapter; the encoder, latent layout,
dataset, and trainer are reused unchanged.
"""

from .formats import build_format
from .dataset import ARLatentDataset

__all__ = ["build_format", "ARLatentDataset"]
