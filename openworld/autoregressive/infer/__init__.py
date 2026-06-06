"""Inference utilities for the autoregressive world model.

``replay`` is the first inference setup: open-loop *trajectory replay* — prime the
model with a ground-truth first frame / history, feed the recorded action
sequence, and let the model autoregressively generate the rest of the clip so it
can be compared side-by-side against the ground truth. It is the simplest stand-in
for the eventual closed-loop "interact with any policy" entrypoint.
"""

from .replay import (
    load_full_episode,
    load_action_stats,
    normalize_actions,
    replay_episode_latents,
)

__all__ = [
    "load_full_episode",
    "load_action_stats",
    "normalize_actions",
    "replay_episode_latents",
]
