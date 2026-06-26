"""View selection for multi-camera training.

The preprocessed latents store every camera of an episode (``[V, C, Lf, h, w]``,
DROID order ``[side, side, wrist]``). At load time we subset those views down to
``num_cams`` so the *same* preprocessed dataset can train either the full 3-view
model or a reduced 2-view model -- no re-preprocessing needed.

The reduced model always keeps the wrist camera and draws its remaining views
from the side cameras. ``select_view_indices`` is the single source of truth for
that choice, shared by the training dataset (random side per clip) and the
preview / eval replay path (deterministic), so they never drift apart.
"""

from __future__ import annotations

import random


def select_view_indices(
    v_stored: int, num_cams: int, wrist_view_idx: int, *, deterministic: bool
) -> list[int]:
    """Pick ``num_cams`` view indices from the ``v_stored`` stored views.

    The wrist view (``wrist_view_idx``) is always kept; the remaining
    ``num_cams - 1`` views are sampled from the side cameras -- randomly per call
    when ``deterministic`` is False (training augmentation), otherwise the first
    side views in stored order (stable previews / eval). When ``num_cams >=
    v_stored`` all views are used in stored order.

    The wrist view is placed last so the height-stack layout is consistent across
    the 2- and 3-view modes (wrist always at the bottom of the stack).
    """
    n = min(num_cams, v_stored)
    if n >= v_stored:
        return list(range(v_stored))
    wrist = wrist_view_idx % v_stored
    sides = [i for i in range(v_stored) if i != wrist]
    n_sides = n - 1
    if n_sides <= 0:
        return [wrist]
    chosen = sides[:n_sides] if deterministic else random.sample(sides, n_sides)
    chosen.sort()  # preserve stored relative order among the side views
    return chosen + [wrist]
