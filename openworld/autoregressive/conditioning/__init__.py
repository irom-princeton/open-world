"""Conditioning for the AR world model: robot actions (+ optional text) and
multi-view packing.

Deliberately minimal: unlike OmniDreams there is no HD-map "state" to render.
Object state is carried by the causal KV-cache, not by the conditioning. The
condition is just per-frame action embeddings (projected to the backbone's
cross-attention width) plus the first/history frames (fed as clean latents) — so
the model must *learn* object dynamics rather than read them off a control image.
"""

from .action import ActionConditioner
from .multiview import ViewPacker, ViewEmbedding

__all__ = ["ActionConditioner", "ViewPacker", "ViewEmbedding"]
