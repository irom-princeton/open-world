"""Format registry. ``build_format(name, root, **kw)`` -> WorldModelFormat.

Add a new raw-dataset type by writing a ``WorldModelFormat`` adapter and
registering it here — nothing else in the pipeline changes.
"""

from __future__ import annotations

from .base import WorldModelFormat
from .droid_ctrl_world import DroidCtrlWorldFormat

FORMATS = {
    "droid_ctrl_world": DroidCtrlWorldFormat,
}


def build_format(name: str, root: str, **kwargs) -> WorldModelFormat:
    if name not in FORMATS:
        raise ValueError(f"unknown data format {name!r}; choose from {list(FORMATS)}")
    return FORMATS[name](root, **kwargs)
