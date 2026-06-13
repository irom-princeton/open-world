from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional


DEFAULT_CTRL_WORLD_PARENT = Path(__file__).resolve().parents[2] / "external"


def ensure_ctrl_world_on_path(repo_path: Optional[str] = None) -> Path:
    """Put the parent directory of the ``ctrl_world`` package on ``sys.path``.

    The vendored Ctrl-World package lives at ``open-world/external/ctrl_world``.
    Adding its parent (``external``) to ``sys.path`` makes ``import ctrl_world``
    resolve to the vendored copy.  Pass ``repo_path`` to point at a different
    checkout (e.g. the upstream Ctrl-World repo directly).
    """
    parent = Path(repo_path or DEFAULT_CTRL_WORLD_PARENT).resolve()
    if not (parent / "ctrl_world").exists():
        raise FileNotFoundError(
            f"Ctrl-World package not found at {parent / 'ctrl_world'}. "
            "Set `world_model.params.repo_path` explicitly if you want to use a "
            "different Ctrl-World checkout."
        )

    parent_str = str(parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
    return parent
