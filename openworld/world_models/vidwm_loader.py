from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional


# The ``vidwm`` package is vendored as source under ``external/vidwm`` (it has no
# standalone upstream — it lives in the open-world monorepo). Put that directory
# on sys.path so ``import vidwm`` resolves to the in-repo copy.
DEFAULT_VIDWM_REPO = Path(__file__).resolve().parents[2] / "external" / "vidwm"


def ensure_vidwm_repo_on_path(repo_path: Optional[str] = None) -> Path:
    repo_root = Path(repo_path or DEFAULT_VIDWM_REPO).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"VidWM package root not found at {repo_root}. "
            "Set `world_model.params.repo_path` explicitly if you want to use a "
            "different VidWM checkout."
        )

    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root
