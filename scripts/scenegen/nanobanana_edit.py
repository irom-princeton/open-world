"""Reusable nanobanana (Gemini 2.5 Flash Image) single-image edit.

The implementation now lives in ``openworld.scenegen.nanobanana`` (the canonical
home, shared with the suite builder); this module re-exports it so the existing
``from nanobanana_edit import nanobanana_edit`` recipes and the standalone CLI
keep working:

    GOOGLE_API_KEY=... python scripts/scenegen/nanobanana_edit.py <src.png> <dst.png> "<prompt>"

Requires ``GOOGLE_API_KEY`` and ``uv sync --extra scenegen`` (google-genai).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/scenegen/nanobanana_edit.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openworld.scenegen.nanobanana import nanobanana_edit  # noqa: E402,F401

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: nanobanana_edit.py <src.png> <dst.png> <prompt>")
    nanobanana_edit(sys.argv[1], sys.argv[2], sys.argv[3])
