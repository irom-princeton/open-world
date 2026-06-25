"""nanobanana all-views scene edit -> Initialization suite (no GPU / no FLUX).

This is the *usable* counterpart to ``openworld.scenegen.pipeline``: instead of
the multiview add-object path (which needs a GPU + the diffusers fork), it edits
each of a base's three real views directly with nanobanana (Gemini 2.5 Flash
Image). Nothing is added or removed from a fresh viewpoint — the scene *content*
(objects, robot, framing) stays put and only what the edit prompt targets
changes. That makes it the right tool for background / lighting / material /
content edits, and it runs anywhere ``GOOGLE_API_KEY`` is set.

Two layers:

* :func:`nanobanana_edit` -- the single-image upscale -> edit -> downscale helper
  (the canonical home; ``scripts/scenegen/nanobanana_edit.py`` re-exports it).
* :func:`build_suite` -- turn a *base* (``assets/tri`` or ``assets/irom``, or any
  dir of three views) plus a *list of edits* into a full suite under
  ``data/benchmark/<name>/``. Drive it from a YAML spec with
  :func:`build_suite_from_spec` (see ``scripts/scenegen/build_suite.py``).

Each edit becomes one ``init_<i>`` case: the three views are nanobanana-edited
with the edit's prompt (plus a shared ``keep`` clause that pins everything the
edit should leave alone) and an ``initialization.yaml`` is written, cloning the
robot start state from the base's ``template.yaml``.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml
from PIL import Image

# Repo root = two levels up from this file (openworld/scenegen/nanobanana.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPO_ROOT / "assets"
BENCHMARK_ROOT = REPO_ROOT / "data" / "benchmark"

VIEWS = ("wrist", "exterior_left", "exterior_right")

MODEL = os.environ.get("NANOBANANA_MODEL", "gemini-2.5-flash-image")
EDIT_WIDTH = int(os.environ.get("NANOBANANA_EDIT_WIDTH", "1024"))

_client = None


def _get_client() -> "object":
    global _client
    if _client is None:
        from google import genai  # imported lazily: only needed when actually editing

        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def nanobanana_edit(src: str, dst: str, prompt: str) -> None:
    """Edit ``src`` with ``prompt`` and write the result to ``dst`` at src's size.

    Editing a tiny 320x192 world-model frame directly makes nanobanana
    reframe/warp, so we upscale the input, edit at higher resolution, then resize
    back to native. Requires ``GOOGLE_API_KEY`` + ``uv sync --extra scenegen``.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    big = img.resize((EDIT_WIDTH, round(EDIT_WIDTH * h / w)), Image.LANCZOS)
    resp = _get_client().models.generate_content(model=MODEL, contents=[prompt, big])
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            out = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            out = out.resize((w, h), Image.LANCZOS)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            out.save(dst)
            print(f"  edited {os.path.basename(src)} -> {dst}  ({w}x{h})")
            return
        if getattr(part, "text", None):
            print(f"  [nanobanana text] {part.text}")
    raise RuntimeError(f"nanobanana returned no image for {src}")


def resolve_base(base: str) -> Dict[str, object]:
    """Resolve a base name (``tri`` / ``irom``) or a path into views + template.

    A *base* is a directory with ``wrist.png``, ``exterior_left.png``,
    ``exterior_right.png`` and (optionally) a ``template.yaml`` carrying
    ``initial_state`` and a default ``scene`` tag. Named bases live in
    ``assets/<name>``; anything else is treated as an explicit path.

    Returns ``{"dir": Path, "initial_state": dict, "scene": Optional[str]}``.
    """
    candidate = ASSETS_ROOT / base
    base_dir = candidate if candidate.is_dir() else Path(base).expanduser().resolve()
    if not base_dir.is_dir():
        known = sorted(p.name for p in ASSETS_ROOT.iterdir() if p.is_dir()) if ASSETS_ROOT.is_dir() else []
        raise FileNotFoundError(
            f"base '{base}' not found (looked in assets/{base} and as a path). "
            f"Known assets/ bases: {known or '(none)'}"
        )
    missing = [v for v in VIEWS if not (base_dir / f"{v}.png").exists()]
    if missing:
        raise FileNotFoundError(f"base {base_dir} is missing view(s): {missing}")

    initial_state: Optional[dict] = None
    scene: Optional[str] = None
    tpl = base_dir / "template.yaml"
    if tpl.exists():
        loaded = yaml.safe_load(tpl.read_text()) or {}
        initial_state = loaded.get("initial_state")
        scene = loaded.get("scene")
    if initial_state is None:
        raise ValueError(
            f"base {base_dir} has no initial_state; add a template.yaml with an "
            "initial_state block (see assets/tri/template.yaml)."
        )
    return {"dir": base_dir, "initial_state": initial_state, "scene": scene}


def build_suite(
    *,
    base: str,
    edits: Sequence[dict],
    name: Optional[str] = None,
    out_dir: Optional[str] = None,
    keep: Optional[str] = None,
    task_type: str = "manipulation",
    scene: Optional[str] = None,
    start_index: int = 0,
    benchmark_root: Path = BENCHMARK_ROOT,
    google_api_key_env: str = "GOOGLE_API_KEY",
    verbose: bool = True,
) -> List[Path]:
    """Build an Initialization suite by nanobanana-editing all three base views.

    ``edits`` is a list of dicts; each becomes one ``init_<i>`` case and may set:
      - ``prompt`` (required): the nanobanana edit prompt.
      - ``instruction`` (required): the policy command stored in the case.
      - ``label`` (optional): short tag recorded under ``metadata.edit_label``.
      - ``keep`` (optional): per-edit override of the shared ``keep`` clause.
      - ``views`` (optional): subset of the three views to edit (default: all;
        un-edited views are copied straight from the base).

    The suite is written to ``out_dir`` if given, else ``data/benchmark/<name>``.
    Per-edit failures abort the run so a directed request fails loudly.
    """
    if out_dir:
        suite_dir = Path(out_dir).expanduser().resolve()
    elif name:
        suite_dir = (Path(benchmark_root) / name).resolve()
    else:
        raise ValueError("provide either `name` (-> data/benchmark/<name>) or `out_dir`")
    if not edits:
        raise ValueError("`edits` is empty; nothing to build")

    if not os.environ.get(google_api_key_env):
        raise RuntimeError(
            f"{google_api_key_env} is not set; nanobanana cannot run. "
            "Export your Gemini API key first (uv sync --extra scenegen)."
        )

    resolved = resolve_base(base)
    base_dir: Path = resolved["dir"]  # type: ignore[assignment]
    initial_state = resolved["initial_state"]
    suite_scene = scene or resolved["scene"] or "scenegen"

    suite_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    manifest_cases = []

    for i, edit in enumerate(edits):
        idx = start_index + i
        case_id = f"init_{idx}"
        prompt = edit.get("prompt")
        instruction = edit.get("instruction")
        if not prompt or not instruction:
            raise ValueError(f"edit {idx} needs both 'prompt' and 'instruction': {edit!r}")
        label = edit.get("label", "")
        keep_clause = edit.get("keep", keep)
        full_prompt = f"{prompt} {keep_clause}".strip() if keep_clause else prompt
        edit_views = edit.get("views") or list(VIEWS)

        case_dir = suite_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[scenegen] {case_id} ({label or 'edit'}): editing {edit_views} ...")

        for v in VIEWS:
            src = base_dir / f"{v}.png"
            dst = case_dir / f"{v}.png"
            if v in edit_views:
                nanobanana_edit(str(src), str(dst), full_prompt)
            else:  # leave this view untouched — copy the base view through
                Image.open(src).convert("RGB").save(dst)

        init = {
            "initial_state": initial_state,
            "instruction": instruction,
            "metadata": {
                "suite": suite_dir.name,
                "scene": suite_scene,
                "task_type": task_type,
                "case_id": case_id,
                "state_length": 7,
                "edit_mode": "nanobanana_all_views",
                "edit_label": label,
                "edit_prompt": full_prompt,
                "base": base,
            },
        }
        with open(case_dir / "initialization.yaml", "w") as f:
            yaml.safe_dump(init, f, sort_keys=False)
        written.append(case_dir)
        manifest_cases.append({
            "case_id": case_id,
            "label": label,
            "instruction": instruction,
            "prompt": full_prompt,
            "views": edit_views,
        })
        if verbose:
            print(f"[scenegen] {case_id}: ok -> {case_dir}")

    manifest = {
        "base": base,
        "base_dir": str(base_dir),
        "scene": suite_scene,
        "task_type": task_type,
        "keep": keep,
        "edit_mode": "nanobanana_all_views",
        "num_cases": len(written),
        "cases": manifest_cases,
    }
    (suite_dir / "scenegen_manifest.json").write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"[scenegen] wrote {len(written)} case(s) -> {suite_dir}")
    return written


def build_suite_from_spec(spec_path: str, **overrides) -> List[Path]:
    """Load a YAML suite spec and build it. ``overrides`` win over spec keys.

    Spec keys: ``base``, ``name`` / ``out_dir``, ``keep``, ``task_type``,
    ``scene``, ``start_index``, and ``edits`` (the list of per-case edits).
    """
    spec = yaml.safe_load(Path(spec_path).read_text()) or {}
    spec.update(overrides)
    edits = spec.pop("edits", None)
    if not edits:
        raise ValueError(f"spec {spec_path} has no 'edits' list")
    return build_suite(edits=edits, **spec)
