"""Orchestrate: instruction + image -> guardrail -> nanobanana+multiview -> suite.

This is the directed counterpart to ``openworld.redteam``. You provide one
instruction and one initial wrist image; it writes an Initialization suite that
``scripts/generate_videos.py`` / ``scripts/run_evaluation.py`` can roll out.

Per case it:
  1. runs the guardrail (``openworld.scenegen.guardrail.build_edit_prompt``) once
     to turn the plain instruction into a nanobanana-ready edit prompt;
  2. spawns the bundled diffusers pipeline
     (``external/diffusers/.../multiview_droid_with_nanobanana.py``) as an
     isolated subprocess so FLUX's ~8 GB of VRAM is reclaimed between cases;
  3. resizes the three generated views to the world-model resolution and writes
     ``initialization.yaml`` (robot start state cloned from a template).

The subprocess imports the *fork's* diffusers (FLUX.2-klein lives only there).
On branches where the fork is `uv`-sourced that already resolves; on branches
where it isn't (e.g. ``autoregressive``, which ships PyPI ``diffusers==0.34.0``)
we prepend ``<diffusers-dir>/src`` to the subprocess ``PYTHONPATH`` so the fork
wins. Override the interpreter with ``python_exec`` if you keep the fork in its
own venv.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

from openworld.scenegen.guardrail import DEFAULT_GEMINI_MODEL, build_edit_prompt
from openworld.utils.io import load_yaml

# Repo root = three levels up from this file (openworld/scenegen/pipeline.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DIFFUSERS_DIR = REPO_ROOT / "external" / "diffusers"
DEFAULT_MULTIVIEW_SCRIPT = (
    DEFAULT_DIFFUSERS_DIR / "examples" / "inference" / "multiview_droid_with_nanobanana.py"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "multiview_droid_v0"
DEFAULT_WRIST = DEFAULT_DIFFUSERS_DIR / "assets" / "droid" / "wrist.jpg"
DEFAULT_SIDES = [
    DEFAULT_DIFFUSERS_DIR / "assets" / "droid" / "side1.jpg",
    DEFAULT_DIFFUSERS_DIR / "assets" / "droid" / "side2.jpg",
]
DEFAULT_TEMPLATE_INIT = REPO_ROOT / "configs" / "scenegen" / "template_initialization.yaml"

# Generated suites land here by default, named by --name (see generate_test_cases).
DEFAULT_SUITE_ROOT = REPO_ROOT / "data" / "initializations"

# Maps suite view file <- raw multiview output (see external/.../multiview_*.py).
DEFAULT_VIEW_MAPPING: Dict[str, str] = {
    "wrist": "edited_wrist.jpg",
    "exterior_left": "pred_side1.jpg",
    "exterior_right": "pred_side2.jpg",
}
# World-model resolution (W, H); suite PNGs must match it.
DEFAULT_TARGET_SIZE: Tuple[int, int] = (320, 192)


def _run_multiview(
    *,
    script: Path,
    diffusers_dir: Path,
    python_exec: str,
    edit_prompt: str,
    output_dir: Path,
    checkpoint_path: Path,
    wrist_input: Path,
    side_cond: Sequence[Path],
    num_inference_steps: int,
    seed: int,
    api_key: str,
) -> None:
    """Spawn multiview_droid_with_nanobanana.py for a single case."""
    cmd = [
        python_exec, str(script),
        "--prompt", edit_prompt,
        "--output_dir", str(output_dir),
        "--checkpoint_path", str(checkpoint_path),
        "--wrist_input", str(wrist_input),
        "--num_inference_steps", str(num_inference_steps),
        "--seed", str(seed),
    ]
    for s in side_cond:
        cmd += ["--side_cond", str(s)]

    env = os.environ.copy()
    # nanobanana inside the script reads GOOGLE_API_KEY verbatim.
    env["GOOGLE_API_KEY"] = api_key
    # Make the fork's diffusers importable even if a different diffusers is
    # installed (FLUX.2-klein only exists in the fork).
    fork_src = diffusers_dir / "src"
    if fork_src.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{fork_src}{os.pathsep}{existing}" if existing else str(fork_src)

    proc = subprocess.run(cmd, env=env, cwd=str(diffusers_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"multiview subprocess failed (exit {proc.returncode})")


def _build_case(
    *,
    case_dir: Path,
    raw_dir: Path,
    view_mapping: Dict[str, str],
    target_size: Tuple[int, int],
    template_state: dict,
    instruction: str,
    metadata: dict,
) -> None:
    """Resize the multiview outputs and write one Initialization suite case."""
    case_dir.mkdir(parents=True, exist_ok=True)
    target_w, target_h = int(target_size[0]), int(target_size[1])

    for view_name, src_filename in view_mapping.items():
        src = raw_dir / src_filename
        if not src.exists():
            raise FileNotFoundError(f"multiview output missing: {src}")
        img = Image.open(src).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        img.save(case_dir / f"{view_name}.png")

    # `initial_observation` is intentionally omitted: InitializationDataset infers
    # it from the three view PNGs in the case directory.
    init = {
        "initial_state": template_state,
        "instruction": instruction,
        "metadata": metadata,
    }
    with open(case_dir / "initialization.yaml", "w") as f:
        yaml.safe_dump(init, f, sort_keys=False)


def generate_test_cases(
    *,
    instruction: str,
    init_image: str,
    name: Optional[str] = None,
    out_suite: Optional[str] = None,
    num_cases: int = 1,
    scene_edit: Optional[str] = None,
    guardrail_backend: str = "gemini",
    guardrail_model: str = DEFAULT_GEMINI_MODEL,
    multiview_script: str = str(DEFAULT_MULTIVIEW_SCRIPT),
    diffusers_dir: str = str(DEFAULT_DIFFUSERS_DIR),
    python_exec: Optional[str] = None,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    side_cond: Optional[Sequence[str]] = None,
    num_inference_steps: int = 50,
    seed: int = 0,
    template_init: str = str(DEFAULT_TEMPLATE_INIT),
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    scene: str = "scenegen",
    task_type: str = "manipulation",
    case_prefix: str = "init_",
    start_index: int = 0,
    google_api_key_env: str = "GOOGLE_API_KEY",
    keep_raw: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """Generate ``num_cases`` Initialization cases from one instruction + image.

    The suite is written to ``out_suite`` if given, else to
    ``data/initializations/<name>`` (one of the two is required).

    Returns the list of written case directories. Per-case failures abort the
    run (unlike the redteam loop, which tolerates them) so a directed request
    fails loudly rather than silently producing fewer cases than asked.
    """
    # Resolve the output suite: an explicit out_suite wins, else data/initializations/<name>.
    if out_suite:
        out_suite_path = Path(out_suite).resolve()
    elif name:
        out_suite_path = (DEFAULT_SUITE_ROOT / name).resolve()
    else:
        raise ValueError("provide either `name` (-> data/initializations/<name>) or `out_suite`")

    script = Path(multiview_script).resolve()
    if not script.exists():
        raise FileNotFoundError(
            f"multiview script not found: {script}\n"
            "Clone the diffusers fork into external/diffusers "
            "(see external/README.md) or pass --diffusers-dir / --multiview-script."
        )
    diffusers_path = Path(diffusers_dir).resolve()
    wrist_input = Path(init_image).resolve()
    if not wrist_input.exists():
        raise FileNotFoundError(f"init image not found: {wrist_input}")

    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"multiview checkpoint not found: {ckpt}\n"
            "Download it (bash external/download_models.sh) or pass --checkpoint-path."
        )

    sides = [Path(s).resolve() for s in (side_cond or [str(p) for p in DEFAULT_SIDES])]

    # nanobanana (inside the subprocess) hard-requires the key; fail before GPU time.
    api_key = os.environ.get(google_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{google_api_key_env} is not set; nanobanana cannot run. "
            "Export your Gemini API key first."
        )

    template = load_yaml(str(template_init))
    template_state = template.get("initial_state")
    if template_state is None:
        raise ValueError(f"template_init has no 'initial_state' block: {template_init}")

    # Stage 1: guardrail. Done once; cases differ via the seed / nanobanana sampling.
    edit_prompt = build_edit_prompt(
        scene_edit or instruction,
        backend=guardrail_backend,
        model=guardrail_model,
        api_key_env=google_api_key_env,
        verbose=verbose,
    )

    python_exec = python_exec or sys.executable
    out_suite_path.mkdir(parents=True, exist_ok=True)
    raw_root = out_suite_path / "_raw"

    written: List[Path] = []
    for i in range(num_cases):
        case_id = f"{case_prefix}{start_index + i}"
        raw_dir = raw_root / case_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[scenegen] {case_id}: generating (seed={seed + i}) ...")

        _run_multiview(
            script=script,
            diffusers_dir=diffusers_path,
            python_exec=python_exec,
            edit_prompt=edit_prompt,
            output_dir=raw_dir,
            checkpoint_path=ckpt,
            wrist_input=wrist_input,
            side_cond=sides,
            num_inference_steps=num_inference_steps,
            seed=seed + i,
            api_key=api_key,
        )

        case_dir = out_suite_path / case_id
        _build_case(
            case_dir=case_dir,
            raw_dir=raw_dir,
            view_mapping=DEFAULT_VIEW_MAPPING,
            target_size=target_size,
            template_state=template_state,
            instruction=instruction,
            metadata={
                "suite": out_suite_path.name,
                "scene": scene,
                "task_type": task_type,
                "case_id": case_id,
                "state_length": 7,
                "edit_prompt": edit_prompt,
                "source_image": str(wrist_input),
            },
        )
        written.append(case_dir)
        if verbose:
            print(f"[scenegen] {case_id}: ok -> {case_dir}")

    # Drop a small manifest recording how the suite was built (provenance).
    manifest = {
        "instruction": instruction,
        "scene_edit": scene_edit or instruction,
        "edit_prompt": edit_prompt,
        "guardrail_backend": guardrail_backend,
        "init_image": str(wrist_input),
        "num_cases": num_cases,
        "seed": seed,
        "cases": [p.name for p in written],
    }
    (out_suite_path / "scenegen_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not keep_raw:
        # Best-effort cleanup of the intermediate full-res outputs.
        import shutil

        shutil.rmtree(raw_root, ignore_errors=True)

    if verbose:
        print(f"[scenegen] wrote {len(written)} case(s) -> {out_suite_path}")
    return written
