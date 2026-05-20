"""Stage B: scene generation for the redteam loop.

For each task proposed in stage A, spawn the multiview image-edit pipeline
(nanobanana wrist edit + FLUX.2-klein multiview) as an isolated subprocess,
then assemble the three generated views into an ``Initialization`` suite case
that ``scripts/generate_videos.py`` can roll out.

Run as:
    python -m openworld.redteam.generate_scenes --config CFG --tasks tasks.json \\
        --out-suite SUITE_DIR

Each subprocess loads FLUX.2-klein (~8GB) and exits before the next stage, so
VRAM is reclaimed between tasks and before the world-model rollout. Per-task
failures are logged to ``<out_suite>/../scene_errors.json`` and skipped; the
process exits non-zero only if zero cases were produced.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

from openworld.redteam.config import load_redteam_config
from openworld.utils.io import load_yaml


def _run_multiview(
    script, prompt, output_dir, checkpoint_path, wrist_input, side_cond,
    num_inference_steps, env,
):
    """Spawn multiview_droid_with_nanobanana.py for a single task."""
    cmd = [
        sys.executable, str(script),
        "--prompt", prompt,
        "--output_dir", str(output_dir),
        "--checkpoint_path", str(checkpoint_path),
        "--wrist_input", str(wrist_input),
        "--num_inference_steps", str(num_inference_steps),
    ]
    for s in side_cond:
        cmd += ["--side_cond", str(s)]
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"multiview subprocess failed (exit {proc.returncode})")


def _build_case(
    case_dir, raw_dir, view_mapping, target_size, template_state, task,
    iter_idx, case_id,
):
    """Resize the multiview outputs and write one Initialization suite case."""
    case_dir.mkdir(parents=True, exist_ok=True)
    target_w, target_h = int(target_size[0]), int(target_size[1])

    for view_name, src_filename in view_mapping.items():
        src = raw_dir / src_filename
        if not src.exists():
            raise FileNotFoundError(f"multiview output missing: {src}")
        img = Image.open(src).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        img.save(case_dir / f"{view_name}.png")

    # `initial_observation` is intentionally omitted: InitializationDataset
    # infers it from the three view PNGs in the case directory.
    init = {
        "initial_state": template_state,
        "instruction": task.get("robot_instruction") or task.get("task_prompt", ""),
        "metadata": {
            "suite": "redteam",
            "iter": iter_idx,
            "case_id": case_id,
            "failure_mode": task.get("failure_mode", ""),
            "based_on": task.get("based_on", ""),
            "task_prompt": task.get("task_prompt", ""),
        },
    }
    with open(case_dir / "initialization.yaml", "w") as f:
        yaml.safe_dump(init, f, sort_keys=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Redteam stage B: scene generation")
    p.add_argument("--config", required=True, help="Redteam YAML config")
    p.add_argument("--tasks", required=True, help="tasks.json from stage A")
    p.add_argument("--out-suite", required=True, help="Output Initialization suite dir")
    args = p.parse_args()

    cfg = load_redteam_config(args.config)
    rt = cfg["redteam"]
    mv = rt["multiview"]

    # Fail fast if the Gemini API key is missing -- before any GPU time.
    key_env = mv.get("google_api_key_env", "GOOGLE_API_KEY")
    api_key = os.environ.get(key_env)
    if not api_key:
        print(
            f"[generate_scenes] {key_env} is not set; nanobanana cannot run",
            file=sys.stderr,
        )
        sys.exit(1)

    script = Path(mv["script"]).resolve()
    checkpoint_path = Path(mv["checkpoint_path"]).resolve()
    wrist_input = Path(mv["wrist_input"]).resolve()
    side_cond = [str(Path(s).resolve()) for s in mv["side_cond"]]
    num_inference_steps = int(mv.get("num_inference_steps", 50))
    view_mapping = rt["view_mapping"]
    target_size = rt["view_target_size"]

    template = load_yaml(rt["template_initialization"])
    template_state = template.get("initial_state")
    if template_state is None:
        print(
            "[generate_scenes] template_initialization has no 'initial_state' block",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks_data = json.loads(Path(args.tasks).read_text())
    iter_idx = tasks_data.get("iter", 0)
    tasks = tasks_data.get("tasks", [])

    out_suite = Path(args.out_suite)
    out_suite.mkdir(parents=True, exist_ok=True)
    raw_root = out_suite.parent / "raw"

    # The bundled diffusers script reads the env var named GOOGLE_API_KEY verbatim.
    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = api_key

    produced = 0
    errors = []
    for j, task in enumerate(tasks):
        case_id = f"iter{iter_idx}_task{j}"
        raw_dir = raw_root / case_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_multiview(
                script, task.get("task_prompt", ""), raw_dir, checkpoint_path,
                wrist_input, side_cond, num_inference_steps, env,
            )
            _build_case(
                out_suite / case_id, raw_dir, view_mapping, target_size,
                template_state, task, iter_idx, case_id,
            )
            produced += 1
            print(f"[generate_scenes] {case_id}: ok")
        except Exception as exc:  # noqa: BLE001 - isolate per-task failures
            print(f"[generate_scenes] {case_id}: FAILED ({exc})", file=sys.stderr)
            errors.append({
                "case_id": case_id,
                "task_prompt": task.get("task_prompt", ""),
                "error": str(exc),
            })

    if errors:
        err_path = out_suite.parent / "scene_errors.json"
        err_path.write_text(
            json.dumps({"iter": iter_idx, "errors": errors}, indent=2)
        )
        print(f"[generate_scenes] {len(errors)} task(s) failed -> {err_path}")

    print(
        f"[generate_scenes] iter {iter_idx}: produced {produced}/{len(tasks)} cases "
        f"-> {out_suite}"
    )
    if produced == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
