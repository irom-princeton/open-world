#!/usr/bin/env python3
"""End-to-end utility: edit instructions -> annotated rollout videos.

Given a file of scene-edit instructions and an output directory, this:

  1. [scenes]  nanobanana wrist edit + multiview model build an Initialization
               suite from each edit instruction (``openworld.redteam.generate_scenes``).
  2. [rollout] the policy is rolled out inside the world model from each
               generated initialization, with a VLM judging the latest frame of
               every interaction (``scripts/generate_videos.py``).
  3. [charts]  two annotated videos are written per episode:
                 - annotated_chart/  : the rollout's vertical view stack with
                                       the VLM reward line chart animated below.
                 - annotated_paired/ : the left side view + wrist view arranged
                                       horizontally, with the same animated chart
                                       below the two views.

The heavy stages (FLUX multiview, then the world-model rollout) run in isolated
subprocesses, exactly as ``bash_scripts/run_custom_tasks.sh`` does, so GPU memory
is reclaimed between them.

Usage:
    python scripts/run_edit_tasks.py EDIT_INSTRUCTIONS OUTPUT_DIR [--config CFG]

The instructions file may be any of:
  * ``{"iter": 0, "tasks": [{"task_prompt": ..., "robot_instruction": ...}, ...]}``
  * a bare JSON list of such task dicts
  * a bare JSON list of strings (each treated as a ``task_prompt``)

``robot_instruction`` (the language command for the policy and VLM) defaults to
``task_prompt`` when omitted. See ``configs/redteam/custom_tasks.example.json``.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from openworld.utils.annotated_video import annotate_from_manifest
from openworld.utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs/redteam/example_redteam.yaml"


def _normalize_tasks(instructions_path: Path) -> dict:
    """Coerce the various accepted input shapes into a tasks.json dict."""
    data = json.loads(instructions_path.read_text())

    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
        iter_idx = data.get("iter", 0)
    elif isinstance(data, list):
        tasks = data
        iter_idx = 0
    else:
        raise ValueError(
            f"{instructions_path}: expected a dict with 'tasks' or a JSON list, "
            f"got {type(data).__name__}"
        )

    norm = []
    for i, task in enumerate(tasks):
        if isinstance(task, str):
            task = {"task_prompt": task}
        elif not isinstance(task, dict):
            raise ValueError(f"task {i} must be a string or object, got {type(task).__name__}")
        if not task.get("task_prompt"):
            raise ValueError(f"task {i} is missing a non-empty 'task_prompt'")
        task.setdefault("robot_instruction", task["task_prompt"])
        norm.append(task)

    if not norm:
        raise ValueError(f"{instructions_path}: no tasks found")
    return {"iter": iter_idx, "tasks": norm}


def _run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output; raise on failure."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("instructions", help="File of edit instructions (see module docstring)")
    parser.add_argument("output_dir", help="Directory for the suite, videos, and charts")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Redteam YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["exterior_left", "wrist"],
        help="Views placed side-by-side in the paired video (default: left side + wrist).",
    )
    parser.add_argument(
        "--chart-height", type=int, default=220, help="Pixel height of the reward chart strip."
    )
    args = parser.parse_args()

    instructions_path = Path(args.instructions).resolve()
    if not instructions_path.exists():
        raise SystemExit(f"instructions file not found: {instructions_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_dir = out_dir / "suite"
    eval_cfg_path = out_dir / "eval_cfg.yaml"
    manifest_path = out_dir / "manifest.json"
    rewards_path = out_dir / "rewards.json"

    # --- normalize the edit instructions into the tasks.json schema ---
    tasks = _normalize_tasks(instructions_path)
    tasks_path = out_dir / "tasks.json"
    tasks_path.write_text(json.dumps(tasks, indent=2))
    logger.info("=" * 60)
    logger.info("edit-task run")
    logger.info("  config:       %s", args.config)
    logger.info("  instructions: %s  (%d tasks)", instructions_path, len(tasks["tasks"]))
    logger.info("  output_dir:   %s", out_dir)
    logger.info("=" * 60)

    # --- the line chart is the VLM reward, so the eval must score inline ---
    cfg = load_yaml(args.config)
    rm_name = cfg.get("reward_model", {}).get("name")
    if rm_name != "vlm":
        raise SystemExit(
            f"reward_model.name is '{rm_name}' in {args.config}, but this utility "
            "produces a VLM reward line chart. Set reward_model.name: vlm (with a "
            "backend + api_key_env) in the config."
        )

    # --- [scenes] nanobanana + multiview -> Initialization suite ---
    _run([
        sys.executable, "-m", "openworld.redteam.generate_scenes",
        "--config", args.config,
        "--tasks", str(tasks_path),
        "--out-suite", str(suite_dir),
    ])

    # --- [rollout] strip the redteam block, inject paths, roll out + score ---
    _run([
        sys.executable, "-m", "openworld.redteam.config",
        "--config", args.config, "--emit-eval-config",
        "--suite", str(suite_dir),
        "--video-dir", str(out_dir),
        "--out", str(eval_cfg_path),
    ])
    _run([
        sys.executable, str(REPO_ROOT / "scripts/generate_videos.py"),
        "--config", str(eval_cfg_path),
        "--manifest-output", str(manifest_path),
        "--rewards-output", str(rewards_path),
    ])

    if not manifest_path.exists():
        raise SystemExit(f"rollout produced no manifest: {manifest_path}")

    # --- [charts] vertical stack + chart, and left+wrist horizontal + chart ---
    stacked = annotate_from_manifest(
        out_dir,
        manifest_path=manifest_path,
        rewards_path=rewards_path,
        output_subdir="annotated_chart",
        layout="stacked",
        chart_height=args.chart_height,
    )
    paired = annotate_from_manifest(
        out_dir,
        manifest_path=manifest_path,
        rewards_path=rewards_path,
        output_subdir="annotated_paired",
        layout="paired",
        views=tuple(args.views),
        chart_height=args.chart_height,
    )

    logger.info("")
    logger.info("===== done =====")
    logger.info("suite (generated scenes): %s", suite_dir)
    logger.info("rollout videos:           %s", out_dir / "videos")
    logger.info("VLM rewards:              %s", rewards_path)
    logger.info("stacked + chart:          %s  (%d videos)", out_dir / "annotated_chart", len(stacked))
    logger.info(
        "%s + chart:    %s  (%d videos)",
        "+".join(args.views), out_dir / "annotated_paired", len(paired),
    )


if __name__ == "__main__":
    main()
