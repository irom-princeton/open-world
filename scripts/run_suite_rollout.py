#!/usr/bin/env python3
"""Roll out the world model + policy on an existing Initialization suite.

Unlike ``scripts/run_edit_tasks.py``, this skips scene generation entirely and
takes a pre-built suite directory as input. For each initialization it:

  1. [rollout] rolls the policy out inside the world model, with a VLM judging
               the latest frame of every interaction (``scripts/generate_videos.py``).
  2. [charts]  writes, per episode:
                 - videos/<id>.mp4          : the raw world-model rollout (vertical
                                              view stack), produced by the rollout.
                 - annotated_paired/<id>.mp4 : two chosen views side-by-side with
                                              the VLM reward line chart below.

A suite directory is one whose child dirs each hold
``{wrist,exterior_left,exterior_right}.png`` + an ``initialization.yaml`` -- exactly
the layout produced by ``run_edit_tasks.py`` / ``generate_scenes.py``.

Usage:
    python scripts/run_suite_rollout.py SUITE_DIR [OUTPUT_DIR] [--config CFG]
        [--views exterior_left wrist] [--chart-height 220] [--stacked]

OUTPUT_DIR defaults to the suite's parent directory.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from openworld.utils.annotated_video import annotate_from_manifest
from openworld.utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs/redteam/example_redteam.yaml"


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
    parser.add_argument("suite", help="Existing Initialization suite directory")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory for videos, manifest, and charts (default: the suite's parent)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Redteam/eval YAML config (default: {DEFAULT_CONFIG})",
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
    parser.add_argument(
        "--stacked",
        action="store_true",
        help="Also write the vertical view-stack + chart (annotated_chart/).",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite).resolve()
    if not suite_dir.is_dir():
        raise SystemExit(f"suite directory not found: {suite_dir}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else suite_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg_path = out_dir / "eval_cfg.yaml"
    manifest_path = out_dir / "manifest.json"
    rewards_path = out_dir / "rewards.json"

    logger.info("=" * 60)
    logger.info("suite rollout")
    logger.info("  config:     %s", args.config)
    logger.info("  suite:      %s", suite_dir)
    logger.info("  output_dir: %s", out_dir)
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

    # --- [rollout] strip the redteam block, inject paths, roll out + score ---
    # Same path as run_edit_tasks.py, just pointed at an existing suite. The two
    # heavy stages run as isolated subprocesses so GPU memory is reclaimed.
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

    # --- [charts] two-view + reward chart (and optionally the vertical stack) ---
    paired = annotate_from_manifest(
        out_dir,
        manifest_path=manifest_path,
        rewards_path=rewards_path,
        output_subdir="annotated_paired",
        layout="paired",
        views=tuple(args.views),
        chart_height=args.chart_height,
    )
    stacked = []
    if args.stacked:
        stacked = annotate_from_manifest(
            out_dir,
            manifest_path=manifest_path,
            rewards_path=rewards_path,
            output_subdir="annotated_chart",
            layout="stacked",
            chart_height=args.chart_height,
        )

    logger.info("")
    logger.info("===== done =====")
    logger.info("rollout videos:        %s", out_dir / "videos")
    logger.info("VLM rewards:           %s", rewards_path)
    logger.info(
        "%s + chart:  %s  (%d videos)",
        "+".join(args.views), out_dir / "annotated_paired", len(paired),
    )
    if args.stacked:
        logger.info("stacked + chart:       %s  (%d videos)", out_dir / "annotated_chart", len(stacked))


if __name__ == "__main__":
    main()
