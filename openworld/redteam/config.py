"""Redteam config loading + small CLI helpers used by bash_scripts/redteam.sh.

The bash orchestrator stays minimal by shelling out to this module for the
three things it needs:
  --print-key KEY        scalar lookup (dotted path)
  --count-tasks PATH     number of tasks in a tasks.json
  --emit-eval-config ... strip the `redteam` block and inject dataset_path /
                         video_dir, producing a config that generate_videos.py
                         can consume directly.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from openworld.utils.io import load_yaml

REQUIRED_REDTEAM_KEYS = [
    "meta_objective",
    "tasks_per_iteration",
    "run_dir_root",
    "llm",
    "multiview",
    "template_initialization",
    "view_target_size",
    "view_mapping",
]


def load_redteam_config(path: str) -> dict:
    """Load and validate a redteam YAML config.

    Returns the full config dict (the `redteam` block plus the standard
    evaluation blocks). Raises ValueError on a missing/invalid block.
    """
    cfg = load_yaml(str(path))
    if not isinstance(cfg, dict) or "redteam" not in cfg:
        raise ValueError(f"{path}: missing top-level 'redteam' block")
    rt = cfg["redteam"]
    if not isinstance(rt, dict):
        raise ValueError(f"{path}: 'redteam' block must be a mapping")
    missing = [k for k in REQUIRED_REDTEAM_KEYS if k not in rt]
    if missing:
        raise ValueError(f"{path}: redteam block missing keys: {missing}")
    for block in ("world_model", "policy"):
        if block not in cfg:
            raise ValueError(f"{path}: missing top-level '{block}' block")
    return cfg


def _get_dotted(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _emit_eval_config(cfg: dict, suite: str, video_dir: str, out: str) -> None:
    """Write an evaluation config for generate_videos.py.

    Drops the `redteam` block and injects absolute `dataset_path` (the suite
    directory) and `video_dir`.
    """
    eval_cfg = {k: v for k, v in cfg.items() if k != "redteam"}
    eval_cfg["dataset_path"] = str(Path(suite).resolve())
    eval_cfg["video_dir"] = str(Path(video_dir).resolve())
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(eval_cfg, f, sort_keys=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Redteam config helper")
    p.add_argument("--config", default=None, help="Path to the redteam YAML config")
    p.add_argument("--print-key", default=None,
                   help="Dotted key to print, e.g. redteam.run_dir_root")
    p.add_argument("--count-tasks", default=None,
                   help="Path to a tasks.json; prints the number of tasks")
    p.add_argument("--emit-eval-config", action="store_true",
                   help="Emit an eval config (requires --suite/--video-dir/--out)")
    p.add_argument("--suite", default=None)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # --count-tasks does not depend on the redteam config being valid (or present).
    if args.count_tasks is not None:
        try:
            data = json.loads(Path(args.count_tasks).read_text())
            tasks = data.get("tasks", []) if isinstance(data, dict) else []
        except (OSError, json.JSONDecodeError):
            tasks = []
        print(len(tasks))
        return

    if args.config is None:
        p.error("--config is required for --print-key / --emit-eval-config")

    cfg = load_redteam_config(args.config)

    if args.print_key is not None:
        try:
            val = _get_dotted(cfg, args.print_key)
        except KeyError:
            print(f"redteam config: no such key '{args.print_key}'", file=sys.stderr)
            sys.exit(1)
        print(val)
        return

    if args.emit_eval_config:
        if not (args.suite and args.video_dir and args.out):
            p.error("--emit-eval-config requires --suite, --video-dir, and --out")
        _emit_eval_config(cfg, args.suite, args.video_dir, args.out)
        return

    p.error("no action requested (--print-key / --count-tasks / --emit-eval-config)")


if __name__ == "__main__":
    main()
