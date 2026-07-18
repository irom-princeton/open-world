#!/usr/bin/env python3

import argparse
import json
import logging
import os
from pathlib import Path

from openworld.datasets import Initialization, InitializationDataset
from openworld.envs import ActionChunkScheduler, WorldModelEnv
from openworld.policies import build_policy
from openworld.runners import Evaluator
from openworld.utils.io import load_yaml
from openworld.world_models import build_world_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_dataset_path(config_path: str, dataset_path: str) -> str:
    dataset = Path(dataset_path)
    if dataset.is_absolute():
        return str(dataset)
    if dataset.exists():
        return str(dataset.resolve())
    return str((Path(config_path).resolve().parent / dataset).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rollout videos")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest-output", type=str, required=True,
                        help="Path to write the episode manifest JSON")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_path = Path(args.config).resolve()

    # Resolve all relative I/O paths to absolute *before* building the world
    # model. Some adapters (e.g. WEAVER) os.chdir() into their vendored repo
    # during load_checkpoint for repo-relative assets, which would otherwise
    # silently re-root any later relative path (dataset, video_dir, checkpoints,
    # the openpi repo, the action adapter, ...).
    if cfg.get("dataset_path"):
        cfg["dataset_path"] = _resolve_dataset_path(str(config_path), cfg["dataset_path"])
    if cfg.get("video_dir"):
        cfg["video_dir"] = str(Path(cfg["video_dir"]).resolve())

    def _abs_if_local(value):
        """Make a relative path absolute iff it points at an existing local file/dir.
        Leaves non-paths (config names, gs:// URLs, HF repo ids, already-absolute
        or non-existent strings) untouched."""
        if isinstance(value, str) and value and not os.path.isabs(value) and os.path.exists(value):
            return str(Path(value).resolve())
        return value

    # Checkpoints + every known path-valued param, resolved up front so a later
    # adapter chdir can't break them.
    _PARAM_PATH_KEYS = (
        "config_path", "stats_root", "vae_dir", "svd_model_path", "clip_model_path",
        "norm_stats_path", "weaver_repo", "repo_path", "action_adapter_checkpoint_path",
    )
    for _sec in ("world_model", "policy"):
        sec = cfg.get(_sec)
        if not isinstance(sec, dict):
            continue
        if "checkpoint_path" in sec:
            sec["checkpoint_path"] = _abs_if_local(sec["checkpoint_path"])
        params = sec.get("params")
        if isinstance(params, dict):
            for k in _PARAM_PATH_KEYS:
                if k in params:
                    params[k] = _abs_if_local(params[k])

    # Build world model
    wm_cfg = cfg.get("world_model", {})
    world_model = build_world_model(wm_cfg.get("name", "dummy"), **wm_cfg.get("params", {}))
    if wm_cfg.get("checkpoint_path"):
        world_model.load_checkpoint(wm_cfg["checkpoint_path"])

    scheduler = ActionChunkScheduler(
        chunk_size=cfg.get("scheduler", {}).get("chunk_size", 15)
    )
    env = WorldModelEnv(world_model=world_model, action_chunk_scheduler=scheduler)

    # Build policy
    pol_cfg = cfg.get("policy", {})
    policy = build_policy(pol_cfg.get("name", "dp"), **pol_cfg.get("params", {}))
    if pol_cfg.get("checkpoint_path"):
        policy.load_checkpoint(pol_cfg["checkpoint_path"])

    # Build dataset
    dataset_path = cfg.get("dataset_path")
    dataset_entries = cfg.get("dataset", [])
    if dataset_path:
        dataset = InitializationDataset.from_yaml(
            _resolve_dataset_path(str(config_path), dataset_path)
        )
    elif dataset_entries:
        dataset = InitializationDataset.from_list(dataset_entries)
    else:
        dataset = InitializationDataset([
            Initialization(
                id="dummy_init_0",
                initial_state=None,
                initial_observation=None,
                instruction="pick up the cup",
            )
        ])

    # Optional init subset: run only the listed init ids (folder names).
    init_ids = cfg.get("init_ids")
    if init_ids:
        wanted = set(init_ids)
        selected = [init for init in dataset if init.id in wanted]
        found = {init.id for init in selected}
        missing = wanted - found
        if missing:
            raise ValueError(f"init_ids not found in dataset: {sorted(missing)}")
        dataset = InitializationDataset(selected)
        logger.info("Running init subset (%d): %s", len(selected), sorted(found))

    evaluator = Evaluator(env=env, policy=policy)

    chunk_size = cfg.get("scheduler", {}).get("chunk_size", 15)
    action_hz = cfg.get("action_hz", 15)
    num_frames = wm_cfg.get("params", {}).get("num_frames", 5)

    if "duration" in cfg:
        duration = cfg["duration"]
        max_steps = int(duration * action_hz)
    else:
        max_steps = cfg.get("max_steps", 50)
        duration = max_steps / action_hz

    video_fps = int(num_frames * action_hz / chunk_size)
    video_dir = cfg.get("video_dir")
    if video_dir:
        video_dir = str(Path(video_dir).resolve() / "videos")

    logger.info(
        "Video generation: world_model=%s policy=%s "
        "duration=%.1fs max_steps=%d video_fps=%d",
        wm_cfg.get("name", "dummy"),
        pol_cfg.get("name", "dp"),
        duration,
        max_steps,
        video_fps,
    )

    per_view = bool(cfg.get("per_view_videos", False))
    view_order = wm_cfg.get("params", {}).get("view_order")

    results = evaluator.run_dataset(
        dataset, max_steps=max_steps, video_dir=video_dir, video_fps=video_fps,
        per_view=per_view, view_order=tuple(view_order) if view_order else None,
    )

    logger.info("Video generation complete: %d episodes → %s", len(results), video_dir)

    # Write manifest
    episodes = []
    for r in results:
        ep = {
            "id": r["initialization_id"],
            "instruction": r.get("instruction", ""),
            "num_frames": len(r["frames"]),
            "metadata": r.get("metadata", {}),
        }
        if video_dir:
            if per_view and r.get("view_video_paths"):
                ep["view_video_paths"] = [str(Path(p).resolve()) for p in r["view_video_paths"]]
            else:
                ep["video_path"] = str(Path(video_dir).resolve() / f"{r['initialization_id']}.mp4")
        episodes.append(ep)

    manifest = {"episodes": episodes}
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_output).write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest written to %s", args.manifest_output)


if __name__ == "__main__":
    main()
