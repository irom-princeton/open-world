#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

from openworld.datasets import Initialization, InitializationDataset
from openworld.envs import ActionChunkScheduler, WorldModelEnv
from openworld.policies import build_policy
from openworld.rewards import build_reward_model
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
    parser.add_argument("--rewards-output", type=str, default=None,
                        help="When the reward_model is an inline frame judge "
                             "(e.g. name: vlm), write a robometer-shaped "
                             "rewards.json here so no separate scoring stage is needed.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_path = Path(args.config).resolve()

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

    # Optional inline frame judge. When the reward model scores individual frames
    # (e.g. a VLM), it is queried after every world-model interaction during the
    # rollout rather than as a separate post-hoc stage.
    frame_scorer = None
    rm_cfg = cfg.get("reward_model", {})
    if rm_cfg.get("name") == "vlm":
        reward_model = build_reward_model("vlm", **rm_cfg.get("params", {}))
        frame_scorer = reward_model.score_frame
        logger.info("Inline VLM frame judge enabled (model=%s)",
                    rm_cfg.get("params", {}).get("model_id", "gemini-2.5-flash"))

    evaluator = Evaluator(env=env, policy=policy, frame_scorer=frame_scorer)

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

    results = evaluator.run_dataset(
        dataset, max_steps=max_steps, video_dir=video_dir, video_fps=video_fps,
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
        if "vlm_scores" in r:
            ep["vlm_scores"] = r["vlm_scores"]
            ep["vlm_score_max"] = r["vlm_score_max"]
        if video_dir:
            ep["video_path"] = str(Path(video_dir).resolve() / f"{r['initialization_id']}.mp4")
        episodes.append(ep)

    manifest = {"episodes": episodes}
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_output).write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest written to %s", args.manifest_output)

    # When scoring happened inline (VLM frame judge), emit a robometer-shaped
    # rewards.json so the downstream history-merge stage is unchanged.
    if args.rewards_output and frame_scorer is not None:
        reward_episodes = []
        for r in results:
            scores = r.get("vlm_scores", [])
            reward_episodes.append({
                "id": r["initialization_id"],
                "vlm_scores": scores,
                "vlm_score_max": r.get("vlm_score_max"),
                # also expose as per_frame_progress for inspection / generic readers
                "per_frame_progress": scores,
            })
        Path(args.rewards_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.rewards_output).write_text(
            json.dumps({"episodes": reward_episodes}, indent=2, default=str)
        )
        logger.info("VLM rewards written to %s", args.rewards_output)


if __name__ == "__main__":
    main()
