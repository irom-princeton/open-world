import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from openworld.utils.io import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _generate_videos(cfg: dict) -> list[dict]:
    """Phase 1: Run video generation in a subprocess.

    By running in a separate process, all GPU memory is reclaimed by the OS
    when the subprocess exits — guaranteeing a clean slate for Phase 2.
    """
    video_dir = cfg.get("video_dir")
    if not video_dir:
        logger.error("video_dir must be set in the config for two-phase evaluation")
        return []

    manifest_path = Path(video_dir).resolve() / "manifest.json"

    # Write the resolved config (with CLI overrides applied) to a temp file
    # so the subprocess sees the same values.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="eval_cfg_", delete=False
    ) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_config_path = tmp.name

    try:
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent / "generate_videos.py"),
            "--config", tmp_config_path,
            "--manifest-output", str(manifest_path),
        ]

        logger.info("Phase 1 — Spawning video generation subprocess ...")
        logger.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    finally:
        Path(tmp_config_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        logger.error("Video generation subprocess failed (exit code %d)", proc.returncode)
        return []

    if not manifest_path.exists():
        logger.error("Expected manifest not found: %s", manifest_path)
        return []

    manifest = json.loads(manifest_path.read_text())
    return manifest.get("episodes", [])


def _score_with_robometer(
    episodes: list[dict],
    rm_params: dict,
    video_dir: str,
) -> list[dict]:
    """Phase 2: Score saved videos with Robometer via subprocess."""

    robometer_dir = Path("external/robometer").resolve()
    score_script = Path("scripts/score_videos_robometer.py").resolve()

    if not robometer_dir.exists():
        logger.error(
            "Robometer not found at %s. Clone it first:\n"
            "  git clone https://github.com/robometer/robometer.git external/robometer",
            robometer_dir,
        )
        return []

    # Write manifest for the scoring script.
    manifest_path = Path(video_dir).resolve() / "manifest.json"
    rewards_path = Path(video_dir).resolve() / "rewards.json"
    manifest_path.write_text(json.dumps({"episodes": episodes}, indent=2, default=str))

    model_path = rm_params.get("model_path", "robometer/Robometer-4B")
    fps = rm_params.get("fps", 2.0)

    cmd = [
        "uv", "run", "python", str(score_script),
        "--manifest", str(manifest_path),
        "--model-path", model_path,
        "--output", str(rewards_path),
        "--fps", str(fps),
    ]

    logger.info("Phase 2 — Reward scoring with Robometer (model=%s) ...", model_path)
    logger.info("Running: cd %s && %s", robometer_dir, " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=str(robometer_dir),
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if proc.returncode != 0:
        logger.error("Robometer scoring failed (exit code %d)", proc.returncode)
        return []

    if not rewards_path.exists():
        logger.error("Expected rewards file not found: %s", rewards_path)
        return []

    rewards_data = json.loads(rewards_path.read_text())
    return rewards_data.get("episodes", [])


_VIEW_NAMES = ["view_0", "view_1", "view_2"]


def _create_annotated_videos(
    episodes: list[dict],
    reward_episodes: list[dict],
    video_dir: str,
    success_threshold: float = 0.5,
) -> None:
    """Create copies of evaluation videos with per-view and avg scores overlaid."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        import imageio.v3 as iio
    except ImportError:
        logger.error("imageio is required for annotated videos; skipping")
        return

    reward_by_id = {r["id"]: r for r in reward_episodes}
    annotated_dir = Path(video_dir).resolve() / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for ep in episodes:
        ep_id = ep["id"]
        video_path = ep.get("video_path")
        if not video_path or not Path(video_path).exists():
            continue

        ri = reward_by_id.get(ep_id, {})
        if "error" in ri:
            continue

        avg_progress = ri.get("per_frame_progress", [])
        avg_success = ri.get("success_probs", [])
        if not avg_progress:
            continue

        # Collect per-view data.
        view_data = []
        for vn in _VIEW_NAMES:
            vd = ri.get(vn, {})
            view_data.append({
                "progress": vd.get("per_frame_progress", []),
                "success": vd.get("success_probs", []),
            })

        # Read original video frames and metadata.
        frames = iio.imread(video_path)  # (T, H, W, 3)
        meta = iio.immeta(video_path)
        video_fps = meta.get("fps", 5)

        T = frames.shape[0]
        n_scores = len(avg_progress)

        annotated_frames = []
        for i in range(T):
            score_idx = min(int(i * n_scores / T), n_scores - 1)

            avg_prog = avg_progress[score_idx]
            avg_succ = avg_success[score_idx] if score_idx < len(avg_success) else 0.0

            frame = Image.fromarray(frames[i])
            draw = ImageDraw.Draw(frame)
            W, H = frame.size
            margin = 4

            # Per-view annotations at the top of each view section.
            view_width = W // 3
            for vi, vn in enumerate(_VIEW_NAMES):
                vp = view_data[vi]["progress"]
                vs = view_data[vi]["success"]
                p = vp[score_idx] if score_idx < len(vp) else 0.0
                s = vs[score_idx] if score_idx < len(vs) else 0.0
                vtext = f"{vn}: prog={p:.2f} succ={s:.2f}"
                vbbox = draw.textbbox((0, 0), vtext, font=font)
                vtw, vth = vbbox[2] - vbbox[0], vbbox[3] - vbbox[1]
                vx = vi * view_width + margin
                vy = margin
                draw.rectangle(
                    [vx, vy, vx + vtw + 2 * margin, vy + vth + 2 * margin],
                    fill=(0, 0, 0, 180),
                )
                draw.text((vx + margin, vy + margin), vtext, fill=(255, 255, 255), font=font)

            # Average annotation at the very bottom.
            avg_text = f"avg: prog={avg_prog:.2f} succ={avg_succ:.2f}"
            avg_bbox = draw.textbbox((0, 0), avg_text, font=font)
            atw, ath = avg_bbox[2] - avg_bbox[0], avg_bbox[3] - avg_bbox[1]
            ax = margin
            ay = H - ath - 3 * margin
            draw.rectangle(
                [ax, ay, ax + atw + 2 * margin, ay + ath + 2 * margin],
                fill=(0, 0, 0, 180),
            )
            draw.text((ax + margin, ay + margin), avg_text, fill=(255, 255, 255), font=font)

            annotated_frames.append(np.array(frame))

        out_path = str(annotated_dir / f"{ep_id}.mp4")
        iio.imwrite(out_path, np.stack(annotated_frames), fps=video_fps)
        logger.info("Annotated video saved: %s", out_path)


def _cleanup_temp_files(video_dir: str) -> None:
    """Remove intermediate files (manifest, rewards JSON)."""
    for name in ("manifest.json", "rewards.json"):
        path = Path(video_dir).resolve() / name
        if path.exists():
            path.unlink()
            logger.info("Removed %s", path)


def _print_reward_summary(
    episodes: list[dict],
    reward_episodes: list[dict],
    success_threshold: float = 0.5,
) -> None:
    """Merge reward results with episode metadata and print summary."""

    reward_by_id = {r["id"]: r for r in reward_episodes}

    # Per-task aggregation (keyed by instruction, each entry has avg + per-view rewards).
    task_results: dict[str, list[dict]] = {}
    for ep in episodes:
        ep_id = ep["id"]
        instruction = ep.get("instruction") or ep.get("metadata", {}).get("task_type", "unknown")
        ri = reward_by_id.get(ep_id, {})

        if "error" in ri:
            logger.warning("Episode %s: scoring error: %s", ep_id, ri["error"])
            continue

        progress = ri.get("per_frame_progress", [])
        success_probs = ri.get("success_probs", [])
        reward = float(progress[-1]) if progress else 0.0
        success = bool(success_probs and max(success_probs) >= success_threshold)

        # Per-view rewards.
        view_rewards = {}
        for vn in _VIEW_NAMES:
            vd = ri.get(vn, {})
            vp = vd.get("per_frame_progress", [])
            vs = vd.get("success_probs", [])
            view_rewards[vn] = {
                "reward": float(vp[-1]) if vp else 0.0,
                "success": bool(vs and max(vs) >= success_threshold),
            }

        view_str = "  ".join(
            f"{vn}={view_rewards[vn]['reward']:.4f}" for vn in _VIEW_NAMES
        )
        logger.info(
            "Episode %s  avg_reward=%.4f  success=%s  [%s]  instruction=%s",
            ep_id, reward, success, view_str, instruction,
        )

        task_results.setdefault(instruction, []).append({
            "reward": reward,
            "success": success,
            **{vn: view_rewards[vn] for vn in _VIEW_NAMES},
        })

    logger.info("=" * 60)
    logger.info("REWARD SUMMARY (per-view + average)")
    logger.info("=" * 60)

    all_rewards = []
    all_successes = []
    all_view_rewards: dict[str, list[float]] = {vn: [] for vn in _VIEW_NAMES}
    all_view_successes: dict[str, list[bool]] = {vn: [] for vn in _VIEW_NAMES}

    for task, infos in sorted(task_results.items()):
        rewards = [i["reward"] for i in infos]
        successes = [i["success"] for i in infos]
        n = len(rewards)
        mean_reward = sum(rewards) / n if n else 0.0
        success_rate = sum(successes) / n if n else 0.0

        view_summary_parts = []
        for vn in _VIEW_NAMES:
            vr = [i[vn]["reward"] for i in infos]
            vs = [i[vn]["success"] for i in infos]
            mr = sum(vr) / n if n else 0.0
            sr = sum(vs) / n if n else 0.0
            view_summary_parts.append(f"{vn}={mr:.4f}/{sr:.2f}")
            all_view_rewards[vn].extend(vr)
            all_view_successes[vn].extend(vs)

        logger.info(
            "  Task: %s  |  n=%d  avg_reward=%.4f  success_rate=%.2f  [%s]",
            task, n, mean_reward, success_rate, "  ".join(view_summary_parts),
        )

        all_rewards.extend(rewards)
        all_successes.extend(successes)

    total = len(all_rewards)
    if total:
        logger.info("-" * 60)
        view_overall = []
        for vn in _VIEW_NAMES:
            mr = sum(all_view_rewards[vn]) / total
            sr = sum(all_view_successes[vn]) / total
            view_overall.append(f"{vn}={mr:.4f}/{sr:.2f}")
        logger.info(
            "  OVERALL  |  n=%d  avg_reward=%.4f  success_rate=%.2f  [%s]",
            total,
            sum(all_rewards) / total,
            sum(all_successes) / total,
            "  ".join(view_overall),
        )
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run world-model evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to evaluation YAML config"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Override dataset_path in config"
    )
    parser.add_argument(
        "--video_dir", type=str, default=None, help="Override video_dir in config"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override world_model.checkpoint_path in config",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Override duration (seconds) in config",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.dataset_path is not None:
        cfg["dataset_path"] = args.dataset_path
    if args.video_dir is not None:
        cfg["video_dir"] = args.video_dir
    if args.checkpoint is not None:
        cfg.setdefault("world_model", {})["checkpoint_path"] = args.checkpoint
    if args.duration is not None:
        cfg["duration"] = args.duration
    # Phase 1: generate videos
    video_dir = cfg.get("video_dir")
    episodes = _generate_videos(cfg)

    # Phase 2: score with reward model (if configured and videos were saved)
    rm_cfg = cfg.get("reward_model", {})
    rm_name = rm_cfg.get("name", "dummy")
    rm_params = rm_cfg.get("params", {})

    if rm_name == "robometer" and video_dir and episodes:
        reward_episodes = _score_with_robometer(episodes, rm_params, video_dir)
        if reward_episodes:
            _create_annotated_videos(
                episodes, reward_episodes, video_dir,
                success_threshold=rm_params.get("success_threshold", 0.5),
            )
            _print_reward_summary(
                episodes,
                reward_episodes,
                success_threshold=rm_params.get("success_threshold", 0.5),
            )
        _cleanup_temp_files(video_dir)
    elif rm_name != "dummy":
        logger.warning(
            "Reward model '%s' is not supported in two-phase mode. "
            "Only 'robometer' and 'dummy' are supported.",
            rm_name,
        )
    else:
        logger.info("Evaluation complete: %d episodes (no reward scoring)", len(episodes))


if __name__ == "__main__":
    main()
