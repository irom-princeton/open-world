#!/usr/bin/env python3
"""
Score generated videos using local Robometer checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def load_video_frames(video_path: str, fps: float = 2.0) -> np.ndarray:
    """Load frames from a video file. Returns uint8 (T, H, W, C)."""
    import decord

    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    try:
        native_fps = float(vr.get_avg_fps())
    except Exception:
        native_fps = fps

    desired = max(1, min(int(round(total * (fps / native_fps))), total))
    if desired == total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, desired, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    del vr
    return frames


VIEW_NAMES = ["view_0", "view_1", "view_2"]


def _score_frames(
    frames: np.ndarray,
    instruction: str,
    model,
    tokenizer,
    batch_collator,
    exp_config,
    device: torch.device,
) -> dict:
    """Score a single (T, H, W, C) frame array and return progress/success info."""
    T = int(frames.shape[0])

    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=instruction,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    results = compute_batch_outputs(
        model,
        tokenizer,
        progress_inputs,
        sample_type="progress",
        is_discrete_mode=is_discrete,
        num_bins=num_bins,
    )

    progress_pred = results.get("progress_pred", [])
    progress = (
        [float(x) for x in progress_pred[0]]
        if progress_pred and len(progress_pred) > 0
        else []
    )

    outputs_success = results.get("outputs_success", {})
    success_probs_raw = (
        outputs_success.get("success_probs", []) if outputs_success else []
    )
    success_probs = (
        [float(x) for x in success_probs_raw[0]]
        if success_probs_raw and len(success_probs_raw) > 0
        else []
    )

    return {
        "per_frame_progress": progress,
        "success_probs": success_probs,
    }


def score_episode(
    video_path: str,
    instruction: str,
    model,
    tokenizer,
    batch_collator,
    exp_config,
    device: torch.device,
    fps: float = 2.0,
) -> dict:
    """Score a single video by splitting into 3 views and scoring each.

    The generated videos have 3 camera views concatenated horizontally.
    Robometer expects a single view, so we split the width into 3 equal
    parts, score each independently, and return per-view results plus
    the average across views.
    """
    frames = load_video_frames(video_path, fps=fps)  # (T, H, W, C)
    W = frames.shape[2]
    view_width = W // 3

    view_frames = [
        frames[:, :, i * view_width : (i + 1) * view_width, :]
        for i in range(3)
    ]

    per_view_results = {}
    all_progress = []
    all_success = []

    for i, vf in enumerate(view_frames):
        view_name = VIEW_NAMES[i]
        result = _score_frames(
            frames=vf,
            instruction=instruction,
            model=model,
            tokenizer=tokenizer,
            batch_collator=batch_collator,
            exp_config=exp_config,
            device=device,
        )
        per_view_results[view_name] = result
        if result["per_frame_progress"]:
            all_progress.append(result["per_frame_progress"])
        if result["success_probs"]:
            all_success.append(result["success_probs"])

    # Compute average across views (element-wise per frame).
    if all_progress:
        avg_progress = [
            float(np.mean([p[j] for p in all_progress]))
            for j in range(len(all_progress[0]))
        ]
    else:
        avg_progress = []

    if all_success:
        avg_success = [
            float(np.mean([s[j] for s in all_success]))
            for j in range(len(all_success[0]))
        ]
    else:
        avg_success = []

    return {
        "per_frame_progress": avg_progress,
        "success_probs": avg_success,
        **{view_name: per_view_results[view_name] for view_name in VIEW_NAMES},
    }


def _score_manifest(args_manifest, args_model_path, args_output, args_fps,
                    model=None, tokenizer=None, batch_collator=None,
                    exp_config=None, device=None):
    """Score all episodes in a manifest file.

    If model/tokenizer/etc. are None, they will be loaded fresh.
    """
    manifest = json.loads(Path(args_manifest).read_text())
    episodes = manifest["episodes"]

    if not episodes:
        Path(args_output).write_text(json.dumps({"episodes": []}))
        return

    if model is None:
        print(f"Loading Robometer model from {args_model_path} ...",
              file=sys.stderr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=args_model_path, device=device
        )
        model.eval()
        batch_collator = setup_batch_collator(
            processor, tokenizer, exp_config, is_eval=True
        )
        print(f"Model loaded on {device}", file=sys.stderr)

    results = []
    for i, ep in enumerate(episodes):
        video_path = ep["video_path"]
        instruction = ep.get("instruction", "")
        ep_id = ep["id"]

        if not Path(video_path).exists():
            print(f"  [{i+1}/{len(episodes)}] SKIP {ep_id} "
                  f"(video not found: {video_path})", file=sys.stderr)
            results.append({"id": ep_id, "error": "video not found"})
            continue

        print(f"  [{i+1}/{len(episodes)}] Scoring {ep_id} ...",
              file=sys.stderr)
        try:
            info = score_episode(
                video_path=video_path,
                instruction=instruction,
                model=model,
                tokenizer=tokenizer,
                batch_collator=batch_collator,
                exp_config=exp_config,
                device=device,
                fps=args_fps,
            )
            info["id"] = ep_id
            results.append(info)
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            results.append({"id": ep_id, "error": str(exc)})

    output_data = {"episodes": results}
    Path(args_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_output).write_text(json.dumps(output_data, indent=2))
    print(f"Rewards written to {args_output}", file=sys.stderr)


def run_server(model_path: str, fps: float) -> None:
    """Run as a persistent server: load model once, then process requests
    from stdin.

    Protocol (line-based JSON over stdin/stdout):
      1. Server loads model, prints ``{"status": "ready"}`` to stdout.
      2. Parent sends a JSON line: ``{"manifest": "<path>", "output": "<path>"}``
      3. Server scores all episodes, writes results to the output path,
         then prints ``{"status": "done"}`` to stdout.
      4. Repeat from step 2.  Send ``{"command": "shutdown"}`` to exit.

    stdout is reserved for protocol messages only.  During model loading
    and scoring, stdout is temporarily redirected to stderr so that
    library ``print()`` calls don't pollute the JSON channel.
    """
    # Keep a reference to the real stdout for protocol messages.
    _real_stdout = sys.stdout

    # Redirect stdout → stderr during model loading so that any
    # library print() calls (transformers, unsloth, etc.) don't
    # pollute the JSON protocol on stdout.
    sys.stdout = sys.stderr

    import os as _os
    print(f"Loading Robometer model from {model_path} ...")
    print(f"  HF_HOME={_os.environ.get('HF_HOME', '<unset>')}")
    print(f"  HF_HUB_CACHE={_os.environ.get('HF_HUB_CACHE', '<unset>')}")
    print(f"  HF_HUB_OFFLINE={_os.environ.get('HF_HUB_OFFLINE', '<unset>')}")
    print(f"  TRANSFORMERS_OFFLINE={_os.environ.get('TRANSFORMERS_OFFLINE', '<unset>')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(
        processor, tokenizer, exp_config, is_eval=True
    )
    print(f"Robometer server ready on {device}")

    # Restore stdout for protocol, then signal readiness.
    sys.stdout = _real_stdout
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(
                json.dumps({"status": "error", "error": str(e)}) + "\n"
            )
            sys.stdout.flush()
            continue

        if request.get("command") == "shutdown":
            break

        manifest_path = request.get("manifest", "")
        output_path = request.get("output", "")
        req_fps = request.get("fps", fps)

        # Redirect stdout → stderr during scoring too.
        sys.stdout = sys.stderr
        try:
            _score_manifest(
                manifest_path, model_path, output_path, req_fps,
                model=reward_model, tokenizer=tokenizer,
                batch_collator=batch_collator, exp_config=exp_config,
                device=device,
            )
            sys.stdout = _real_stdout
            sys.stdout.write(json.dumps({"status": "done"}) + "\n")
        except Exception as exc:
            sys.stdout = _real_stdout
            sys.stdout.write(
                json.dumps({"status": "error", "error": str(exc)}) + "\n"
            )
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score videos with Robometer")
    parser.add_argument(
        "--manifest", required=False,
        help="Path to manifest JSON from run_evaluation.py",
    )
    parser.add_argument(
        "--model-path",
        default="robometer/Robometer-4B",
        help="HuggingFace model id or local checkpoint (default: robometer/Robometer-4B)",
    )
    parser.add_argument(
        "--output", required=False, help="Path to write rewards JSON"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS to sample from videos (default: 2.0)",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as a persistent server (stdin/stdout JSON protocol)",
    )
    args = parser.parse_args()

    if args.server:
        run_server(args.model_path, args.fps)
        return

    if not args.manifest or not args.output:
        parser.error("--manifest and --output are required in non-server mode")

    _score_manifest(args.manifest, args.model_path, args.output, args.fps)


if __name__ == "__main__":
    main()
