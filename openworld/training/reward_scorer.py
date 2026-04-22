"""Robometer reward scorer for RL training.

Runs Robometer as a **persistent subprocess server** to avoid two problems:

1. **Fork deadlock**: JAX is multithreaded, so ``os.fork()`` from the
   training process deadlocks.  By starting the server *before* JAX is
   imported (via ``start_server``), the fork happens in a single-threaded
   process.
2. **Cold-start latency**: The 4B model takes ~10 min to load.  A
   persistent server loads the model once and keeps it warm across all
   scoring calls.

Usage in the training entry-point::

    # Before importing JAX or building the world model:
    server_proc = RobometerRewardScorer.start_server(
        robometer_dir="external/robometer",
        model_path="robometer/Robometer-4B",
        gpu=1,
    )

    # ... import JAX, build models ...

    scorer = RobometerRewardScorer(server_proc=server_proc, fps=2.0)

The per-frame progress values are aggregated into per-chunk rewards by
``frames_to_chunk_rewards``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Resolve paths relative to the open-world repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROBOMETER_DIR = _REPO_ROOT / "external" / "robometer"
_SCORER_SCRIPT = _REPO_ROOT / "scripts" / "score_videos_robometer.py"


def _pipe_stderr(proc: subprocess.Popen) -> None:
    """Forward the server's stderr to the main process logger.

    Noisy lines (HuggingFace HTTP warnings, progress bars, ANSI-heavy
    debug output, etc.) are suppressed to keep the training log readable.
    """
    assert proc.stderr is not None
    # Substrings that indicate noisy/uninteresting log lines.
    _SUPPRESS = (
        "huggingface_hub",
        "thrown while requesting",
        "Retrying in",
        "Fetching",
        "WARNING]Retrying",
        "|WARNING]",
        "it/s]",  # tqdm progress bars
        "it/s]\r",
        "Loading pipeline",
        "computation placer already registered",
        "Unable to register cu",
        "TF_ENABLE_ONEDNN",
        "WARNING: All log messages before absl",
        "tensorflow/core",
        "FutureWarning",
        "UserWarning",
        "Please restructure your imports",
        "google/api_core",
        "torchao version",
        "Skipping import of cpp",
        "| RG:",  # per-layer weight info from setup_model_and_processor
        "setup_model_and_processor",
        "checkpoint shards",
        "Qwen",  # tokenizer/processor dumps
        "AddedToken",
        "processor_class",
        "image_processor",
        "video_processor",
        "| DEBUG",
        "compute_batch_outputs",
    )
    for line in proc.stderr:
        line = line.rstrip("\n")
        if not line:
            continue
        if any(s in line for s in _SUPPRESS):
            continue
        logger.info("[robometer-server] %s", line)


class RobometerRewardScorer:
    """Score world-model predicted frames using a persistent Robometer server.

    The server subprocess is started once via ``start_server()`` (which
    must be called **before JAX is imported** to avoid fork deadlock),
    then reused for all scoring calls.

    The server loads the model in the background.  Readiness is checked
    lazily on the first ``score_episodes`` call, so the model loads in
    parallel with the world model and policy on GPU 0.

    Args:
        server_proc: A running ``Popen`` object returned by ``start_server``.
        fps: FPS to sample from saved videos (Robometer parameter).
        work_dir: Directory for temporary files.
        timeout: Seconds to wait for a single scoring request.
        ready_timeout: Seconds to wait for initial model load on first use.
    """

    def __init__(
        self,
        server_proc: subprocess.Popen,
        fps: float = 2.0,
        work_dir: str | None = None,
        timeout: int = 120,
        ready_timeout: int = 900,
    ):
        self._proc = server_proc
        self.fps = fps
        self.work_dir = Path(work_dir) if work_dir else None
        self.timeout = timeout
        self._ready_timeout = ready_timeout
        self._ready = False

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def start_server(
        robometer_dir: str | None = None,
        model_path: str = "robometer/Robometer-4B",
        fps: float = 2.0,
        gpu: int | str | None = None,
    ) -> subprocess.Popen:
        """Launch the Robometer scoring server as a subprocess.

        **Must be called before JAX is imported** so that the fork
        happens in a single-threaded process.

        Returns immediately after spawning the process.  The model loads
        in the background; readiness is checked lazily on first use.

        Args:
            robometer_dir: Path to the robometer repo.
            model_path: HuggingFace model id or local path.
            fps: Default FPS for scoring.
            gpu: GPU index to pin the server to via CUDA_VISIBLE_DEVICES.

        Returns:
            A ``Popen`` object with stdin/stdout pipes for communication.
        """
        robometer_dir = str(
            Path(robometer_dir or _DEFAULT_ROBOMETER_DIR).resolve()
        )

        cmd = [
            "uv", "run", "--frozen", "python", str(_SCORER_SCRIPT),
            "--server",
            "--model-path", model_path,
            "--fps", str(fps),
        ]

        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Disable unsloth telemetry/stats which tries to download from
        # huggingface.co and fails on compute nodes without internet.
        env.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
        # Force offline mode so that transformers/HF Hub only read from
        # the local cache.  Without this, AutoConfig.from_pretrained
        # tries to contact HuggingFace, times out on offline compute
        # nodes, and silently returns None — causing unsloth to error
        # with "No config file found".
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(
            "Starting Robometer server (gpu=%s): cd %s && %s",
            gpu if gpu is not None else "inherit",
            robometer_dir,
            " ".join(cmd),
        )

        proc = subprocess.Popen(
            cmd,
            cwd=robometer_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Forward server stderr to logger in a background thread.
        stderr_thread = threading.Thread(
            target=_pipe_stderr, args=(proc,), daemon=True
        )
        stderr_thread.start()

        logger.info("Robometer server launched (model loading in background)")
        return proc

    def _wait_until_ready(self) -> None:
        """Block until the server signals readiness (called once lazily)."""
        if self._ready:
            return

        logger.info(
            "Waiting for Robometer server to finish loading model "
            "(timeout=%ds) ...", self._ready_timeout,
        )
        assert self._proc.stdout is not None
        try:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    # EOF — server process exited.
                    raise RuntimeError(
                        "Robometer server exited before signaling ready "
                        f"(returncode={self._proc.poll()})"
                    )
                line = line.strip()
                if not line:
                    continue
                # Skip non-JSON lines (library print() that escaped
                # the stdout redirect).
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Unexpected non-JSON on server stdout: %s", line[:200]
                    )
                    continue
                if msg.get("status") == "ready":
                    break
                if msg.get("status") == "error":
                    raise RuntimeError(
                        f"Robometer server error during init: {msg.get('error')}"
                    )
        except Exception:
            self._proc.kill()
            raise

        self._ready = True
        logger.info("Robometer server is ready")

    def shutdown(self) -> None:
        """Gracefully shut down the server."""
        if self._proc.poll() is not None:
            return
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(
                json.dumps({"command": "shutdown"}) + "\n"
            )
            self._proc.stdin.flush()
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_episodes(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score a batch of episodes.

        Each episode dict must contain:
            "id": str identifier
            "frames": np.ndarray of shape (T, H, W, C), uint8
            "instruction": str task description

        Returns a list of result dicts, one per episode, each containing:
            "id": episode id
            "per_frame_progress": list of floats, one per frame
            "success_probs": list of floats, one per frame
        """
        if not episodes:
            return []

        with tempfile.TemporaryDirectory(
            dir=self.work_dir, prefix="rl_reward_"
        ) as tmp_dir:
            tmp_path = Path(tmp_dir)
            return self._run_scoring(episodes, tmp_path)

    def _run_scoring(
        self,
        episodes: list[dict[str, Any]],
        tmp_dir: Path,
    ) -> list[dict[str, Any]]:
        """Save frames as videos, send request to server, parse results.

        On the first call this blocks until the server finishes loading
        the model (if it hasn't already).
        """
        self._wait_until_ready()
        manifest_entries = []

        for ep in episodes:
            ep_id = ep["id"]
            frames = ep["frames"]  # (T, H, W, C) uint8
            instruction = ep.get("instruction", "")

            video_path = tmp_dir / f"{ep_id}.mp4"
            self._save_frames_as_video(frames, str(video_path), fps=self.fps)

            manifest_entries.append({
                "id": ep_id,
                "video_path": str(video_path),
                "instruction": instruction,
            })

        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"episodes": manifest_entries}))

        output_path = tmp_dir / "rewards.json"

        # Send scoring request to the persistent server.
        request = {
            "manifest": str(manifest_path),
            "output": str(output_path),
            "fps": self.fps,
        }
        logger.info("Sending scoring request to Robometer server (%d episodes)",
                     len(episodes))

        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Robometer server has exited (returncode={self._proc.returncode})"
            )

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        # Wait for response.
        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError(
                "Robometer server closed stdout unexpectedly "
                f"(returncode={self._proc.poll()})"
            )

        response = json.loads(response_line.strip())
        if response.get("status") == "error":
            raise RuntimeError(
                f"Robometer server returned error: {response.get('error')}"
            )

        output_data = json.loads(output_path.read_text())
        return output_data.get("episodes", [])

    @staticmethod
    def _save_frames_as_video(
        frames: np.ndarray,
        path: str,
        fps: float = 2.0,
    ) -> None:
        """Save a (T, H, W, C) uint8 array as an MP4 video."""
        try:
            import imageio.v3 as iio

            iio.imwrite(path, frames, fps=fps, codec="libx264")
        except ImportError:
            import imageio

            writer = imageio.get_writer(path, fps=fps, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()


def frames_to_chunk_rewards(
    per_frame_progress: list[float],
    num_chunks: int,
) -> list[float]:
    """Distribute per-frame progress values into per-chunk rewards.

    The world model produces ``frames_per_chunk`` frames per action chunk
    (typically 5 frames for 5 actions with action_downsample=1).  This
    function splits the progress values across chunks and computes the
    reward for each chunk.

    The reward for each chunk is the **change in progress** (delta) within
    that chunk, so that the total reward across chunks approximates the
    final progress minus the initial progress.

    Args:
        per_frame_progress: List of per-frame progress values (monotonically
            increasing from 0 to 1 for successful episodes).
        num_chunks: Number of action chunks in the episode.

    Returns:
        List of per-chunk reward values.
    """
    n_frames = len(per_frame_progress)
    if n_frames == 0 or num_chunks == 0:
        return [0.0] * max(num_chunks, 1)

    frames_per_chunk = max(n_frames // num_chunks, 1)
    chunk_rewards = []

    for i in range(num_chunks):
        start = i * frames_per_chunk
        end = min((i + 1) * frames_per_chunk, n_frames)
        if start >= n_frames:
            chunk_rewards.append(0.0)
            continue

        chunk_start_progress = per_frame_progress[start]
        chunk_end_progress = per_frame_progress[min(end, n_frames) - 1]
        # Delta reward: how much progress was made in this chunk
        chunk_rewards.append(chunk_end_progress - chunk_start_progress)

    return chunk_rewards
