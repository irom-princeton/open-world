"""Sanity-check: run pi0.5_libero in a real LIBERO simulator (not the WM).

This is the "ground-truth" baseline that the WM-based evaluation is meant
to approximate. It's a minimal port of
``external/openpi/examples/libero/main.py`` that uses the openpi local
in-process policy + the LIBERO env, with optional support for routing
through the LIBERO action adapter (so the comparison against WM rollouts
is apples-to-apples).

Usage:
    python scripts/eval_libero_real_env.py \\
        --task_suite libero_spatial \\
        --num_trials_per_task 10 \\
        --pi05_checkpoint gs://openpi-assets/checkpoints/pi05_libero \\
        --video_out_path outputs/libero_real_env

The world model is *not* used here. For WM-based eval, see
``scripts/run_evaluation.py --config configs/evaluation/libero_pi05.yaml``.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import pathlib
import sys

import numpy as np
import tqdm
from PIL import Image

logger = logging.getLogger(__name__)


SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat):
    """Convert robosuite (x,y,z,w) quaternion to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


def _ensure_openpi_paths(openpi_repo: str) -> None:
    if openpi_repo not in sys.path:
        sys.path.insert(0, openpi_repo)
        sys.path.insert(0, str(pathlib.Path(openpi_repo) / "src"))
        sys.path.insert(0, str(pathlib.Path(openpi_repo) / "packages" / "openpi-client" / "src"))
        sys.path.insert(0, str(pathlib.Path(openpi_repo) / "third_party" / "libero"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi_repo", type=str, default="external/openpi")
    ap.add_argument("--task_suite", type=str, required=True,
                    choices=list(SUITE_MAX_STEPS.keys()))
    ap.add_argument("--num_trials_per_task", type=int, default=10)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--resize_size", type=int, default=224)
    ap.add_argument("--replan_steps", type=int, default=5)
    ap.add_argument("--pi05_checkpoint", type=str, required=True,
                    help="Path or gs:// URL to a pi0.5_libero checkpoint.")
    ap.add_argument("--video_out_path", type=str, default="outputs/libero_real_env")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    _ensure_openpi_paths(args.openpi_repo)

    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    # In-process pi0.5 policy (no websocket).
    from openworld.policies.openpi_loader import load_policy_from_checkpoint
    policy = load_policy_from_checkpoint(
        config_name="pi05_libero",
        checkpoint_path=args.pi05_checkpoint,
        repo_path=args.openpi_repo,
        default_prompt=None,
        pytorch_device="cuda",
    )

    bench = benchmark.get_benchmark_dict()
    suite = bench[args.task_suite]()
    max_steps = SUITE_MAX_STEPS[args.task_suite]

    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    LIBERO_ENV_RES = 256

    total, hits = 0, 0
    for task_id in tqdm.tqdm(range(suite.n_tasks), desc=f"{args.task_suite}"):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": LIBERO_ENV_RES,
            "camera_widths": LIBERO_ENV_RES,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)
        task_description = task.language

        for trial in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(init_states[trial])

            t = 0
            replay = []
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                replay.append(img)

                if not action_plan:
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    chunk = policy.infer(element)["actions"]
                    action_plan.extend(chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist() if hasattr(action, "tolist") else list(action))
                if done:
                    hits += 1
                    break
                t += 1
            total += 1

            if replay:
                Image.fromarray(np.concatenate(replay[::4], axis=1)).save(
                    pathlib.Path(args.video_out_path) / f"task{task_id:02d}_trial{trial:02d}.png"
                )

        env.close()

    print(f"\nReal-env success: {hits}/{total} = {hits / max(1, total):.3f}")


if __name__ == "__main__":
    main()
