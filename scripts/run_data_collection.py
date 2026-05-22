"""Collect policy roll-out trajectories in the real LIBERO simulator and
store them in the world-model training-data format.

This rolls out an openpi policy (``pi05_libero`` by default) in the real
LIBERO env -- exactly as ``scripts/eval_libero_real_env.py`` does -- but,
instead of just reporting success, it records every executed frame + EEF
state and writes each trajectory in the same on-disk layout that
``scripts/preprocess_libero_for_wm.py`` produces from human demos:

    <output_root>/<suite>/annotation/<split>/<episode_id>.json
    <output_root>/<suite>/raw_videos/{agentview,wrist}/<episode_id>.mp4
    <output_root>/<suite>/latent_videos/{agentview,wrist}/<episode_id>.pt
    <output_root>/<suite>/{train,val}_sample.json

So collected data is directly consumable by ``LiberoLatentDataset`` (the WM
trainer) with no extra preprocessing.

Each annotation additionally records the source environment so the specific
benchmark + task can be recovered from any trajectory:

    "task_suite": "libero_goal",
    "task_id": 1,
    "task_name": "...",          # task.language
    "trial": 3,                  # which init state within the task
    "bddl": "<bddl stem>",
    "is_success": true,
    "episode_steps": 137,
    "policy_checkpoint": "gs://..."

Usage:
    python scripts/run_data_collection.py --config configs/collection/libero_pi05.yaml

Run inside the project env (the bash launcher sets MUJOCO_GL=egl etc.):
    bash bash_scripts/collect_libero.sh
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

# Reuse the exact on-disk writers used by the demo preprocessor so collected
# data is byte-format-identical to WM training data.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocess_libero_for_wm import (  # noqa: E402
    LatentEncoder,
    write_episode,
    write_sample_list,
)

from openworld.utils.io import load_yaml  # noqa: E402

logger = logging.getLogger(__name__)


# Per-suite episode caps (external/openpi/examples/libero/main.py:55-71).
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a robosuite (x, y, z, w) quaternion to axis-angle."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


def _ensure_openpi_paths(openpi_repo: str) -> None:
    repo = pathlib.Path(openpi_repo)
    for p in (
        str(repo),
        str(repo / "src"),
        str(repo / "packages" / "openpi-client" / "src"),
        str(repo / "third_party" / "libero"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    _enable_legacy_torch_load()


def _enable_legacy_torch_load() -> None:
    """Let LIBERO load its bundled init-state files under PyTorch >=2.6.

    LIBERO's ``benchmark.get_task_init_states`` does a bare
    ``torch.load(<task>.pruned_init)`` on files that are just pickled numpy
    arrays. PyTorch 2.6 flipped ``torch.load``'s default to
    ``weights_only=True``, which refuses to unpickle them. These files ship
    with LIBERO and are trusted, so restore the pre-2.6 behavior.
    """
    import torch

    if getattr(torch.load, "_libero_legacy", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._libero_legacy = True
    torch.load = _load


def _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list):
    """Append one env-rate sample (frame + EEF state) for both cameras.

    Frames are vertically flipped to undo MuJoCo's bottom-up offscreen
    render -- the same convention ``preprocess_libero_for_wm.py`` uses, so
    the SVD-VAE sees images in the orientation the WM was trained on. (The
    extra 180-deg rotation pi0 expects is applied only at the policy
    boundary, not here.)
    """
    agent_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]).copy())
    wrist_frames.append(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]).copy()
    )
    cart_list.append(
        np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]))
        ).astype(np.float32)
    )
    grip_list.append(
        float(np.mean(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)))
    )


def _next_episode_id(output_root: pathlib.Path, suite: str) -> int:
    """First unused integer episode id across both splits for a suite, so
    repeated runs append rather than clobber."""
    used = []
    for split in ("train", "val"):
        ann_dir = output_root / suite / "annotation" / split
        if ann_dir.is_dir():
            for p in ann_dir.glob("*.json"):
                if p.stem.isdigit():
                    used.append(int(p.stem))
    return (max(used) + 1) if used else 0


def _rollout_one_episode(
    *, policy, env, init_state, task_description, prompt, max_steps,
    num_steps_wait, replan_steps, resize_size, env_max_reward,
):
    """Run a single trajectory. Returns (payload-fields, is_success, steps)."""
    from openpi_client import image_tools

    env.reset()
    obs = env.set_init_state(init_state)
    action_plan: collections.deque = collections.deque()

    agent_frames, wrist_frames, cart_list, grip_list = [], [], [], []
    is_success = False
    executed = 0

    t = 0
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            # Let objects settle (they spawn slightly above the table).
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # Record the frame/state the policy is about to act from.
        _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list)

        if not action_plan:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, resize_size, resize_size)
            )
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": str(prompt if prompt is not None else task_description),
            }
            chunk = policy.infer(element)["actions"]
            action_plan.extend(chunk[:replan_steps])

        # pi0 emits an 8-D action (the model's action_dim is padded); the
        # LIBERO env's OSC_POSE controller wants 7 (6 EEF delta + 1 gripper).
        # Index 7 is padding -- drop it. See docs/LIBERO.md sec 1.2.
        action = action_plan.popleft()[:7]
        obs, reward, done, _ = env.step(
            action.tolist() if hasattr(action, "tolist") else list(action)
        )
        executed += 1
        if done or reward == env_max_reward:
            is_success = True
            break
        t += 1

    # Trailing frame so the last state has a matching image.
    _record_obs(obs, agent_frames, wrist_frames, cart_list, grip_list)

    payload = {
        "agent_rgb": np.stack(agent_frames, axis=0),
        "wrist_rgb": np.stack(wrist_frames, axis=0),
        "cart": np.stack(cart_list, axis=0),
        "grip": np.asarray(grip_list, dtype=np.float32),
    }
    return payload, is_success, executed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_root", type=str, default=None,
                    help="Override save.output_root from the config.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_yaml(args.config)

    pol_cfg = cfg.get("policy", {})
    col_cfg = cfg.get("collection", {})
    env_cfg = cfg.get("env", {})
    save_cfg = cfg.get("save", {})

    openpi_repo = pol_cfg.get("repo_path", "external/openpi")
    _ensure_openpi_paths(openpi_repo)

    output_root = pathlib.Path(args.output_root or save_cfg["output_root"])
    write_raw = bool(save_cfg.get("write_raw", True))
    encode_latents = bool(save_cfg.get("encode_latents", True))
    down_sample = max(1, int(save_cfg.get("down_sample", 1)))
    raw_fps = int(save_cfg.get("raw_fps", 20))
    out_fps = max(1, raw_fps // down_sample)
    val_fraction = float(save_cfg.get("val_fraction", 0.0))
    num_history = int(save_cfg.get("num_history", 6))
    num_frames = int(save_cfg.get("num_frames", 5))

    resolution = int(env_cfg.get("resolution", 256))
    num_steps_wait = int(env_cfg.get("num_steps_wait", 10))
    replan_steps = int(env_cfg.get("replan_steps", 5))
    resize_size = int(env_cfg.get("resize_size", 224))
    seed = int(env_cfg.get("seed", 7))
    env_max_reward = float(env_cfg.get("env_max_reward", 1.0))

    suites = col_cfg.get("task_suites") or col_cfg.get("task_suite")
    if isinstance(suites, str):
        suites = [suites]
    if not suites:
        raise SystemExit("config collection.task_suites must list >=1 suite")
    trajectories_per_task = int(col_cfg.get("trajectories_per_task", 10))
    task_id_filter = col_cfg.get("task_ids")  # None => all tasks in suite
    prompt_override = col_cfg.get("prompt_override")  # None => use task.language

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- policy (in-process pi0.5) -------------------------------------- #
    from openworld.policies.openpi_loader import load_policy_from_checkpoint
    checkpoint_path = pol_cfg["checkpoint_path"]
    logger.info("Loading policy %s from %s",
                pol_cfg.get("config_name", "pi05_libero"), checkpoint_path)
    policy = load_policy_from_checkpoint(
        config_name=pol_cfg.get("config_name", "pi05_libero"),
        checkpoint_path=checkpoint_path,
        repo_path=openpi_repo,
        default_prompt=None,
        pytorch_device=pol_cfg.get("pytorch_device", "cuda"),
    )

    # --- optional SVD-VAE latent encoder -------------------------------- #
    encoder = None
    if encode_latents:
        svd_path = save_cfg.get("svd_path")
        if not svd_path:
            raise SystemExit(
                "save.encode_latents is true but save.svd_path is unset.")
        encoder = LatentEncoder(svd_path, device=save_cfg.get("device", "cuda"))

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()

    for suite in suites:
        if suite not in SUITE_MAX_STEPS:
            raise SystemExit(f"unknown task_suite {suite!r}")
        max_steps = SUITE_MAX_STEPS[suite]
        task_suite = bench[suite]()
        task_ids = (
            list(task_id_filter) if task_id_filter is not None
            else list(range(task_suite.n_tasks))
        )

        episode_id = _next_episode_id(output_root, suite)
        train_ids: list[str] = []
        val_ids: list[str] = []
        logger.info("[%s] collecting %d traj over %d task(s); first eid=%06d",
                    suite, trajectories_per_task * len(task_ids),
                    len(task_ids), episode_id)

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            bddl_file = task.bddl_file
            if not pathlib.Path(bddl_file).exists():
                # Older LIBERO stores a relative bddl name.
                from libero.libero import get_libero_path
                bddl_file = str(
                    pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder / task.bddl_file
                )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=resolution,
                camera_widths=resolution,
            )
            env.seed(seed)
            task_description = task.language
            bddl_stem = pathlib.Path(task.bddl_file).stem

            n_ok = 0
            for trial in tqdm.tqdm(
                range(trajectories_per_task),
                desc=f"{suite} task{task_id:02d}",
            ):
                init_state = init_states[trial % len(init_states)]
                payload, is_success, steps = _rollout_one_episode(
                    policy=policy, env=env, init_state=init_state,
                    task_description=task_description, prompt=prompt_override,
                    max_steps=max_steps, num_steps_wait=num_steps_wait,
                    replan_steps=replan_steps, resize_size=resize_size,
                    env_max_reward=env_max_reward,
                )
                n_ok += int(is_success)

                # Temporal subsample to the WM rate before saving.
                if down_sample > 1:
                    payload = {
                        "agent_rgb": payload["agent_rgb"][::down_sample],
                        "wrist_rgb": payload["wrist_rgb"][::down_sample],
                        "cart": payload["cart"][::down_sample],
                        "grip": payload["grip"][::down_sample],
                    }
                payload["language"] = task_description
                payload["bddl"] = bddl_stem

                eid = f"{episode_id:06d}"
                split = "val" if rng.random() < val_fraction else "train"
                write_episode(
                    suite=suite,
                    split=split,
                    episode_id=eid,
                    output_root=output_root,
                    encoder=encoder,
                    payload=payload,
                    fps=out_fps,
                    write_raw=write_raw,
                    extra_annotation={
                        "task_id": int(task_id),
                        "task_name": task_description,
                        "trial": int(trial),
                        "is_success": bool(is_success),
                        "episode_steps": int(steps),
                        "num_steps_wait": int(num_steps_wait),
                        "policy_checkpoint": str(checkpoint_path),
                        "source": "policy_rollout",
                    },
                )
                (train_ids if split == "train" else val_ids).append(eid)
                episode_id += 1

            env.close()
            logger.info("[%s] task %d (%s): %d/%d success",
                        suite, task_id, task_description, n_ok,
                        trajectories_per_task)

        suite_root = output_root / suite
        # Rebuild the sample index over EVERY episode on disk for this suite
        # (not just this run's), so repeated collection runs accumulate
        # instead of clobbering the index. On-disk data is already at the WM
        # rate (pre-strided), so the sample list walks it with down_sample=1.
        for split in ("train", "val"):
            all_ids = sorted(
                p.stem
                for p in (suite_root / "annotation" / split).glob("*.json")
                if p.stem.isdigit()
            )
            if all_ids:
                write_sample_list(suite_root, split, all_ids,
                                  num_history=num_history, num_frames=num_frames,
                                  down_sample=1)
        logger.info("[%s] done this run: %d train + %d val trajectories -> %s",
                    suite, len(train_ids), len(val_ids), suite_root)


if __name__ == "__main__":
    main()
