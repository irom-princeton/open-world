"""Entry point for RL fine-tuning of Pi0/Pi0.5 in a world-model environment.

Usage:
    python scripts/run_rl_finetune.py --config configs/rl/rl_finetune_pi0_droid.yaml

See configs/rl/rl_finetune_pi0_droid.yaml for configuration reference.
"""

import argparse
import logging

# =====================================================================
# IMPORTANT: Start the Robometer server BEFORE importing JAX.
#
# JAX spawns threads on import, and subprocess.Popen uses fork() which
# deadlocks in a multithreaded process.  By launching the server here
# (before any JAX import), the fork happens safely.  The server loads
# the reward model on its own GPU in the background while we proceed
# to load the world model and policy below.
# =====================================================================
from openworld.utils.io import load_yaml
from openworld.training.reward_scorer import RobometerRewardScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Suppress verbose loggers from dependencies
for _noisy in ["jax", "flax", "orbax", "openpi", "absl", "tensorflow",
               "diffusers", "transformers", "jax._src", "jax.interpreters"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted key=value overrides to a nested config dict.

    Example: ``--set reward_model.gpu=0`` sets ``cfg["reward_model"]["gpu"] = 0``.
    Values are parsed as YAML scalars (so ``0`` becomes int, ``true`` becomes bool, etc.).
    """
    import yaml

    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = yaml.safe_load(value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="RL fine-tuning with world model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to RL YAML config"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help="Override config values (dotted paths), e.g. --set reward_model.gpu=0",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.overrides:
        _apply_overrides(cfg, args.overrides)

    # --- Start Robometer server (before JAX import) ---
    rm_cfg = cfg.get("reward_model", {})
    logger.info("Launching Robometer server (loads model in background) ...")
    server_proc = RobometerRewardScorer.start_server(
        robometer_dir=rm_cfg.get("robometer_dir"),
        model_path=rm_cfg.get("model_path", "robometer/Robometer-4B"),
        fps=rm_cfg.get("fps", 2.0),
        gpu=rm_cfg.get("gpu"),
    )

    # --- Now safe to import JAX ---
    import numpy as np
    np.set_printoptions(threshold=20, edgeitems=2)
    import jax
    jax.numpy.set_printoptions(threshold=20, edgeitems=2)

    from openworld.envs import ActionChunkScheduler, WorldModelEnv
    from openworld.datasets.initialization_dataset import InitializationDataset
    from openworld.runners.rl_finetune_runner import RLFineTuneRunner
    from openworld.training.openpi_trainable import (
        TrainablePi0Config,
        freeze_base_model,
        load_trainable_pi0,
    )
    from openworld.training.ppo import PPOConfig
    from openworld.world_models import build_world_model

    # --- Build world model environment ---
    wm_cfg = cfg.get("world_model", {})
    world_model = build_world_model(
        wm_cfg.get("name", "dummy"), **wm_cfg.get("params", {})
    )
    if wm_cfg.get("checkpoint_path"):
        world_model.load_checkpoint(wm_cfg["checkpoint_path"])

    chunk_size = cfg.get("scheduler", {}).get("chunk_size", 15)
    scheduler = ActionChunkScheduler(chunk_size=chunk_size)
    env = WorldModelEnv(world_model=world_model, action_chunk_scheduler=scheduler)

    # --- Load trainable policy ---
    pol_cfg = cfg.get("policy", {})
    trainable_cfg_dict = pol_cfg.get("trainable_params", {})
    trainable_config = TrainablePi0Config(**trainable_cfg_dict) if trainable_cfg_dict else None

    # Suppress stdout/stderr during model loading — Flax NNX prints
    # full parameter arrays during model construction.
    import io, sys, contextlib
    _devnull = io.StringIO()
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        model = load_trainable_pi0(
            config_name=pol_cfg.get("config_name", "pi0_droid"),
            checkpoint_path=pol_cfg["checkpoint_path"],
            trainable_config=trainable_config,
            repo_path=pol_cfg.get("repo_path"),
        )
        freeze_base_model(model)
    logger.info("Policy loaded: %s", model)

    # --- Build reward scorer (wraps the already-running server) ---
    reward_scorer = RobometerRewardScorer(
        server_proc=server_proc,
        fps=rm_cfg.get("fps", 2.0),
        work_dir=rm_cfg.get("work_dir"),
        timeout=rm_cfg.get("timeout", 120),
        ready_timeout=rm_cfg.get("ready_timeout", 900),
    )

    # --- Load initialization dataset ---
    data_cfg = cfg.get("dataset", {})
    init_dataset = InitializationDataset.from_yaml(data_cfg["path"])

    # --- Build PPO config ---
    ppo_cfg_dict = cfg.get("ppo", {})
    ppo_config = PPOConfig(**ppo_cfg_dict)

    # --- Build and run trainer ---
    train_cfg = cfg.get("training", {})
    wandb_cfg = cfg.get("wandb", {})
    adapter_cfg = cfg.get("action_adapter", pol_cfg.get("action_adapter", {}))
    # RLFineTuneRunner creates PPOTrainer which calls nnx.split + optimizer.init
    # — these can trigger verbose Flax output.
    _devnull2 = io.StringIO()
    with contextlib.redirect_stdout(_devnull2), contextlib.redirect_stderr(_devnull2):
        runner = RLFineTuneRunner(
            env=env,
            policy=model,
            reward_scorer=reward_scorer,
            init_dataset=init_dataset,
            ppo_config=ppo_config,
            max_chunks_per_episode=train_cfg.get("max_chunks_per_episode", 10),
            episodes_per_iter=train_cfg.get("episodes_per_iter", 4),
            num_iterations=train_cfg.get("num_iterations", 100),
            checkpoint_dir=train_cfg.get("checkpoint_dir"),
            checkpoint_every=train_cfg.get("checkpoint_every", 10),
            log_every=train_cfg.get("log_every", 1),
            video_dir=train_cfg.get("video_dir") or cfg.get("video_dir"),
            wandb_project=wandb_cfg.get("project"),
            wandb_entity=wandb_cfg.get("entity"),
            wandb_run_name=wandb_cfg.get("run_name"),
            wandb_config=cfg,
            rng_seed=train_cfg.get("rng_seed", 42),
            action_adapter_checkpoint_path=adapter_cfg.get("checkpoint_path"),
            action_adapter_gripper_max=adapter_cfg.get("gripper_max", 1.0),
            action_adapter_device=adapter_cfg.get("device"),
            policy_skip_step=adapter_cfg.get("policy_skip_step", 1),
            num_action_steps=adapter_cfg.get("num_action_steps"),
        )

    # --- Load action normalization stats ---
    # Pi0 outputs actions in z-score normalized space.  We need to
    # denormalize before passing to the world model.
    norm_stats_cfg = pol_cfg.get("norm_stats")
    if norm_stats_cfg:
        # Direct stats in config
        runner.set_action_norm_stats(
            mean=np.array(norm_stats_cfg["action_mean"]),
            std=np.array(norm_stats_cfg["action_std"]),
        )
    else:
        # Load from openpi checkpoint assets
        norm_stats_path = pol_cfg.get("norm_stats_path")
        if norm_stats_path is None:
            # Default: try to load from the openpi checkpoint's assets dir.
            # The checkpoint_path may be a GCS URL — resolve it through
            # openpi's cache so we find the already-downloaded copy.
            import json
            from pathlib import Path
            ckpt_path = pol_cfg["checkpoint_path"]

            from openworld.policies.openpi_loader import ensure_openpi_repo_on_path
            ensure_openpi_repo_on_path(pol_cfg.get("repo_path"))
            import openpi.shared.download as _download

            try:
                local_ckpt = _download.maybe_download(ckpt_path)
            except Exception:
                local_ckpt = Path(ckpt_path)

            candidates = [
                Path(local_ckpt) / "assets" / "droid" / "norm_stats.json",
                Path(ckpt_path) / "assets" / "droid" / "norm_stats.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    norm_stats_path = str(candidate)
                    break
            if norm_stats_path is None:
                # Try downloading from GCS
                gcs_path = ckpt_path.rstrip("/") + "/assets/droid/norm_stats.json"
                import subprocess as _sp
                local_path = Path("/tmp/openpi_norm_stats.json")
                try:
                    _sp.run(
                        ["gsutil", "cp", gcs_path, str(local_path)],
                        check=True, capture_output=True, timeout=60,
                    )
                    norm_stats_path = str(local_path)
                except Exception as e:
                    logger.warning("Could not load norm stats from %s: %s", gcs_path, e)

        if norm_stats_path:
            import json
            from pathlib import Path
            raw = json.loads(Path(norm_stats_path).read_text())
            stats = raw.get("norm_stats", raw)
            action_stats = stats["actions"]
            runner.set_action_norm_stats(
                mean=np.array(action_stats["mean"]),
                std=np.array(action_stats["std"]),
            )
            # State normalization — the eval path applies Normalize(norm_stats)
            # which z-score normalizes the state input before the model sees it.
            # Without this, the model receives raw joint angles instead of the
            # near-zero normalized values it was trained on.
            if "state" in stats:
                state_stats = stats["state"]
                runner.set_state_norm_stats(
                    mean=np.array(state_stats["mean"]),
                    std=np.array(state_stats["std"]),
                )
            else:
                logger.warning(
                    "No state norm stats found — state will NOT be normalized. "
                    "Policy conditioning on state will be incorrect."
                )
        else:
            logger.warning(
                "No action norm stats found — actions will NOT be denormalized. "
                "World model outputs will likely be blurry/incorrect."
            )

    logger.info("Starting RL fine-tuning...")
    try:
        runner.train()
    finally:
        reward_scorer.shutdown()


if __name__ == "__main__":
    main()
