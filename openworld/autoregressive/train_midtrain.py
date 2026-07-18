"""Mid-training entry point (omni-dreams stages L2a / L1b).

Plain flow-matching fine-tuning of ONE ARWorldModel from the base backbone:

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_studentinit_droid.py    # causal (L2a)
    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_teacher_droid.py         # bidirectional (L1b)

The two stages are independent (both start from the base model) so they run as
parallel jobs; ``train_self_forcing`` then loads both via ``student_init_ckpt`` /
``teacher_ckpt``. ``cfg.stage`` selects the attention pattern (see
``ARWMArgs.stage_is_causal``).

Reuses the self-forcing trainer's FSDP / checkpoint-retention / crash-safe-resume
/ sample-preview machinery; the loop itself is a single forward + MSE + step.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from .config import ARWMArgs
from .distill.midtrain import DiffusionTrainer
from .distill.scheduler import FlowMatchScheduler
from .model import ARWorldModel
from .train_self_forcing import (
    _RESUME_NAME,
    _CheckpointKeeper,
    _fsdp_shard_models,
    _gather_full_state_dict,
    _load_config,
    _SamplePreviewer,
    _WARMUP_STEPS,
)


# ---- single-model resume bundle (the SF version bundles two models) ----------
def _gather_one(model, opt, step, distributed):
    if not distributed:
        return {"model": model.state_dict(), "opt": opt.state_dict(), "global_step": step}
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
    m, o = get_state_dict(model, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {"model": m, "opt": o, "global_step": step}


def _restore_one(path, model, opt, distributed):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not distributed:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        return int(state["global_step"])
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["opt"],
                   options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return int(state["global_step"])


def _save_atomic(state, out_dir):
    import os
    out = Path(out_dir) / _RESUME_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, out)
    return out


def main(args: ARWMArgs) -> None:
    from accelerate import Accelerator
    from accelerate.logging import get_logger

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = get_logger(__name__)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, log_with="wandb", project_dir=args.output_dir,
    )
    causal = args.stage_is_causal
    model = ARWorldModel(args).to(accelerator.device, args.dtype)
    distributed = _fsdp_shard_models([model], enabled=args.use_fsdp)
    if accelerator.is_main_process:
        logger.info(f"stage={args.stage} (causal={causal})  FSDP {'ON' if distributed else 'off'}")

    sched = FlowMatchScheduler(args.denoising_step_list, num_train_timestep=args.num_train_timestep,
                               warp=args.warp_denoising_step)
    trainer = DiffusionTrainer(
        model, sched, frames_per_block=args.frames_per_block, causal=causal,
        lr=args.midtrain_lr, weight_decay=args.midtrain_weight_decay,
        betas=args.adam_betas, max_grad_norm=args.midtrain_grad_clip,
    )

    from openworld.autoregressive.data import ARLatentDataset
    loader = torch.utils.data.DataLoader(
        ARLatentDataset(args, split="train"), batch_size=args.train_batch_size,
        shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    loader = accelerator.prepare(loader)
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config={},
                                  init_kwargs={"wandb": {"name": args.wandb_run_name}})

    global_step = 0
    resume_path = args.resume_from or (Path(args.output_dir) / _RESUME_NAME)
    if args.save_resume_state and Path(resume_path).exists():
        global_step = _restore_one(resume_path, model, trainer.opt, distributed)
        logger.info(f"RESUMED from {resume_path} at step {global_step}")

    # Gradient accumulation: do `accum` micro-batches per optimizer step, so the
    # effective batch is train_batch_size * accum * world_size. global_step counts
    # optimizer steps (so max_train_steps / checkpointing are update-based). Grads
    # accumulate in `.grad` across micro-batches (FSDP reduce-scatters each one);
    # the 1/accum loss scale keeps the update identical to a true batch `accum`x
    # larger -- this model has no cross-sample ops, so it matches exactly.
    accum = max(1, args.gradient_accumulation_steps)
    loss_sum, micro = 0.0, 0

    keeper = _CheckpointKeeper(args.output_dir, args.checkpointing_steps, args.permanent_checkpoint_steps)
    previewer = _SamplePreviewer(args, is_main=accelerator.is_main_process)  # no-op if log_samples off
    t_start, t_warm, start_step = time.monotonic(), None, global_step
    logger.info(f"Starting mid-training: stage={args.stage}, {args.backbone}, "
                f"causal={causal}; rolling every {args.checkpointing_steps} steps, "
                f"permanent every {args.permanent_checkpoint_steps}")

    while global_step < args.max_train_steps:
        for batch in loader:
            latent = batch["latent"].to(accelerator.device, args.dtype)   # [B, F, C, H, W]
            actions = batch["action"].to(accelerator.device, args.dtype)
            cond = model.encode_cond(actions, texts=batch.get("text"), cfg_drop=True)
            loss_sum += trainer.forward_backward(latent, cond, loss_scale=1.0 / accum)
            micro += 1
            if micro < accum:
                continue                       # keep accumulating; no step yet
            gn = trainer.optimizer_step()
            logs = {"loss": loss_sum / accum, "grad_norm": gn}
            loss_sum, micro = 0.0, 0
            global_step += 1
            if global_step == start_step + _WARMUP_STEPS:
                t_warm = time.monotonic()
            if accelerator.is_main_process and global_step % 50 == 0:
                accelerator.log(logs, step=global_step)
                thru = ""
                if t_warm is not None and global_step > start_step + _WARMUP_STEPS:
                    sps = (time.monotonic() - t_warm) / (global_step - start_step - _WARMUP_STEPS)
                    thru = f" | {sps:.2f}s/step (~{round(3600 / sps)} steps/h)"
                logger.info(f"step {global_step}: {logs}{thru}")
            if keeper.due(global_step):
                sd = _gather_full_state_dict(model, distributed)
                if accelerator.is_main_process:
                    out, perm = keeper.save(sd, global_step)
                    logger.info(f"saved {'PERMANENT' if perm else 'rolling'} {out} "
                                f"(step {global_step}, {(time.monotonic() - t_start) / 3600:.2f}h)")
                if args.save_resume_state:
                    rs = _gather_one(model, trainer.opt, global_step, distributed)
                    if accelerator.is_main_process:
                        _save_atomic(rs, args.output_dir)
                previewer(model, global_step, accelerator=accelerator, logger=logger)
            if global_step >= args.max_train_steps:
                break

    sd = _gather_full_state_dict(model, distributed)
    if accelerator.is_main_process:
        out, _ = keeper.save(sd, global_step, force_permanent=True)
        logger.info(f"saved final PERMANENT {out} (step {global_step})")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    a = parser.parse_args()
    main(_load_config(a.config))


if __name__ == "__main__":
    cli()
