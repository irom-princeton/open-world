"""LIBERO world-model training entry point.

This is a near-verbatim port of
``Fast-Control-World/scripts/train_fast_wm.py`` adapted for the LIBERO data
layout. It instantiates the same ``CrtlWorld`` model (from the
Fast-Control-World repo) so all the diffusion / flow-matching / shortcut
machinery is shared.

Usage (single-GPU):

    python -m openworld.training.world_model.train_wm \\
        --config configs/training/libero_wm.py

Usage (multi-GPU):

    accelerate launch -m openworld.training.world_model.train_wm \\
        --config configs/training/libero_wm.py

The config can be either:
* A path to a Python file that defines ``get_args() -> LiberoWMArgs``
  (preferred, mirrors the Fast-Control-World convention).
* Omitted, in which case the defaults from
  :class:`openworld.training.world_model.config.LiberoWMArgs` are used.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import math
import os
import sys
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from openworld.training.world_model.config import LiberoWMArgs
from openworld.training.world_model.dataset import LiberoLatentDataset


# ---------------------------------------------------------------------------
# Vendor the Fast-Control-World model code so we don't reimplement diffusion.
# ---------------------------------------------------------------------------

FCW_DEFAULT_PATH = os.environ.get(
    "FCW_PATH", "/n/fs/iromdata/project/Fast-Control-World"
)


def _import_crtl_world(fcw_path: str):
    if fcw_path not in sys.path:
        sys.path.insert(0, fcw_path)
    from models.flow_map_ctrl_world import CrtlWorld  # noqa: E402
    from models.pipeline_flow_map_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402

    return CrtlWorld, CtrlWorldDiffusionPipeline


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(config_arg: str | None) -> LiberoWMArgs:
    if config_arg is None:
        return LiberoWMArgs()
    path = Path(config_arg)
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Config not found: {config_arg}")
    spec = importlib.util.spec_from_file_location("user_libero_wm_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_args"):
        raise AttributeError(f"{path} must define get_args() -> LiberoWMArgs")
    return mod.get_args()


# ---------------------------------------------------------------------------
# Validation: render a few prediction videos.
# ---------------------------------------------------------------------------


def validate_video_generation(
    model,
    val_dataset,
    args: LiberoWMArgs,
    train_steps: int,
    videos_dir: str,
    sample_id: int,
    accelerator: Accelerator,
    pipeline_cls,
    inference_steps: int | None = None,
) -> None:
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = max(1, args.video_num)
    videos_col = 2

    # Sample evenly from val.
    indices = list(range(0, len(val_dataset), max(1, len(val_dataset) // (videos_row * videos_col))))
    indices = indices[sample_id * videos_col : (sample_id + 1) * videos_col]
    if not indices:
        return
    batch_list = [val_dataset[i] for i in indices]
    video_gt = torch.cat([t["latent"].unsqueeze(0) for t in batch_list], dim=0).to(device)
    text = [t["text"] for t in batch_list]
    actions = torch.cat([t["action"].unsqueeze(0) for t in batch_list], dim=0).to(device)

    his_latent_gt = video_gt[:, : args.num_history]
    future_latent_gt = video_gt[:, args.num_history :]
    current_latent = future_latent_gt[:, 0]

    expected_latent_shape = (4, args.latent_h_total, args.latent_w)
    if current_latent.shape[1:] != expected_latent_shape:
        raise ValueError(
            f"Latent shape mismatch: got {tuple(current_latent.shape[1:])}, expected {expected_latent_shape}."
            " Check num_cams / height / width vs the preprocessed dataset."
        )
    if actions.shape[1:] != (args.num_history + args.num_frames, args.action_dim):
        raise ValueError(
            f"Action shape mismatch: got {tuple(actions.shape[1:])}, expected "
            f"({args.num_history + args.num_frames}, {args.action_dim})"
        )

    with torch.no_grad():
        unwrapped = model.module if accelerator.num_processes > 1 else model
        action_latent = unwrapped.action_encoder(
            actions, text, unwrapped.tokenizer, unwrapped.text_encoder, args.frame_level_cond
        )

        _, pred_latents = pipeline_cls.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(args.num_cams * args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=inference_steps or args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
            flow_map_type=args.flow_map_type,
            flow_map_loss_type=args.flow_map_loss_type,
        )

    # Split the stacked-camera latents back into a per-camera batch dim, decode each.
    pred_latents = einops.rearrange(
        pred_latents, "b f c (m h) (n w) -> (b m n) f c h w", m=args.num_cams, n=1
    )
    full_gt = torch.cat([his_latent_gt, future_latent_gt], dim=1)
    full_gt = einops.rearrange(
        full_gt, "b f c (m h) (n w) -> (b m n) f c h w", m=args.num_cams, n=1
    )

    def _decode(latents: torch.Tensor) -> np.ndarray:
        bsz, frame_num = latents.shape[:2]
        flat = latents.flatten(0, 1)
        chunks = []
        for i in range(0, flat.shape[0], args.decode_chunk_size):
            chunk = flat[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            chunks.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        decoded = torch.cat(chunks, dim=0).reshape(bsz, frame_num, *chunks[0].shape[1:])
        decoded = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).to(pipeline.unet.dtype)
        return decoded.detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

    video_gt_arr = _decode(full_gt)
    pred_arr = _decode(pred_latents)

    pred_arr = np.concatenate([video_gt_arr[:, : args.num_history], pred_arr], axis=1)
    side_by_side = np.concatenate([video_gt_arr, pred_arr], axis=-3)
    grid = np.concatenate([v for v in side_by_side], axis=-2).astype(np.uint8)

    out_dir = Path(videos_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"train_steps_{train_steps}_{sample_id}.mp4"
    mediapy.write_video(str(fname), grid, fps=2)

    try:
        import wandb
        if wandb.run is not None:
            steps_tag = Path(videos_dir).name
            accelerator.log(
                {f"val_video/{steps_tag}/sample_{sample_id}": wandb.Video(str(fname), fps=2, format="mp4")},
                step=train_steps,
            )
    except Exception as e:
        print(f"[validate_video_generation] wandb video upload failed: {e}")


# ---------------------------------------------------------------------------
# Main training loop -- direct port of Fast-Control-World/scripts/train_fast_wm.py:main
# ---------------------------------------------------------------------------


def main(args: LiberoWMArgs, fcw_path: str) -> None:
    logger = get_logger(__name__, log_level="INFO")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    CrtlWorld, pipeline_cls = _import_crtl_world(fcw_path)

    model = CrtlWorld(args)
    if args.ckpt_path is not None:
        logger.info(f"Loading checkpoint from {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.to(accelerator.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        accelerator.init_trackers(
            args.wandb_project_name,
            config={},
            init_kwargs={"wandb": {"name": f"train_{now}_{args.tag}"}},
        )
        os.makedirs(args.output_dir, exist_ok=True)
        for module_name in ("unet", "vae", "image_encoder", "text_encoder", "action_encoder"):
            mod = getattr(model, module_name, None)
            if mod is None:
                continue
            n = sum(p.numel() for p in mod.parameters())
            logger.info(f"Number of parameters in {module_name}: {n / 1e6:.2f}M")

    train_dataset = LiberoLatentDataset(args, mode="train")
    val_dataset = LiberoLatentDataset(args, mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_batch = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps * total_batch / max(1, len(train_loader)))
    logger.info("***** Running LIBERO WM training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num val examples = {len(val_dataset)}")
    logger.info(f"  Effective total batch = {total_batch}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    train_loss = 0.0
    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress.set_description("Steps")

    for epoch in range(num_train_epochs):
        for _, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen, _ = model(batch)
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss_gen)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                if global_step % 100 == 0:
                    progress.set_postfix({"loss": train_loss})
                    accelerator.log({"train_loss": train_loss / 100}, step=global_step)
                    train_loss = 0.0
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                if (
                    global_step % args.validation_steps == 5
                    and accelerator.is_main_process
                ):
                    model.eval()
                    with torch.no_grad():
                        with accelerator.autocast():
                            val_loss_sum = 0.0
                            val_count = 0
                            for vbatch in val_loader:
                                vloss, _ = model(vbatch)
                                val_loss_sum += float(vloss.item())
                                val_count += 1
                                if val_count >= 50:  # cap to keep validation cheap
                                    break
                            if val_count:
                                accelerator.log(
                                    {"val_loss": val_loss_sum / val_count},
                                    step=global_step,
                                )
                            for sid in range(args.video_num):
                                ds = train_dataset if args.use_train_set_for_val else val_dataset
                                test_steps = list(args.test_num_inference_steps)
                                if test_steps:
                                    for steps in test_steps:
                                        validate_video_generation(
                                            model, ds, args, global_step,
                                            f"{args.output_dir}/steps_{steps}",
                                            sid, accelerator, pipeline_cls, inference_steps=steps,
                                        )
                                else:
                                    validate_video_generation(
                                        model, ds, args, global_step,
                                        f"{args.output_dir}/steps_{args.num_inference_steps}",
                                        sid, accelerator, pipeline_cls,
                                    )
                    model.train()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a Python config file with get_args() -> LiberoWMArgs")
    parser.add_argument("--fcw_path", type=str, default=FCW_DEFAULT_PATH,
                        help="Path to the Fast-Control-World repo (provides the WM model code).")
    # Common overrides for one-off runs:
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    cli_args = parser.parse_args()

    args = _load_config(cli_args.config)
    for field in ("ckpt_path", "dataset_root_path", "dataset_meta_info_path", "dataset_names", "tag"):
        v = getattr(cli_args, field)
        if v is not None:
            setattr(args, field, v)
    if cli_args.tag is not None:
        # re-derive output_dir
        args.output_dir = f"checkpoints/wm_libero/{args.tag}"
        args.wandb_run_name = args.tag

    main(args, cli_args.fcw_path)


if __name__ == "__main__":
    cli()
