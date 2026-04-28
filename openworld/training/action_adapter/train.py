"""LIBERO action-adapter training loop.

Mirrors ``Fast-Control-World/models/action_adapter/train2.py`` but uses
LIBERO data and the LIBERO ``Dynamics`` MLP (no FK).

Usage:
    python -m openworld.training.action_adapter.train \\
        --config configs/training/libero_adapter.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from openworld.training.action_adapter.config import LiberoAdapterArgs
from openworld.training.action_adapter.dataset import LiberoAdapterDataset
from openworld.training.action_adapter.model import LiberoDynamics


def _load_config(config_arg: str | None) -> LiberoAdapterArgs:
    if config_arg is None:
        return LiberoAdapterArgs()
    path = Path(config_arg)
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Config not found: {config_arg}")
    spec = importlib.util.spec_from_file_location("user_libero_adapter_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_args"):
        raise AttributeError(f"{path} must define get_args() -> LiberoAdapterArgs")
    return mod.get_args()


def main(args: LiberoAdapterArgs) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.tag,
            config={k: v for k, v in vars(args).items() if not k.startswith("_")},
            dir=args.output_dir,
        )

    train_ds = LiberoAdapterDataset(
        dataset_root=args.dataset_root,
        suites=list(args.suites),
        mode="train",
        annotation_name=args.annotation_name,
        action_num=args.action_num,
        action_dim=args.action_dim,
        policy_skip_step=args.policy_skip_step,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_ds = None
    try:
        val_ds = LiberoAdapterDataset(
            dataset_root=args.dataset_root,
            suites=list(args.suites),
            mode="val",
            annotation_name=args.annotation_name,
            action_num=args.action_num,
            action_dim=args.action_dim,
            policy_skip_step=args.policy_skip_step,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    except RuntimeError:
        val_loader = None
        print("[train_adapter] No val split found; skipping validation.")

    model = LiberoDynamics(
        action_dim=args.action_dim,
        action_num=args.action_num,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[LiberoDynamics] {n_params/1e6:.2f}M parameters")

    update_step = 0
    running_loss = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            current_pose = batch["current_pose"].to(device)
            delta_chunk = batch["delta_chunk"].to(device)
            future_pose = batch["future_pose"].to(device)

            loss = model(current_pose, delta_chunk, future_pose=future_pose)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_step += 1
            running_loss += loss.item()
            if wandb_run is not None:
                wandb_run.log({"train/loss_step": loss.item(), "epoch": epoch}, step=update_step)
            if update_step % args.log_every == 0:
                avg = running_loss / args.log_every
                print(f"  step {update_step}: loss={avg:.5f}")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": avg}, step=update_step)
                running_loss = 0.0

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                count = 0
                for batch in val_loader:
                    cp = batch["current_pose"].to(device)
                    dc = batch["delta_chunk"].to(device)
                    fp = batch["future_pose"].to(device)
                    val_loss += model(cp, dc, future_pose=fp).item()
                    count += 1
                if count:
                    avg_val = val_loss / count
                    print(f"  [val] epoch {epoch}: loss={avg_val:.5f}")
                    if wandb_run is not None:
                        wandb_run.log({"val/loss": avg_val, "epoch": epoch}, step=update_step)

        if (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = os.path.join(args.output_dir, args.ckpt_name_pattern.format(epoch=epoch))
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a .py file with get_args() -> LiberoAdapterArgs")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    cli_args = parser.parse_args()
    args = _load_config(cli_args.config)
    if cli_args.dataset_root:
        args.dataset_root = cli_args.dataset_root
    if cli_args.output_dir:
        args.output_dir = cli_args.output_dir
    if cli_args.tag:
        args.tag = cli_args.tag
    main(args)


if __name__ == "__main__":
    cli()
