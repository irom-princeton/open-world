"""LIBERO action-adapter training config."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiberoAdapterArgs:
    # ---------------- data --------------------
    dataset_root: str = "data/libero_processed"
    suites: tuple[str, ...] = (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    )
    annotation_name: str = "annotation"
    policy_skip_step: int = 2

    # ---------------- model -------------------
    action_dim: int = 7
    action_num: int = 15
    hidden_size: int = 512

    # ---------------- training ----------------
    learning_rate: float = 1e-4
    batch_size: int = 128
    num_workers: int = 8
    num_epochs: int = 10
    log_every: int = 100
    save_every_epochs: int = 1

    output_dir: str = "checkpoints/action_adapter_libero"
    tag: str = field(default="libero_v0")

    # ---------------- wandb -------------------
    use_wandb: bool = True
    wandb_project: str = "libero_action_adapter"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        # File-name pattern mirrors DROID: model2_<action_num>_<epoch>.pth
        self.ckpt_name_pattern: str = f"model2_{self.action_num}_{{epoch}}.pth"
