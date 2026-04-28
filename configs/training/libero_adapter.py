"""LIBERO action-adapter training config."""

from openworld.training.action_adapter.config import LiberoAdapterArgs


def get_args() -> LiberoAdapterArgs:
    return LiberoAdapterArgs(
        dataset_root="data/libero_processed",
        suites=(
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ),
        action_dim=7,
        action_num=15,
        hidden_size=512,
        learning_rate=1e-4,
        batch_size=128,
        num_epochs=10,
        output_dir="checkpoints/action_adapter_libero",
        tag="libero_v0",
    )
