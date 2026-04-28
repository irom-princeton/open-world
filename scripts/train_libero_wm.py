"""Thin wrapper for LIBERO world-model training.

Equivalent to:

    accelerate launch -m openworld.training.world_model.train_wm \\
        --config configs/training/libero_wm.py
"""

from openworld.training.world_model.train_wm import cli

if __name__ == "__main__":
    cli()
