"""Thin wrapper for LIBERO action-adapter training.

Equivalent to:
    python -m openworld.training.action_adapter.train \\
        --config configs/training/libero_adapter.py
"""

from openworld.training.action_adapter.train import cli

if __name__ == "__main__":
    cli()
