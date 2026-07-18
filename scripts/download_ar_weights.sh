#!/bin/bash
# Download AR-world-model backbone weights into the shared HF cache ($HF_HOME),
# so `from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder=...)` resolves
# on both the login node and compute nodes without a --local-dir.
#
# We fetch only the transformer (the DiT we patch) and the Wan VAE (needed to
# re-encode the robot dataset to 16-ch latents). The UMT5-XXL text encoder
# (~11 GB) is intentionally skipped: this world model conditions on robot
# actions, not text. Add `text_encoder/* tokenizer/*` to --include if you want
# text conditioning later.
#
# Network I/O only (safe on the login node). Run: bash scripts/download_ar_weights.sh
set -euo pipefail

REPO="${1:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
DEST="${2:-external/$(basename "$REPO")}"
echo "Downloading $REPO (transformer + vae) -> local dir: $DEST"

# IMPORTANT: download to a *local directory* (not just the hub cache). The
# compute nodes are offline, and diffusers' sharded-checkpoint loader still
# pings the Hub when given a repo id even with HF_HUB_OFFLINE=1; pointing
# from_pretrained at a local dir reads the cached index directly and works
# offline. Set the backbone_ckpt in your config to this dir.
mkdir -p "$DEST"
.venv/bin/hf download "$REPO" --local-dir "$DEST" \
    --include "transformer/*" "vae/*" "model_index.json" "scheduler/*" "configuration.json"

echo "DONE: $REPO -> $DEST  (set ARWMArgs.backbone_ckpt='$DEST')"
