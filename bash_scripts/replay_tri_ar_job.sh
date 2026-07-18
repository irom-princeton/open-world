#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=replay-tri-ar
#SBATCH --output=slurm_outputs/replay_tri_ar/out_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yy4041@princeton.edu
set -euo pipefail

cd /scratch/gpfs/AM43/yy4041/open-world-autoregressive
mkdir -p slurm_outputs/replay_tri_ar
# Wan VAE/backbone are loaded from local external/, but keep HF offline for safety.
export HF_HOME="${HF_HOME:-/scratch/gpfs/AM43/yy4041/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
# Make the repo-root `configs` package importable when running scripts/ directly.
export PYTHONPATH="/scratch/gpfs/AM43/yy4041/open-world-autoregressive:${PYTHONPATH:-}"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

STAGE=data/tri_0609_ar_stage
LATENTS=data/tri_0609_ar_latents
CKPT=checkpoints/ar_wm/ar_wan_studentinit_aligned/checkpoint-36000.pt
OUT=checkpoints/ar_wm/ar_wan_studentinit_aligned/replay_0609
CONFIG=configs/training/ar_wan_studentinit_droid_aligned.py

# 1) Encode the 5 staged episodes to Wan-VAE latents (val split).
.venv/bin/python scripts/preprocess_ar_latents.py \
    --format droid_ctrl_world --root "$STAGE" --out "$LATENTS" \
    --splits val --stats-split val --num-views 3 --min-latent-frames 8

# 2) Use the TRAINING action stats (not the 5-episode stats) for normalization.
cp -f data/droid_ar_latents/stats.json "$LATENTS/stats.json"
echo "copied training stats.json:"; cat "$LATENTS/stats.json"

# 3) Open-loop AR trajectory replay: prime first GT block, feed full GT action
#    sequence, autoregress the rest. Writes side-by-side GT|PRED + separate gt/pred.
.venv/bin/python scripts/replay_ar.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --latent-root "$LATENTS" \
    --split val \
    --output-dir "$OUT" \
    --separate \
    --history-blocks 1 \
    --denoising-steps 32 \
    --max-blocks 8 \
    --video-fps 8

echo "REPLAY DONE -> $OUT"
