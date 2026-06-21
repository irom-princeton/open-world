#!/bin/bash
#SBATCH --job-name=ar_replay_cosmos
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/ar_replay_cosmos/%j.out
#SBATCH --error=slurm_outputs/ar_replay_cosmos/%j.out
#
# Stage 3 (inference, Cosmos backbone) — open-loop trajectory replay with the
# Cosmos student. Decoding still uses the Wan VAE (shared 16-ch latent space).
#
#   CKPT=checkpoints/ar_wm/ar_cosmos_droid/checkpoint-50000.pt \
#       sbatch bash_scripts/inference/replay_cosmos.sh
#
# Without CKPT the untrained backbone runs (plumbing smoke only — video is noise).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"

CONFIG="${CONFIG:-configs/training/ar_cosmos_droid.py}"
LATENT_ROOT="${LATENT_ROOT:-data/droid_ar_latents}"
SPLIT="${SPLIT:-val}"
HISTORY_BLOCKS="${HISTORY_BLOCKS:-1}"

set -x
exec "$PY" scripts/replay_ar.py \
    --config "$CONFIG" \
    --latent-root "$LATENT_ROOT" \
    --split "$SPLIT" \
    --history-blocks "$HISTORY_BLOCKS" \
    ${CKPT:+--checkpoint "$CKPT"} \
    ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
    ${NUM_EPISODES:+--num-episodes "$NUM_EPISODES"} \
    ${EPISODE_ID:+--episode-id "$EPISODE_ID"} \
    ${MAX_BLOCKS:+--max-blocks "$MAX_BLOCKS"} \
    ${SEPARATE:+--separate}
