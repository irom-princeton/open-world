#!/bin/bash
#SBATCH --job-name=dmd_aligned
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/ar_dmd_aligned/%j.out
#SBATCH --error=slurm_outputs/ar_dmd_aligned/%j.out
#
# Stage L0 self-forcing / DMD distillation, action_cond_mode="cross_attn_aligned".
# Generator <- ar_wan_studentinit_aligned/checkpoint-40000.pt; teacher + critic
# backbone <- the LATEST checkpoint in checkpoints/ar_wm/ar_wan_teacher_aligned/
# (highest step, resolved at job start). Writes checkpoints to
# checkpoints/ar_wm/ar_wan_dmd_aligned/ and logs to wandb run "ar_wan_dmd_aligned".
# Crash-safe resume is automatic: re-submitting continues from the latest
# training_state in the output dir.
#
# Typically submitted held behind the teacher resume job:
#   sbatch --dependency=afterany:<teacher_jobid> bash_scripts/training/train_dmd_aligned.sh
#
# Override GPU count with GPUS=N (keep --gres in sync).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"
mkdir -p slurm_outputs/ar_dmd_aligned

CONFIG=configs/training/ar_wan_droid_aligned.py
GPUS="${GPUS:-8}"

# --- Resolve + pin the teacher checkpoint (single source of truth: the config's
#     own resolver, so the bash log records exactly what the run will load). -----
DMD_TEACHER_CKPT="$(.venv/bin/python -c 'from configs.training.ar_wan_droid_aligned import _latest_teacher_ckpt; print(_latest_teacher_ckpt())')"
export DMD_TEACHER_CKPT
echo "DMD student checkpoint: checkpoints/ar_wm/ar_wan_studentinit_aligned/checkpoint-40000.pt"
echo "DMD teacher checkpoint: ${DMD_TEACHER_CKPT}"

# --- Safety guard ------------------------------------------------------------
# The teacher resume started from step 13000. If it crashed before writing any
# new checkpoint (the failure the dependency=afterany cannot distinguish from a
# clean time-out), the latest step would still be <= 13000. Refuse to distill
# from a teacher that never advanced -- exit non-zero so this is visible.
MIN_TEACHER_STEP=13000
TEACHER_STEP="$(echo "$DMD_TEACHER_CKPT" | sed -E 's#.*checkpoint-([0-9]+).*#\1#')"
echo "DMD teacher step: ${TEACHER_STEP} (must exceed ${MIN_TEACHER_STEP})"
if [ -z "$TEACHER_STEP" ] || [ "$TEACHER_STEP" -le "$MIN_TEACHER_STEP" ]; then
    echo "ERROR: teacher did not advance past step ${MIN_TEACHER_STEP}; the resume" \
         "likely failed. Not launching DMD." >&2
    exit 1
fi

set -x
exec "${TORCHRUN:-.venv/bin/torchrun}" --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29515}" \
    -m openworld.autoregressive.train_self_forcing \
    --config "$CONFIG"
