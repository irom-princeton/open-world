#!/bin/bash
#SBATCH --job-name=eval_wm
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/eval_wm/%j.out
#SBATCH --error=slurm_outputs/eval_wm/%j.out
#
# ============================================================================
# Unified policy-eval launcher  —  pi0.5 closed-loop under any world model.
# ============================================================================
# A run is specified by just three things:
#       --wm {ctrlworld|ar|weaver}     world model
#       --benchmark <name>             benchmark suite     (default: 0617)
#       --checkpoint <path>            world-model ckpt    (default: from config)
#
# Everything else — which venv to use, whether to spin up the pi0.5 websocket
# server, PYTHONPATH, diffusion step count, view order, action adapter — is
# abstracted into the per-WM profile below and the eval YAML, so you never
# re-specify it on the command line.
#
# The eval YAML is resolved as configs/evaluation/<benchmark>_<wm>_pi05.yaml
# unless you pass --config explicitly.
#
# Usage:
#   sbatch bash_scripts/eval_wm.sbatch --wm ar
#   sbatch bash_scripts/eval_wm.sbatch --wm weaver --benchmark 0617
#   sbatch bash_scripts/eval_wm.sbatch --wm ctrlworld --checkpoint checkpoints/ctrlworld/checkpoint-10000.pt
#   # quick 2-init smoke (full duration) by pointing at the smoke suite:
#   sbatch bash_scripts/eval_wm.sbatch --wm ar --dataset data/benchmark/0617_smoke
#
# Optional overrides (all forwarded to run_evaluation.py, no YAML edit needed):
#   --config <path>   --dataset <path>   --video-dir <path>   --duration <sec>
#   --port <n>        websocket port for the weaver pi0.5 server (default 8123)
# ----------------------------------------------------------------------------

# Source bashrc before strict mode (compute-node /etc/bashrc trips set -u).
source ~/.bashrc
set -eo pipefail

OW="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$OW"

# ---- parse args -------------------------------------------------------------
WM=""; BENCHMARK="0617"; CONFIG=""; CHECKPOINT=""
DATASET=""; VIDEO_DIR=""; DURATION=""; PORT="${PORT:-8123}"
while [ $# -gt 0 ]; do
  case "$1" in
    --wm)         WM="$2"; shift 2 ;;
    --benchmark)  BENCHMARK="$2"; shift 2 ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --video-dir)  VIDEO_DIR="$2"; shift 2 ;;
    --duration)   DURATION="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    *) echo "[eval_wm] unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$WM" ]; then echo "[eval_wm] --wm {ctrlworld|ar|weaver} is required" >&2; exit 2; fi
[ -z "$CONFIG" ] && CONFIG="configs/evaluation/${BENCHMARK}_${WM}_pi05.yaml"
if [ ! -f "$CONFIG" ]; then echo "[eval_wm] config not found: $CONFIG" >&2; exit 2; fi

mkdir -p slurm_outputs/eval_wm

# Weights/data are pre-cached locally; compute nodes are offline.
export HF_HOME="${HF_HOME:-/home/yy4041/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# openworld is importable from the repo root for every interpreter we invoke.
export PYTHONPATH="$OW:${PYTHONPATH:-}"

# ---- per-WM runtime profile -------------------------------------------------
# venv-eval hosts torch + jax/openpi; the weaver venv hosts torch-2.7/diffusers.
EVAL_VENV="$OW/.venv-eval"           # jax/openpi + torch (ar, ctrlworld, server)
NEEDS_SERVER=0                        # weaver runs pi0.5 out-of-process
case "$WM" in
  ar|ctrlworld)
    RUN_VENV="$EVAL_VENV" ;;
  weaver)
    NEEDS_SERVER=1
    # Prefer the repo-local .venv-weaver (built by setup_eval_env.sh); fall back
    # to the legacy external WEAVER venv if it hasn't been built yet.
    if   [ -x "$OW/.venv-weaver/bin/python" ]; then RUN_VENV="$OW/.venv-weaver"
    elif [ -x "/scratch/gpfs/AM43/yy4041/WEAVER/.venv/bin/python" ]; then
      RUN_VENV="/scratch/gpfs/AM43/yy4041/WEAVER/.venv"
      echo "[eval_wm] NOTE: .venv-weaver not found; using legacy external WEAVER venv."
    else
      echo "[eval_wm] no weaver venv found. Build one: bash bash_scripts/setup_eval_env.sh --weaver" >&2
      exit 1
    fi ;;
  *) echo "[eval_wm] unknown --wm '$WM' (expected ctrlworld|ar|weaver)" >&2; exit 2 ;;
esac
PY="$RUN_VENV/bin/python"
if [ ! -x "$PY" ]; then echo "[eval_wm] interpreter missing: $PY" >&2; exit 1; fi

# Forward optional overrides to run_evaluation.py (no YAML edit required).
EVAL_ARGS=(--config "$CONFIG")
[ -n "$CHECKPOINT" ] && EVAL_ARGS+=(--checkpoint "$CHECKPOINT")
[ -n "$DATASET" ]    && EVAL_ARGS+=(--dataset_path "$DATASET")
[ -n "$VIDEO_DIR" ]  && EVAL_ARGS+=(--video_dir "$VIDEO_DIR")
[ -n "$DURATION" ]   && EVAL_ARGS+=(--duration "$DURATION")

echo "[eval_wm] wm=$WM config=$CONFIG venv=$RUN_VENV server=$NEEDS_SERVER"
echo "[eval_wm] host=$(hostname) job=${SLURM_JOB_ID:-NA} gpu=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ---- (weaver only) launch the out-of-process pi0.5 websocket server ---------
# The weaver venv (torch-2.7) can't host jax/openpi, so pi0.5 runs in .venv-eval
# and shares the H200; keep jax from grabbing the whole GPU.
if [ "$NEEDS_SERVER" -eq 1 ]; then
  PI_DIR="${PI_DIR:-/scratch/gpfs/AM43/yy4041/.cache/openpi/openpi-assets/checkpoints/pi05_droid}"
  SRVLOG="slurm_outputs/eval_wm/server_${SLURM_JOB_ID:-local}.log"
  OPENPI_PP="$OW/external/openpi/src:$OW/external/openpi/packages/openpi-client/src"
  echo "[eval_wm] launching pi05 server (port $PORT) -> $SRVLOG"
  PYTHONPATH="$OPENPI_PP" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    "$EVAL_VENV/bin/python" "$OW/external/openpi/scripts/serve_policy.py" \
    --env DROID05 --port "$PORT" \
    policy:checkpoint --policy.config pi05_droid --policy.dir "$PI_DIR" \
    > "$SRVLOG" 2>&1 &
  SRV_PID=$!
  trap 'kill $SRV_PID 2>/dev/null || true' EXIT

  echo "[eval_wm] waiting for server on :$PORT ..."
  ready=0
  for i in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then exec 3>&- 3<&-; ready=1; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[eval_wm] server died:"; tail -50 "$SRVLOG"; exit 1; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then echo "[eval_wm] server did not come up:"; tail -50 "$SRVLOG"; exit 1; fi
  echo "[eval_wm] server up."; tail -5 "$SRVLOG"
fi

# ---- run the eval -----------------------------------------------------------
echo "[eval_wm] running: $PY scripts/run_evaluation.py ${EVAL_ARGS[*]}"
"$PY" "$OW/scripts/run_evaluation.py" "${EVAL_ARGS[@]}"
echo "[eval_wm] done."
