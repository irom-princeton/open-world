#!/bin/bash
#SBATCH --job-name=eval0617_weaver
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/eval0617_weaver/%j.out
#SBATCH --error=slurm_outputs/eval0617_weaver/%j.out
#
# Policy eval of pi0.5 under WEAVER on one H200. WEAVER (its own torch-2.7 venv)
# can't host jax/openpi, so pi0.5 runs as an OpenPI websocket server in the
# .venv-eval interpreter, sharing the GPU; weaver connects over localhost.
#
#   sbatch bash_scripts/eval_weaver_0617.sbatch [config.yaml]

# Source bashrc before strict mode (compute-node /etc/bashrc trips set -u).
source ~/.bashrc
set -eo pipefail

OW=/scratch/gpfs/AM43/yy4041/open-world-autoregressive
cd "$OW"
CONFIG="${1:-$OW/configs/evaluation/0617_weaver_pi05.yaml}"
PORT="${PORT:-8123}"
mkdir -p slurm_outputs/eval0617_weaver

# SD3 (weaver encoder) + pi05 ckpt are pre-cached; compute nodes are offline.
export HF_HOME="${HF_HOME:-/home/yy4041/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

EVAL_VENV="$OW/.venv-eval"
WEAVER_VENV=/scratch/gpfs/AM43/yy4041/WEAVER/.venv
PI_DIR=/scratch/gpfs/AM43/yy4041/.cache/openpi/openpi-assets/checkpoints/pi05_droid
SRVLOG="slurm_outputs/eval0617_weaver/server_${SLURM_JOB_ID:-local}.log"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} gpu=${CUDA_VISIBLE_DEVICES:-NA} port=$PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# 1) launch the pi0.5 (pi05_droid) websocket server. Keep jax from grabbing the
#    whole H200 so weaver (torch) has room on the shared GPU.
echo "[weaver-job] launching pi05 server -> $SRVLOG"
# serve_policy.py imports `openpi` / `openpi_client` from the repo src dirs (not
# pip-installed), so put them on PYTHONPATH for the server process.
OPENPI_PP="$OW/external/openpi/src:$OW/external/openpi/packages/openpi-client/src"
PYTHONPATH="$OPENPI_PP" \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  "$EVAL_VENV/bin/python" "$OW/external/openpi/scripts/serve_policy.py" \
  --env DROID05 --port "$PORT" \
  policy:checkpoint --policy.config pi05_droid --policy.dir "$PI_DIR" \
  > "$SRVLOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

# 2) wait until the server's TCP port accepts connections (model load ~1-3 min).
echo "[weaver-job] waiting for server on :$PORT ..."
ready=0
for i in $(seq 1 180); do
  if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then exec 3>&- 3<&- ; ready=1; break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[weaver-job] server process died:"; tail -50 "$SRVLOG"; exit 1; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then echo "[weaver-job] server did not come up in time:"; tail -50 "$SRVLOG"; exit 1; fi
echo "[weaver-job] server up. recent log:"; tail -5 "$SRVLOG"

# 3) run the weaver eval in the WEAVER venv; openworld driver via PYTHONPATH.
export PYTHONPATH="$OW:${PYTHONPATH:-}"
echo "[weaver-job] running weaver eval: $CONFIG"
"$WEAVER_VENV/bin/python" "$OW/scripts/run_evaluation.py" --config "$CONFIG"
echo "[weaver-job] done."
