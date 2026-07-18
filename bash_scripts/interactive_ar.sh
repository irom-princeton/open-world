#!/usr/bin/env bash
# Launch the keyboard-controlled AR world-model browser demo on a GPU node.
#
# Get an H200 first (interactive node needed for the inference frame-rate):
#     get_h200            # alias: salloc --partition=ailab --gres=gpu:1 ...
# then on the allocated node:
#     bash bash_scripts/interactive_ar.sh [aligned|adaln] [checkpoint.pt] [port]
#
# Defaults: mode=aligned, latest checkpoint in that dir, port=8000.
# Finally tunnel from your laptop:  ssh -N -L 8000:<gpu-node>:8000 <you>@<login>
# and open http://localhost:8000
#
# nounset trips on the compute-node bashrc, so -eo only (see CLAUDE memory).
set -eo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-aligned}"
CKPT="${2:-}"
PORT="${3:-8000}"

case "$MODE" in
  aligned) CONFIG="configs/training/ar_wan_studentinit_droid_aligned.py";
           CKDIR="checkpoints/ar_wm/ar_wan_studentinit_aligned" ;;
  adaln)   CONFIG="configs/training/ar_wan_studentinit_droid_adaln.py";
           CKDIR="checkpoints/ar_wm/ar_wan_studentinit_adaln" ;;
  *) echo "unknown mode '$MODE' (use 'aligned' or 'adaln')"; exit 1 ;;
esac

if [ -z "$CKPT" ]; then
  # latest permanent (non-rolling) checkpoint by step number
  CKPT="$(ls "$CKDIR"/checkpoint-*.pt 2>/dev/null | grep -v rolling \
          | sort -t- -k2 -n | tail -1)"
fi
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: '$CKPT' (dir: $CKDIR)"; exit 1
fi

echo "[interactive] mode=$MODE config=$CONFIG"
echo "[interactive] checkpoint=$CKPT  port=$PORT"

exec .venv/bin/python scripts/interactive_ar.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --port "$PORT" \
  "${@:4}"
