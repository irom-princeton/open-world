# Trajectory Replay

Feed a recorded episode's action sequence through a trained world model and compare
the predicted video against ground truth (GT | PRED side-by-side + per-episode
latent/pixel MSE & PSNR). Models: see [MODELS.md](MODELS.md).

## Autoregressive (Wan / Cosmos)

Prime the model with the first ground-truth block(s), feed the full recorded action
sequence open-loop, and let the student generate the rest.

```bash
# launcher (Cosmos: bash_scripts/inference/replay_cosmos.sh):
CKPT=checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
  sbatch bash_scripts/inference/replay_wan.sh

# or the entrypoint directly:
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/replay_ar.py \
    --config configs/training/ar_wan_droid.py \
    --checkpoint checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
    --latent-root data/droid_ar_latents --split val \
    --history-blocks 1 --output-dir checkpoints/ar_wm/ar_wan_droid/replay
```

- `--history-blocks` = ground-truth blocks used to prime the cache (`1` ≈ "first
  frame only"); the rest is generated. Without `--checkpoint` the untrained backbone
  runs (validates plumbing; video is noise).
- Core: `openworld/autoregressive/infer/replay.py` (latent-space, decode-free,
  CPU-testable); VAE decode in `data/decode.py`.

**Interactive (live keyboard control):** `scripts/interactive_ar.py`
(`infer/interactive.py:InteractiveRoller`) serves a browser/MJPEG stream you drive
with the keyboard (Wan only). See its module docstring for controls and tunneling.

## SVD bidirectional (`CrtlWorld`)

Closed-loop replay (5-frame chunks) on LIBERO data:

```bash
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint checkpoints/wm/libero/checkpoint-30000.pt \
    --data_root  data/libero_collected \
    --output_dir outputs/libero/replay
```
