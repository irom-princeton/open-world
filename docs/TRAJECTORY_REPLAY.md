# Trajectory Replay

Feed a recorded episode's action sequence through a trained world model and compare
the predicted video against ground truth (GT | PRED side-by-side + per-episode
latent/pixel MSE & PSNR). Models: see [MODELS.md](MODELS.md).

## Autoregressive (Wan / Cosmos)

Prime the model with the first ground-truth block(s), feed the full recorded action
sequence open-loop, and let the student generate the rest.

```bash
# published 2-view student checkpoint (see docs/MODELS.md to download it):
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/replay_ar.py \
    --config configs/inference/ar_wan_student_2view.py \
    --checkpoint checkpoints/ar_wm/wm_student_2view.pt \
    --latent-root data/droid_ar_latents --split val \
    --history-blocks 4 --output-dir checkpoints/ar_wm/wm_student_2view/replay
```

- Pick the `configs/inference/*` config that matches the checkpoint (see
  [configs/inference/README.md](../configs/inference/README.md)); it pins the view
  count, action dims, block geometry, and state-pred head the weights need. The
  bimanual checkpoint uses `ar_wan_student_3view_bimanual.py` and its own 3-view /
  20-dim latents.
- These students are **undistilled**, so replay samples with the many-step preview
  schedule automatically — do not expect the few-step distilled path here.
- `--history-blocks` = ground-truth blocks used to prime the cache; match the
  config's `num_history_blocks` (4 for the published students). Without `--checkpoint`
  the untrained backbone runs (validates plumbing; video is noise).
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
