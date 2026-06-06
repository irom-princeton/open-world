# Autoregressive, self-forcing world model (`openworld.autoregressive`)

This branch adds an **autoregressive** video world model alongside the existing
bidirectional SVD `CrtlWorld`. It ports the recipe behind NVIDIA OmniDreams's
long-horizon driving video — **a DiT initialised from a strong bidirectional
video prior, made block-causal with a KV-cache, and distilled on its own rollouts
(self-forcing / DMD)** — to robot manipulation.

## Why

The SVD model denoises each future chunk from fresh Gaussian noise with only a
handful of sparsely-sampled history frames and **no persistent latent memory**,
so objects drift and disappear over long rollouts. For driving, OmniDreams can
lean on an explicit rendered "state" (HD-map / bbox / ego-trajectory image); for
manipulation there is **no ground-truth object state to condition on** — the hard
thing to predict (object dynamics under contact) must be *generated*. So the
transferable lever is **not** the conditioning but the **memory + training
recipe**:

1. **Block-causal attention + KV-cache** — a real latent memory carried across
   time (the cache *is* the state), instead of re-deriving the scene each chunk.
2. **Self-forcing / DMD distillation** — train the student on the imperfect
   history it actually produces, closing the train/inference gap that causes
   error accumulation and object disappearance.
3. **Initialise from a bidirectional video prior** (Wan2.1-1.3B / Cosmos-Predict2-2B)
   — strong object semantics for free; "autoregressive" and "init from a
   bidirectional model" are not in tension (just re-mask attention + distill).

Conditioning stays minimal: per-frame **action** embeddings (+ optional text) via
cross-attention, first/history frames as clean latents in the cache.

## Layout

```
openworld/autoregressive/
  config.py            ARWMArgs + BACKBONE_PRESETS
  causal/              mask.py · kv_cache.py · context.py · attention.py   <- novel core (backbone-agnostic)
  backbones/           base.py (ABC) · _attn.py (Wan/Cosmos block-causal procs)
                       wan.py · cosmos_predict2.py · svd.py · dummy.py
  conditioning/        action.py (reuses vidwm encoder) · multiview.py
  distill/             scheduler.py · dmd.py · self_forcing.py
  model.py             ARWorldModel (= self-forcing generator) + build_training_stack
  train_self_forcing.py  accelerate entrypoint (+ --smoke)
  tests/test_ar.py     CPU unit tests (DummyDiT)
configs/training/      ar_wan_1_3b.py · ar_cosmos2_2b.py
```

## The causal core (proven correct)

`tests/test_ar.py` asserts the property the whole design rests on: the **KV-cache
rollout reproduces the block-causal masked forward exactly** — `max|full −
cached_rollout| = 0.0` on the DummyDiT *and* on the real diffusers Wan **and
Cosmos** transformers (RoPE-offset included), for both unbounded and
sliding-window (`max_kv_blocks`) memory. So the autoregressive memory is
mathematically the same computation as the trained-with mask, just streamed.

* Token layout: a video latent is `F` frames × `tokens_per_frame`; frames are
  grouped into blocks of `frames_per_block`; a query attends to its block + all
  earlier blocks (causal), bidirectional within a block (and across camera views
  at the same time). Matches OmniDreams `num_frame_per_block`.
* During few-step denoising of a block, intermediate steps attend to the clean
  cache + the current noisy block **without committing**; only the finalized
  clean block is appended (`commit=True`).

## Backbones

| key | model | status |
|---|---|---|
| `wan_1_3b` | Wan2.1-T2V-1.3B (`WanTransformer3DModel`) | **recommended.** `forward_train` (block-causal mask) + `forward_cached` (KV-cache, RoPE-offset) both validated against real weights. The Self-Forcing recipe was built on Wan. |
| `cosmos_predict2_2b` | Cosmos-Predict2-2B (`CosmosTransformer3DModel`) | OmniDreams's own substrate. `forward_train` + `forward_cached` (KV-cache, RoPE-offset via `cosmos_predict2.py:_offset_rope_cosmos`, mirroring OmniDreams' `start_frame_for_rope`) both validated (`tests/test_ar.py`). |
| `svd` | legacy SVD UNet | intentionally not implemented — UNet+temporal-conv is the wrong substrate for block-causal+cache; the bidirectional `CrtlWorld` remains for the baseline. |
| `dummy` | tiny CPU DiT | tests only. |

Build with `--config configs/training/ar_wan_1_3b.py`; `random_init_backbone=True`
builds untrained small models for CI.

## Self-forcing / DMD

`distill/` implements a faithful reference of the Self-Forcing/DMD2 loop:
generator (causal student) rolls out few-step with the cache; a **critic
("fake score")** learns to denoise the student's samples; a frozen
**bidirectional teacher ("real score", CFG'd)** anchors the data distribution;
the generator's DMD loss is the score difference in clean-latent space. Gradient
is retained only through each block's final denoising step (Self-Forcing) to keep
long rollouts tractable. `train_self_forcing.py --smoke` runs the whole loop
weightless on CPU (and is a unit test).

## How to run

```bash
uv sync --extra autoregressive                       # env (this branch)
python -m openworld.autoregressive.train_self_forcing --smoke   # weightless sanity
.venv/bin/python -m pytest openworld/autoregressive/tests -q    # unit tests

# GPU (this cluster is offline on compute nodes — use sbatch, not the login node).
# Per-stage launchers live under bash_scripts/ (see bash_scripts/README.md):
bash   bash_scripts/download_weights.sh                  # login node: Wan transformer + VAE -> external/
sbatch bash_scripts/data_process/preprocess_latents.sh   # stage 1: RGB -> 16-ch Wan-VAE latents
sbatch bash_scripts/data_process/validate_data.sh        # stage 1 check: latents -> real Wan forward
sbatch bash_scripts/training/train_wan.sh                # stage 2: self-forcing / DMD distillation (Wan)
sbatch bash_scripts/training/train_cosmos.sh             # stage 2: same, Cosmos backbone
CKPT=checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
  sbatch bash_scripts/inference/replay_wan.sh            # stage 3: open-loop trajectory replay

# Ad-hoc commands still go through the generic runner:
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/smoke_wan_real.py   # real-weights GPU smoke
```

**Offline-cluster loading gotcha:** the compute nodes have no internet, and
diffusers' sharded-checkpoint loader pings the Hub for a bare repo id *even with*
`HF_HUB_OFFLINE=1`. So weights must be loaded from a **local directory**
(`backbone_ckpt="external/Wan2.1-T2V-1.3B-Diffusers"`, set in the Wan config).
`bash_scripts/ar_gpu.slurm` exports `HF_HUB_OFFLINE=1`/`HF_HOME` for you.

## Inference: open-loop trajectory replay

The first inference setup (a stand-in for the eventual closed-loop "interact with
any policy" entrypoint). Given a ground-truth trajectory (preprocessed latents +
recorded actions, e.g. the `val` split): prime the model with the first
ground-truth frame(s), feed the **full recorded action sequence open-loop**, and
let the student autoregressively generate the rest. The generated and
ground-truth latents are decoded with the Wan VAE and written **side by side**
(GT | PRED) per episode, with per-episode latent/pixel MSE + PSNR.

```bash
# launcher (Cosmos: bash_scripts/inference/replay_cosmos.sh):
CKPT=checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
  sbatch bash_scripts/inference/replay_wan.sh

# or the underlying entrypoint directly:
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/replay_ar.py \
    --config configs/training/ar_wan_droid.py \
    --checkpoint checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
    --latent-root data/droid_ar_latents --split val \
    --history-blocks 1 --output-dir checkpoints/ar_wm/ar_wan_droid/replay
```

* `--history-blocks` is the number of ground-truth blocks used to prime the cache
  (`1` ≈ "first frame only"); the rest is generated.
* Reusable core lives in `openworld/autoregressive/infer/replay.py` (latent-space,
  decode-free → CPU-testable: `tests/test_replay.py`); VAE decode is
  `data/decode.py` (`VaeLatentDecoder`, the inverse of `data/encode.py`).
* Without `--checkpoint` the untrained backbone runs — validates the full
  inference plumbing (build → rollout → decode → side-by-side video) end to end,
  but the video is noise. Verified on A100: a `val` clip produces a `[29, 576,
  644, 3]` side-by-side mp4 (8 latent → 29 RGB frames, 3 views × 192px, 2×320px+gap).

## Validated on H200 (real weights)

`scripts/smoke_wan_real.py` on one H200, loading the actual Wan2.1-1.3B
(1.42B params, 30 self-attn layers):

* fp32 `forward_train` vs KV-cached rollout: **max err 4.8e-06** — the
  block-causal cache is exact at real scale, not just on the DummyDiT.
* Wan VAE (`AutoencoderKLWan`) loads, **16 latent channels** (the re-encode target).
* **~34 ms / block** (2 latent frames, single forward, bf16, 320² 1 view) → a
  2-step distilled rollout ≈ ~85 ms/block ≈ **~90 fps effective single-view** —
  the OmniDreams/FlashDreams real-time regime.
* **Full data → training-step chain validated** (`scripts/validate_train_step.py`):
  DROID annotations → format adapter → Wan-VAE 16-ch latents (3 cams height-stacked,
  72×40) → `ARLatentDataset` → the **full self-forcing / DMD loop on three real
  Wan-1.3B models** (generator KV-cache rollout + online critic + CFG'd frozen
  teacher) produces finite, decreasing losses on the H200 (`gen_loss≈0.46`,
  `critic_loss 0.51→0.19` over 3 steps with random-init teacher).

## Dtype

Two knobs: **`cfg.dtype`** is the parameter/optimizer dtype, **`cfg.mixed_precision`**
the compute dtype. The real configs use **fp32 master weights + bf16 autocast**
(`dtype=float32`, `mixed_precision="bf16"`) — the standard mixed-precision recipe
and what the Self-Forcing/CausVid references use:

* **fp32 params + fp32 AdamW state.** At `lr=6e-6` on O(1) weights, a per-step
  update ≈1e-6 is below bf16's ~3-digit precision and would be rounded away — the
  optimizer stalls on the smallest-but-real updates over a 200k-step run. The
  master copy must be fp32 (Adam's momentum buffers are fp32 regardless).
* **bf16 compute.** The backbone `_call` coerces inputs to the param dtype, then
  runs the transformer matmuls/convs under `torch.autocast("cuda", bf16)` when
  `cfg.autocast_dtype` is set. `autocast_dtype` is `None` when params already
  carry the compute dtype (pure-fp32 or pure-bf16), so the same path also serves
  a quick `dtype=bf16` smoke.
* **fp32 loss math.** `make_cfg_score_fn` casts the backbone's (bf16) score output
  to fp32, so the precision-sensitive DMD difference `x0_fake - x0_real` and the
  CFG combination — differencing two large, similar tensors — run in fp32.
  `FlowMatchScheduler.add_noise` / `x0_from_velocity` cast `sigma` (built fp32) to
  the latent dtype so the noising arithmetic stays in the latent's dtype
  (regression-tested by `test_scheduler_preserves_dtype`).
* `generate_rollout` draws block noise in the generator's parameter dtype;
  `train_self_forcing.main` casts models + batch to `cfg.dtype`. Backward and the
  optimizer step run **outside** autocast (autocast wraps only the transformer
  forward inside `_call`).

To switch precision: `dtype=float32, mixed_precision="no"` → pure fp32;
`dtype=bfloat16` → uniform bf16 (fast smoke, not for long runs).

## What still needs real weights / a GPU (honest status)

* ~~**VAE re-encode (required).**~~ **Done.** `scripts/preprocess_ar_latents.py`
  re-encodes any registered data format (currently `droid_ctrl_world`) into the
  backbone VAE's **16-channel** latents on disk; `ARLatentDataset` consumes them.
  `cfg.in_channels` reflects the backbone.
* **Stage-0 weights.** Point `student_init_ckpt` (E1: causal-student init) and
  `teacher_ckpt` (E2: robot-finetuned bidirectional teacher) at real checkpoints;
  the critic is initialised from the teacher.
* **`sequence_pack` multiview** — the OmniDreams layout (per-view latents +
  view-id embedding) needs the preprocessor to emit per-view latents; the default
  `height_stack` works with the current dataset unchanged.
* DMD/self-forcing **convergence + tuning** (lrs, `critic_steps_per_gen_step`,
  CFG scale, `denoising_step_list`) needs the real teacher and GPU runs.

## Expected speed (recap)

Per the FlashDreams numbers, OmniDreams' 2-step chunk-2 Wan-class single-view
runner hits ~68 effective FPS on GB300; derated to one H100 and robot-resolution
single-view this is comfortably real-time, ~×3–4 slower for 4 joint views. The
few-step distilled student here targets the same regime.
