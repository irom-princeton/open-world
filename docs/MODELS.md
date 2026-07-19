# Supported Models

The two world-model families this repo supports. The **autoregressive (Wan/Cosmos)
stack is the primary, actively-developed one**; the **SVD bidirectional model** is a
co-equal supported track that currently powers policy evaluation / RL.

Workflows: [world-model training](#training) · [trajectory replay](TRAJECTORY_REPLAY.md)
· [policy evaluation](EVAL.md).

## Autoregressive (AR) — *primary*

Block-causal DiT with a KV-cache memory, initialised from a bidirectional video
prior and distilled on its own rollouts (self-forcing / DMD). Code:
`openworld/autoregressive/` (`model.py:ARWorldModel`). 
<!-- Architecture & design: **[AUTOREGRESSIVE.md](AUTOREGRESSIVE.md)**. -->

| config | backbone | action-cond modes | platforms | status |
|---|---|---|---|---|
| `wan_1_3b` | Wan2.1-T2V-1.3B | `cross_attn_aligned`, `adaln` | DROID, bimanual | ✅ training<br>❌ few-step<br>✅ checkpoint (see below) |
| `cosmos_predict2_2b` | Cosmos-Predict2-2B | `cross_attn` only | DROID | ❌ training<br>❌ few-step<br>❌ checkpoint |

Training Instructions: [world_model_training/autoregressive.md](world_model_training/autoregressive.md)

### Pretrained student checkpoints

Two ready-to-run AR world-model students are published on Hugging Face at
[`tennyyyin/open-world-ar-wm`](https://huggingface.co/tennyyyin/open-world-ar-wm)
(the repo is private — you need a token with read access; run `hf auth login` once).

| file | views | action input | geom. cond | inference config | aux state head |
|---|---|---|---|---|---|
| `wm_student_2view.pt` | 2 (side + wrist) | cartesian absolute, 7-d | — | `configs/inference/ar_wan_student_2view.py` | 8-d |
| `wm_student_3view_bimanual.pt` | 3 (scene + 2 wrists) | cartesian absolute, 20-d | **camera_cond** (9-ch) | `configs/inference/ar_wan_student_3view_bimanual.py` | 16-d |

Both use `cross_attn_aligned` conditioning and single-frame causal blocks
(`frames_per_block=1`, 4 history frames). They are **undistilled** students, so
sample them with the many-step preview schedule — do **not** pass `--distilled`
(that few-step deployment schedule is for a distilled checkpoint and yields a
blurry colour-wash on these).

The `wm_student_3view_bimanual.pt` student is a **camera_cond** model: its
patch-embed takes **25 input channels** (16 latent + 9 geometric: 3 trajectory-band +
6 camera ray-map). It is *not* loadable with a plain 16-channel config, and the
camera_cond geometry MUST be supplied at inference — from a `{split}_camera_cond.npy`
sidecar (`--conditioning episode`, training parity) or FK-synthesized closed-loop from
a `{split}_joint_actions.npy` chunk (`--conditioning action`). See
`openworld/autoregressive/conditioning/camera_cond.py` and `closed_loop_camera_cond.py`.

Download into `checkpoints/ar_wm/` (the path the docs below assume):

```bash
hf download tennyyyin/open-world-ar-wm wm_student_2view.pt          --local-dir checkpoints/ar_wm
hf download tennyyyin/open-world-ar-wm wm_student_3view_bimanual.pt --local-dir checkpoints/ar_wm
```

Pick the inference config that matches the checkpoint — the two are **not**
interchangeable (different view count, action dimensionality, geometric conditioning,
and state-pred head size). See [configs/inference/README.md](../configs/inference/README.md).

## SVD Bidirectional

Stable Video Diffusion UNet base. `CrtlWorld` is the base world model; `vidwm`
adapts it into a flow-map / shortcut consistency model for few-step inference.

| backbone | model | action-cond modes | platforms | status |
|---|---|---|---|---|
| `CrtlWorld` (vendored) | SVD-UNet-1.5B | via action adapter | DROID, LIBERO | ✅ base bidirectional flow-matching SVD WM |
| `vidwm` | SVD-UNet-1.5B (flow-map distilled) | via action adapter | DROID, LIBERO | ✅ few-step consistency model built on top of `CrtlWorld` |

Training Instructions: [world_model_training/svd.md](world_model_training/svd.md)

## World Model Training

**Autoregressive models**: [world_model_training/autoregressive.md](world_model_training/autoregressive.md)

**SVD models**: [world_model_training/svd.md](world_model_training/svd.md)
