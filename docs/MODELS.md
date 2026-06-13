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
| `wan_1_3b` | Wan2.1-T2V-1.3B | `cross_attn_aligned`, `adaln` | DROID | ✅ training<br>❌ few-step<br>❌ checkpoint |
| `cosmos_predict2_2b` | Cosmos-Predict2-2B | `cross_attn` only | DROID | ❌ training<br>❌ few-step<br>❌ checkpoint |

Training Instructions: [world_model_training/autoregressive.md](world_model_training/autoregressive.md)

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
