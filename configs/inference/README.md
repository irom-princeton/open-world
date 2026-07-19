# Inference configs

Inference-only `ARWMArgs` configs for loading a trained AR world-model student and
rolling it out — teleoperation (`scripts/interactive_ar.py`), open-loop replay
(`openworld/autoregressive/infer/replay.py`), and eval. They are **not** for
training (no mid-training / distillation recipe is intended to be run from here).

Each config derives from `configs/training/ar_wan_droid.py` and pins the knobs that
change model/data geometry at inference time.

## Published student checkpoints

The two checkpoints on Hugging Face
([`tennyyyin/open-world-ar-wm`](https://huggingface.co/tennyyyin/open-world-ar-wm),
see [docs/MODELS.md](../../docs/MODELS.md) for download) each have a matching config
that reproduces their exact geometry, so the weights load with no missing/unexpected
keys. These are **undistilled students** — sample with the many-step preview schedule
(do **not** pass `--distilled`).

| Config | checkpoint | views | `action_dim` | geom. cond | state-pred head | block geometry |
| --- | --- | --- | --- | --- | --- | --- |
| `ar_wan_student_2view.py` | `wm_student_2view.pt` | 2 (`num_cams=2`) | 7 (cartesian) | — | 8 | fpb 1, hist 4, roll 12 |
| `ar_wan_student_3view_bimanual.py` | `wm_student_3view_bimanual.pt` | 3 (`view_indices=(1,2,3)`) | 20 (cartesian, bimanual) | camera_cond (9-ch → 25 in) | 16 | fpb 1, hist 4, roll 12 |

```bash
python scripts/interactive_ar.py \
    --config configs/inference/ar_wan_student_2view.py \
    --checkpoint checkpoints/ar_wm/wm_student_2view.pt
```

The `ar_wan_student_3view_bimanual.py` config is a **camera_cond** model
(`camera_cond=True`, `camera_cond_channels=9`): the patch-embed widens 16 → 25 input
channels, so the geometry sidecar must be present at inference. Replay it with an
explicit conditioning source:

```bash
python scripts/replay_ar.py \
    --config configs/inference/ar_wan_student_3view_bimanual.py \
    --checkpoint checkpoints/ar_wm/wm_student_3view_bimanual.pt \
    --latent-root <bimanual_latents> --split val --conditioning episode
```

## Legacy DROID configs

These pin only `num_cams` / `action_space` (fpb-2 geometry from `ar_wan_droid.py`),
for older DROID checkpoints:

| Config | `num_cams` | `action_space` | action dim / stats |
| --- | --- | --- | --- |
| `ar_wan_droid_2view_cartesian.py` | 2 | `cartesian` | 7 / `stats.json` |
| `ar_wan_droid_2view_jointpos.py`  | 2 | `joint_pos` | 8 / `stats_joint.json` |
| `ar_wan_droid_3view_cartesian.py` | 3 | `cartesian` | 7 / `stats.json` |
| `ar_wan_droid_3view_jointpos.py`  | 3 | `joint_pos` | 8 / `stats_joint.json` |

The trained weights come from the `--checkpoint` flag, not from the config. Pick the
config whose geometry matches how the checkpoint was trained.

Notes:
- `num_cams` / `view_indices` only change how many (and which) height-stacked views the
  data path feeds, not the model's parameters. `view_indices` pins an exact stored-view
  subset (needed for the bimanual checkpoint's scene+2-wrist layout); plain `num_cams`
  keeps the wrist + samples the sides.
- `action_space` / `action_dim` *must* match the checkpoint's training conditioning:
  7-dim cartesian, 8-dim joint, and 20-dim bimanual are not interchangeable.
- `state_pred` / `state_pred_dim` must match too — a checkpoint trained with the
  auxiliary state-prediction head carries `backbone.state_head.*` weights, so the model
  has to be built with the head (of the right dim) or the load fails. The head is not
  used by the forward-only rollout.
- `camera_cond` / `camera_cond_channels` must match — a camera_cond checkpoint's
  `patch_embedding.weight` has `16 + camera_cond_channels` input channels (25 for the
  9-channel bimanual student), so the config has to widen the patch-embed to the same
  width or the conv weight has the wrong shape. Camera_cond also requires the geometry
  input at inference: a `{split}_camera_cond.npy` sidecar (`--conditioning episode`) or
  a `{split}_joint_actions.npy` chunk FK-synthesized closed-loop (`--conditioning action`).
- The inherited training-only fields (learning rate, distillation schedule,
  `student_init_ckpt` / `teacher_ckpt` paths) are unused by a forward-only rollout.
