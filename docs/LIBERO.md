# LIBERO support in OpenWorld

This document specifies how LIBERO is plugged into the world-model + policy
stack in this repo, and how it differs from DROID (which the upstream code in
`external/openpi` and the reference training code in
`/n/fs/iromdata/project/Fast-Control-World` were originally built around).

All file references are anchored to `external/openpi` and to this repo.

---

## 1. LIBERO at a glance

LIBERO (Liu et al., NeurIPS 2023) is a procedurally generated MuJoCo /
RoboSuite benchmark with five task suites and a single Franka Panda arm.
We use the bundled fork at
`external/openpi/third_party/libero/libero/libero/`.

### 1.1 Observations

Two RGB cameras + 8-D proprioceptive state.

| Field | Source | Shape | Notes |
|-------|--------|-------|-------|
| `agentview_image` | `obs["agentview_image"]` | 256x256x3 in eval, default 128x128 in the env wrapper | Third-person fixed camera |
| `robot0_eye_in_hand_image` | `obs["robot0_eye_in_hand_image"]` | same | Wrist camera |
| `observation/state` (8-D) | concat of 3 fields | (8,) | EEF pos (3) + axis-angle of EEF quat (3) + gripper qpos (2 finger joints) |

References:
* Camera and resolution: `external/openpi/examples/libero/main.py:18`,
  `external/openpi/third_party/libero/libero/libero/envs/env_wrapper.py:31-36`.
* State concatenation: `examples/libero/main.py:133-139`.
* Both images are rotated 180 degrees before being sent to the policy to
  match training (`main.py:114-122`).

### 1.2 Action space

LIBERO uses RoboSuite's `OSC_POSE` controller
(`third_party/libero/libero/libero/envs/env_wrapper.py:17`).
Each action is **7-D**:

| Index | Meaning |
|-------|---------|
| 0..2 | Delta end-effector position (xyz) |
| 3..5 | Delta end-effector orientation (axis-angle) |
| 6 | Absolute gripper command in {-1 (open), +1 (close)} |

The policy outputs are padded to 8-D internally and sliced back to 7 at
execution time (`src/openpi/policies/libero_policy.py:100`).

### 1.3 Control frequency

20 Hz (`env_wrapper.py:27`, `docs/norm_stats.md:67`).

### 1.4 Episode horizons

Capped per task suite (`examples/libero/main.py:60-71`):

| Suite | Steps | Tasks |
|-------|-------|-------|
| `libero_spatial` | 220 | 10 |
| `libero_object` | 280 | 10 |
| `libero_goal` | 300 | 10 |
| `libero_10` | 520 | 10 |
| `libero_90` | 400 | 90 |

### 1.5 Pi0 zero-shot checkpoint

`pi05_libero` (config in `external/openpi/src/openpi/training/config.py:700-718`):
action_horizon=10, action_dim=8, batch=256, 30k steps, quantile norm.
See also `examples/libero/main.py:29` (`replan_steps=5`).

---

## 2. DROID, side by side

| Field | LIBERO | DROID |
|---|---|---|
| Source | sim (RoboSuite/MuJoCo) | real teleop dataset |
| Robot | Panda (sim) | Panda (real) |
| Cameras | 2: agentview + wrist | 3: 2 exterior + 1 wrist |
| Native image res | 256x256 (eval), 128x128 (default) | 180x320 |
| Policy input res | 224x224 | 224x224 |
| State | 8-D (EEF pos + axis-angle + 2 gripper qpos) | 8-D (7 joint angles + 1 gripper) |
| **Action mode** | **delta EEF pose (OSC_POSE)** + abs gripper, 7-D | **joint velocity** + abs gripper, 8-D |
| Action chunk | 10 (replan every 5) | 10 (open-loop horizon 8) |
| Control freq | **20 Hz** | **15 Hz** |
| Episode max | 220-520 by suite | 600 (eval cap) |
| Norm | quantile (`pi05_libero`) | quantile (`pi05_droid`) |
| Data format on disk | LeRobot-style HDF5 demos | RLDS v1.0.1 |

Why this matters for the world model and adapter:

1. **Different action semantics.** LIBERO actions are *delta EEF poses* in the
   policy's own frame; DROID actions are *joint velocities*. A model trained
   on one will not transfer; both the world model conditioning signal and the
   adapter that translates pi0 outputs into world-model-friendly actions need
   to be retrained for LIBERO.
2. **One fewer camera.** DROID stacks 3 latent video tracks vertically into a
   `(4, 72, 40)` SVD-VAE latent; LIBERO has 2, so the natural stack becomes
   `(4, 48, 40)`. Either we change the spatial layout or zero-pad the
   missing track. We go with `(4, 48, 40)` and document it.
3. **Faster control loop.** 20 Hz vs 15 Hz means the same number of
   action-chunk steps cover less wall-clock time. We keep `pred_step=5`
   actions per WM call (the same 5-frame WM output) but recompute downsample
   ratios accordingly: at 20 Hz with `down_sample=4` the WM runs at 5 Hz, the
   same effective rate as DROID's 15 Hz / 3.
4. **Gripper convention differs.** LIBERO is signed (-1 open / +1 close);
   DROID is unsigned (0 open / 1 close). The dataset loaders normalize this
   to a single convention before feeding the WM.
5. **No FK pass needed in the LIBERO adapter.** The DROID adapter ends with a
   per-step Franka FK call to convert joint positions to Cartesian poses
   (`openworld/policies/openpi_action_adapter.py:101-137`). LIBERO actions are
   already in EEF space; the adapter is a learned integrator from
   (current pose, future delta chunk) to absolute future poses, with no FK.

---

## 3. World-model and adapter design choices for LIBERO

### 3.1 World model

We reuse the same SVD-flow-matching architecture (`vidwm/`) but
re-condition it on LIBERO actions:

* Latent tensor shape per timestep: `(4, 48, 40)` (two cameras stacked
  vertically along H, each contributing 24 latent rows).
* Conditioning action format: **absolute** EEF pose (xyz + axis-angle, 6-D)
  plus normalized gripper command (1-D), giving 7-D conditioning.
  Absolute (rather than delta) is used so the model can be conditioned on a
  consistent global frame, mirroring the DROID setup that conditions on
  `observation.state.cartesian_position` (`Fast-Control-World/dataset/dataset_droid_exp33.py:190-194`).
* Norm stats: percentile (1%, 99%) over each LIBERO sub-suite, stored as
  `dataset_meta_info/libero/stat.json`.
* History indices: `(0, 0, -12, -9, -6, -3)` — same as DROID. (Optional: we
  could shrink for LIBERO's shorter episodes; not changed by default.)

### 3.2 Action adapter

Given that pi0.5_libero outputs *delta EEF poses* and the LIBERO env
consumes the same, an analytical pass-through is possible. But to **mirror
the DROID setup exactly**, we train a learnable MLP `Dynamics` that does:

* Input: current absolute EEF pose `(1, 7)` + future delta-EEF chunk
  `(action_num, 7)` from pi0
* Output: future absolute EEF poses `(action_num, 7)`

Internally this is just SE(3) integration plus a residual; an MLP is more
than expressive enough. The interface mirrors `Dynamics` in
`Fast-Control-World/models/action_adapter/train2.py:38`.
There is no FK step.

The adapter is also responsible for converting the LIBERO gripper convention
(signed) to whatever the WM was trained on (signed in our default).

### 3.3 Layout in this repo

```
docs/LIBERO.md                                        <- this file
dataset_meta_info/libero/stat.json                    <- LIBERO action norm stats
openworld/training/world_model/
    __init__.py
    config.py                                         <- LIBERO WM training config (dataclass)
    dataset.py                                        <- LIBERO WM dataset loader
    train_wm.py                                       <- WM training entry
openworld/training/action_adapter/
    __init__.py
    config.py                                         <- adapter training config
    dataset.py                                        <- (state, delta_chunk, future_state) triples
    model.py                                          <- Dynamics MLP
    train.py                                          <- adapter training entry
openworld/policies/libero_action_adapter.py           <- inference-time adapter wrapper
scripts/preprocess_libero_for_wm.py                   <- HDF5 demos -> latent .pt + annotation .json
scripts/train_libero_wm.py                            <- thin wrapper around openworld.training.world_model.train_wm
scripts/train_libero_adapter.py                       <- thin wrapper around openworld.training.action_adapter.train
configs/training/libero_wm.py                         <- python dataclass override (preferred)
configs/training/libero_wm.yaml                       <- yaml mirror
configs/training/libero_adapter.yaml
configs/world_models/libero.yaml
configs/policies/openpi_libero.yaml
configs/evaluation/libero_pi05.yaml
bash_scripts/train_libero_wm.sh
bash_scripts/train_libero_adapter.sh
bash_scripts/eval_libero_wm.sh
```

`openworld/policies/openpi_policy.py` already supports loading any openpi
checkpoint by path; the LIBERO config just points it at `pi05_libero` and
swaps the action adapter from `OpenPIActionAdapter` to
`OpenPILiberoActionAdapter`.

---

## 4. Hyperparameters (mirrored from Fast-Control-World)

These come from `Fast-Control-World/config_flow_map.py` and are kept the
same for LIBERO unless noted:

| Name | DROID value | LIBERO value | Note |
|---|---|---|---|
| learning_rate | 1e-5 | 1e-5 | |
| train_batch_size | 1 (per device) | 1 | scale with `accelerate` |
| gradient_accumulation_steps | 1 | 1 | |
| mixed_precision | fp16 | fp16 | |
| max_train_steps | 500_000 | 500_000 | |
| checkpointing_steps | 20_000 | 20_000 | |
| validation_steps | 2_500 | 2_500 | |
| max_grad_norm | 1.0 | 1.0 | |
| num_frames | 5 | 5 | future-frame window |
| num_history | 6 | 6 | history length |
| action_dim | 7 | 7 | (6 EEF + 1 gripper) |
| down_sample | 3 (15->5 Hz) | **4 (20->5 Hz)** | match effective WM rate |
| guidance_scale | 2.0 | 2.0 | |
| flow_map_type | flow_matching | flow_matching | default; switch to `flow_map`/`shortcut` for distillation |
| height x width (latent canvas) | 192 x 320 | 192 x 320 | image canvas; latent is /8 |
| view stacking (latent H) | 72 (3x24) | **48 (2x24)** | one fewer camera |

Adapter hyperparameters
(`Fast-Control-World/models/action_adapter/train2.py:289-314`):

| Name | DROID | LIBERO | Note |
|---|---|---|---|
| hidden_size | 512 | 512 | |
| action_num | 15 | 15 | adapter chunk length |
| action_dim | 7 | 7 | |
| optimizer | Adam(lr=1e-4) | Adam(lr=1e-4) | |
| epochs | 10 | 10 | |
| batch_size | 128 | 128 | |
| input | (current joint, joint_vel chunk) | (current EEF pose, delta-EEF chunk) | |
| output | future joints, then FK | future EEF poses | no FK |
| target | normalized joint deltas | normalized EEF deltas | |

---

## 5. End-to-end usage

### 5.1 Preprocess LIBERO demos to VAE latents (one-time)

```bash
python scripts/preprocess_libero_for_wm.py \
    --libero_root data/raw_libero \
    --task_suites libero_spatial libero_object libero_goal libero_10 libero_90 \
    --output_root data/libero_processed \
    --svd_path external/stable-video-diffusion-img2vid \
    --device cuda
```

This writes:

```
data/libero_processed/<suite>/annotation/{train,val}/<episode_id>.json
data/libero_processed/<suite>/latent_videos/{agentview,wrist}/<episode_id>.pt
```

with one annotation JSON per episode containing:
`observation.state.cartesian_position` (= EEF pose),
`observation.state.gripper_position`, `texts` (language goal).

### 5.2 Compute normalization stats

```bash
python scripts/compute_libero_norm_stats.py \
    --processed_root data/libero_processed \
    --suite libero_10 \
    --output dataset_meta_info/libero/stat.json
```

### 5.3 Train the world model

```bash
bash bash_scripts/train_libero_wm.sh
```

Equivalent to:

```bash
accelerate launch openworld/training/world_model/train_wm.py \
    --config configs/training/libero_wm.py
```

### 5.4 Train the action adapter

```bash
bash bash_scripts/train_libero_adapter.sh
```

### 5.5 Evaluate (pi0.5 zero-shot policy in the trained WM)

```bash
python scripts/run_evaluation.py --config configs/evaluation/libero_pi05.yaml
```

This uses the existing `Evaluator` runner with the LIBERO world-model and
adapter wired in.

---

## 6. Open items / non-defaults

* **Action conditioning frame.** We condition the WM on absolute EEF pose
  for symmetry with the DROID setup. If LIBERO's EEF pose distribution turns
  out to be hard to learn (e.g. many similar poses across diverse scenes),
  switching to delta-pose conditioning is a one-line change in
  `openworld/training/world_model/dataset.py`.
* **Camera count.** Default is 2 cameras (agentview + wrist). If you add a
  third (e.g. `frontview` or a second exterior), bump `num_cams` in the
  config and re-encode the dataset; the latent canvas height becomes
  `num_cams * 24`.
* **Per-suite training.** By default we mix all 5 suites with equal
  probability. Per-suite specialization is supported via the `dataset_names`
  / `prob` fields in the WM config.
