# World Model Training

Two-stage workflow: train a **flow-matching teacher** for fidelity, then
**distill** it into a **shortcut student** for few-step inference.

The two stages share the same architecture (SVD UNet + action encoder) and
the same data; they differ only in the loss objective and a couple of
config knobs. The student is initialized from the teacher checkpoint, so
distillation needs ~20% of teacher training steps.

---

## Stage 1 — Flow-matching teacher

Standard rectified-flow / flow-matching objective. The model learns the
velocity field `v(x_t, t)`; at inference it is integrated with Euler over
many steps (`num_inference_steps≈50`).

Config: `configs/training/libero_wm.py`. Key settings:

```python
flow_map_type="flow_matching"
distance_conditioning=False        # UNet has no dt head
learning_rate=1e-5
max_train_steps=500_000
```

Launch:

```bash
uv run accelerate launch \
    --num_processes 4 \
    --mixed_precision fp16 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/libero_wm.py
```

At eval time, set `world_model.params.flow_map_type: flow_matching` in the
evaluation YAML (e.g. `configs/evaluation/libero_pi05.yaml`).

---

## Stage 2 — Shortcut distillation

Trains the model to take a single jump of size `dt` instead of integrating
many small steps. The UNet gains a distance head conditioned on `dt_base`,
so few-step (1/2/4/8) inference is possible.

Config: `configs/training/libero_wm_shortcut.py`. Key differences vs. the
teacher:

```python
ckpt_path="checkpoints/wm_libero/libero_flow_matching_v0/checkpoint-200000.pt"
flow_map_type="shortcut"
distance_conditioning=True         # UNet now receives dt_base
learning_rate=5e-6
max_train_steps=100_000
test_num_inference_steps=(1, 2, 4, 8)
```

Update `ckpt_path` to point at your trained teacher checkpoint, then
launch:

```bash
uv run accelerate launch \
    --num_processes 4 \
    --mixed_precision fp16 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/libero_wm_shortcut.py
```

At eval time, set `world_model.params.flow_map_type: shortcut` and drop
`num_inference_steps` to 1, 2, 4, or 8.

---

## Why two stages

`flow_matching`, `shortcut`, and `flow_map` are different training
objectives — same architecture, different loss heads, so the weights are
**not** interchangeable. You cannot use a flow-matching checkpoint as a
shortcut model (or vice versa). Concretely:

* The flow-matching teacher does not have the UNet `distance` head that
  shortcut/flow_map need.
* Even if the architecture matched, the objectives are different — a
  flow-matching model is trained to predict the velocity at a single
  timestep, not a multi-step jump.

Training shortcut from scratch is possible (set `ckpt_path=None`,
`max_train_steps=500_000`), but converges much slower than distilling
from a flow-matching teacher. The recommended path is teacher → student.

---

## Inference modes supported

`VidWMDiffusionPipeline` (`vidwm/video_models/vidwm_diffusion.py`)
dispatches on `flow_map_type`:

| `flow_map_type` | Solver | Typical `num_inference_steps` | Requires `distance_conditioning` |
|-----------------|--------|------------------------------|----------------------------------|
| `flow_matching` | `euler_solver` | 25–50 | No |
| `shortcut`      | `short_cut_solver` | 1–8 | Yes |
| `flow_map`      | `flow_map_solver` | 1–8 | Yes |

The `flow_map_type` used at eval **must** match what the checkpoint was
trained with.
