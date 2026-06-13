# World Model Training

## Stage 1 — Flow-matching teacher

Standard rectified-flow / flow-matching objective. The model learns the
velocity field `v(x_t, t)`; at inference it is integrated with Euler over
many steps (`num_inference_steps≈50`).

```bash
uv run accelerate launch \
    --num_processes 8 \
    --mixed_precision fp16 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/libero_wm.py
```

Set `world_model.params.flow_map_type: flow_matching` during evaluation.

---

## Stage 2 — Shortcut distillation

Trains the model to take a single jump of size `dt` instead of integrating
many small steps. The UNet gains a distance head conditioned on `dt_base`,
so few-step (1/2/4/8) inference is possible.

Config: `configs/training/libero_wm_shortcut.py`. Key differences vs. the
teacher:

```python
ckpt_path="scheckpoints/wm/libero/checkpoint-30000.pt"
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
